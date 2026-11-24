import asyncio
import ipaddress
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilterLXML
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from config import merge_options, settings
from crawler.models import BatchCrawlReport, CrawlOptions, CrawlTarget, PageOutcome
from embedding.pipeline import EmbeddingPipeline
from store.job_store import JobStore
from store.markdown_store import MarkdownVaultStore
from utils import sha256_hex


async def block_browser_requests(page, **kwargs):
    async def abort(route):
        await route.abort()

    await page.route("http*://**/*", abort)
    return page


def default_crawler_factory() -> AsyncWebCrawler:
    os.environ.setdefault("CRAWL4AI_NO_CLOUD_NOTICE", "1")
    browser = AsyncWebCrawler(
        config=BrowserConfig(headless=True, verbose=False, java_script_enabled=False),
        base_directory=str(settings.storage.base_dir.resolve()),
    )
    browser.crawler_strategy.set_hook("on_page_context_created", block_browser_requests)
    return browser


class CrawlFleet:
    def __init__(
        self,
        vault: MarkdownVaultStore | None = None,
        jobs: JobStore | None = None,
        pipeline: EmbeddingPipeline | None = None,
        crawler_factory: Callable[[], Any] | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.vault = vault or MarkdownVaultStore()
        self.jobs = jobs or JobStore()
        self.pipeline = pipeline
        self.crawler_factory = crawler_factory or default_crawler_factory
        self.http_transport = http_transport

    @staticmethod
    def normalize_targets(
        targets: list[CrawlTarget | dict[str, Any] | str], default_topic: str = "general",
    ) -> list[CrawlTarget]:
        normalized: dict[str, CrawlTarget] = {}
        for item in targets:
            target = (
                CrawlTarget(url=item, topic=default_topic)
                if isinstance(item, str)
                else CrawlTarget.model_validate(item)
            )
            normalized.setdefault(target.url_str, target)
        if len(normalized) > 20:
            raise ValueError("A crawl batch supports up to 20 source URLs")
        return list(normalized.values())

    @staticmethod
    def extract_title(markdown_text: str, fallback_url: str) -> str:
        for line in markdown_text.splitlines():
            if line.strip().startswith("# "):
                return line.strip()[2:].strip()[:120]
        return urlsplit(fallback_url).path.rstrip("/").rsplit("/", 1)[-1] or urlsplit(fallback_url).hostname or "web-page"

    @staticmethod
    async def validate_url(url: str) -> None:
        parts = urlsplit(url)
        host = parts.hostname
        if (
            parts.scheme != "https" or host is None or parts.port not in (None, 443)
            or parts.username is not None or parts.password is not None
            or host not in settings.crawler.allowed_domains
        ):
            raise ValueError("Crawl URLs must use HTTPS and a configured documentation host")
        addresses = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(address[4][0]).is_global for address in addresses):
            raise ValueError("Documentation hosts must resolve to public IP addresses")

    async def fetch_html(self, client: httpx.AsyncClient, url: str) -> str:
        for _ in range(5):
            await self.validate_url(url)
            async with client.stream("GET", url) as result:
                if result.is_redirect:
                    url = urljoin(url, result.headers["location"])
                    continue
                result.raise_for_status()
                if "html" not in result.headers.get("content-type", "").lower():
                    raise ValueError("Expected an HTML documentation page")
                data = bytearray()
                async for part in result.aiter_bytes():
                    data.extend(part)
                    if len(data) > 2_000_000:
                        raise ValueError("Documentation exceeds the 2 MB size limit")
                return data.decode(result.encoding or "utf-8", errors="replace")
        raise ValueError("Too many redirects")

    async def crawl_single(
        self, crawler: Any, client: httpx.AsyncClient, target: CrawlTarget,
        semaphore: asyncio.Semaphore, run_cfg: CrawlerRunConfig,
    ) -> PageOutcome:
        url = target.url_str
        async with semaphore:
            try:
                async with asyncio.timeout(run_cfg.page_timeout / 1000):
                    html = await self.fetch_html(client, url)
                    result = await crawler.arun(url=f"raw:{html}", config=run_cfg.clone(base_url=url))
                if not result.success or not result.markdown:
                    return PageOutcome(url=url, topic=target.topic, status="failed", error="Could not extract documentation")
                markdown = result.markdown
                clean_md = (getattr(markdown, "fit_markdown", None)
                            or getattr(markdown, "raw_markdown", str(markdown))).strip()
                if len(clean_md) < 50:
                    raise ValueError("Extracted documentation is too short")
                title = self.extract_title(clean_md, url)
                previous = await asyncio.to_thread(self.jobs.indexed_url_path, url)
                path = await asyncio.to_thread(
                    self.vault.save_markdown, clean_md, title, url, target.topic,
                    {"tags": ", ".join(target.tags)} if target.tags else None,
                    existing_path=previous,
                )
                return PageOutcome(
                    url=url, topic=target.topic, status="saved", markdown_path=str(path),
                    title=title, char_count=len(clean_md),
                )
            except TimeoutError:
                return PageOutcome(url=url, topic=target.topic, status="failed", error="Crawl timed out")
            except Exception as exc:
                return PageOutcome(url=url, topic=target.topic, status="failed", error=str(exc))

    async def crawl_urls(
        self, targets: list[CrawlTarget | dict[str, Any] | str],
        options: CrawlOptions | dict | None = None, default_topic: str = "general",
    ) -> BatchCrawlReport:
        opts = merge_options(CrawlOptions(), options)
        cfg = settings.crawler
        ttl = opts.cache_ttl_hours if opts.cache_ttl_hours is not None else cfg.cache_ttl_hours
        report = BatchCrawlReport()
        pending = []
        for target in self.normalize_targets(targets, default_topic):
            fresh = not opts.force_refresh and await asyncio.to_thread(self.jobs.indexed_url_path, target.url_str, ttl)
            if fresh:
                report.outcomes.append(PageOutcome(url=target.url_str, topic=target.topic, status="skipped_fresh"))
            else:
                pending.append(target)
        if not pending:
            return report

        pruning = opts.pruning_threshold if opts.pruning_threshold is not None else cfg.pruning_threshold
        run_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS, page_timeout=opts.page_timeout_ms, verbose=False,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilterLXML(threshold=pruning, threshold_type="fixed")
            ),
        )
        semaphore = asyncio.Semaphore(opts.max_concurrency or cfg.max_concurrency)
        async with httpx.AsyncClient(transport=self.http_transport, trust_env=False) as client:
            async with self.crawler_factory() as crawler:
                report.outcomes.extend(await asyncio.gather(*[
                    self.crawl_single(crawler, client, target, semaphore, run_cfg) for target in pending
                ]))

        if opts.auto_embed:
            for outcome in report.outcomes:
                if outcome.status != "saved" or outcome.markdown_path is None:
                    continue
                try:
                    path = Path(outcome.markdown_path)
                    self.pipeline = self.pipeline or EmbeddingPipeline()
                    stats = await asyncio.to_thread(self.pipeline.ingest, path)
                    old_path = await asyncio.to_thread(self.jobs.indexed_url_path, outcome.url)
                    vault_root = self.vault.vault_dir.resolve()
                    if old_path and old_path.resolve() != path.resolve() and old_path.resolve().is_relative_to(vault_root):
                        await asyncio.to_thread(self.pipeline.remove_tracked, old_path)
                        old_path.unlink(missing_ok=True)
                    content_hash = sha256_hex(path.read_text(encoding="utf-8"))
                    await asyncio.to_thread(
                        self.jobs.record_crawled_url, outcome.url, outcome.topic, content_hash, path,
                    )
                    report.chunks_indexed += stats.total_chunks_upserted
                    outcome.status = "indexed"
                except Exception as exc:
                    outcome.status = "failed"
                    outcome.error = str(exc)
        return report
