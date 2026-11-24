"""
Embedding Model Factory (Cached Singleton)
==========================================
Purpose:
    Instantiates and caches either a local SentenceTransformer model or a remote
    OpenAI-compatible embedding client using defaults from `default_config.yaml`,
    accepting a single `EmbeddingClientOptions` Pydantic model or dict for overrides.

Call Hierarchy:
    ChromaStoreManager.__init__() ──► get_embedding_client(options)
                                            ├── "local" ──► HuggingFaceEmbeddings
                                            └── "api"   ──► OpenAIEmbeddings
"""

from functools import lru_cache
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import EmbeddingClientOptions, merge_options, settings
from model_policy import require_external_permission


@lru_cache(maxsize=8)
def build_cached_embedding_client(
    provider: str,
    model_name: str,
    device: str,
    base_url: str,
    api_key: str,
    normalize_embeddings: bool,
    check_ctx_length: bool,
    revision: str | None = None,
) -> Embeddings:
    """Hashable cache wrapper that instantiates the underlying LangChain Embeddings client."""
    if provider == "api":
        return OpenAIEmbeddings(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            check_embedding_ctx_length=check_ctx_length,
            http_client=httpx.Client(trust_env=False, timeout=60),
            http_async_client=httpx.AsyncClient(trust_env=False, timeout=60),
        )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, **({"revision": revision} if revision else {})},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )


def get_embedding_client(
    options: EmbeddingClientOptions | dict[str, Any] | None = None,
) -> Embeddings:
    """
    Resolve `options` against `settings.embedding` and return a cached `Embeddings` client.

    Args:
        options: Optional `EmbeddingClientOptions` model or dict overriding provider,
            model_name, device, base_url, api_key, or normalize_embeddings.

    Returns:
        A cached, ready-to-encode LangChain `Embeddings` instance.
    """
    opts = merge_options(EmbeddingClientOptions(), options)
    provider = opts.provider or settings.embedding.provider

    if provider == "api":
        api_cfg = merge_options(settings.embedding.api, opts)
        require_external_permission("API embeddings", api_cfg.base_url)
        return build_cached_embedding_client(
            provider="api",
            model_name=api_cfg.model_name,
            device="",
            base_url=api_cfg.base_url,
            api_key=api_cfg.api_key,
            normalize_embeddings=True,
            check_ctx_length=api_cfg.check_ctx_length,
        )

    defaults = settings.embedding.local
    if opts.model_name and opts.model_name != defaults.model_name:
        defaults = defaults.model_copy(update={"revision": None})
    local_cfg = merge_options(defaults, opts)
    return build_cached_embedding_client(
        provider="local",
        model_name=local_cfg.model_name,
        device=local_cfg.device,
        base_url="",
        api_key="",
        normalize_embeddings=local_cfg.normalize_embeddings,
        check_ctx_length=False,
        revision=local_cfg.effective_revision,
    )
