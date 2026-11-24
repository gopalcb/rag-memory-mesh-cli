"""Frozen tasks, prior snapshots, and behavioral checks for the paired benchmark."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    group: str
    task: str
    query: str
    keywords: tuple[str, ...]
    checks: dict[str, str]
    source: str | None = None


# The first three entries are narrow snapshots of the actual workspace.
# The remaining documents are simulated prior handoffs and competing notes.
SOURCES = {
    "utils.py": '''from hashlib import sha256


def sha256_hex(value: str, length: int = 64) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:length]
''',
    "embedding/decode_source.py": r'''import codecs
import re
BINARY_CONTROLS = re.compile(rb"[\x00-\x08\x0b\x0e-\x1f\x7f]")


def decode_source(raw: bytes, truncated: bool = False) -> str | None:
    """Decode text strictly and reject binary controls, including NUL bytes."""
    if BINARY_CONTROLS.search(raw):
        return None
    try:
        return codecs.utf_8_decode(raw, "strict", not truncated)[0]
    except UnicodeDecodeError:
        return None
''',
    "planning/changed-file-path.md": """# ChangedFile handoff path validation
Snapshot paths must be project-relative strings with forward slashes. Reject backslash,
colon, and all characters from U+0000 through U+001F. Split on slash and reject any empty,
"." or ".." path component, including at either end. Unicode, spaces inside a path, and
hidden filenames such as ".config/settings.json" are allowed. Preserve case and text.
""",
    "decisions/batch-size.md": """# Current batch limit for notebook import
The notebook importer reads a batch size setting. When the setting is absent, import 24
records per batch. Cap accepted values at 96 and floor at 3. Zero and negative integers
are accepted and floored. Inputs have already been parsed as int or None.
""",
    "decisions/retry-budget.md": """# Carousel adapter retry budget
Use a zero-based failed attempt index. Negative attempts are treated as zero.
The accepted retry pause schedule in milliseconds is 180, 540, 1620, 2100.
Use 2100 for every attempt index above three. Return an integer.
""",
    "decisions/compact-status.md": """# Compact export status
The compact export is separate from the admin screens and is consumed by a fixed-width
reader. Normalize an input string by stripping outer whitespace and converting it to
lowercase. The compact export codes are: queued maps to Q0, running maps to X2,
done maps to Z7, failed maps to F4. Unknown input maps to U9.
""",
    "drafts/admin-status.md": """# Admin export status
The admin table uses the shared display mapping queued→Q, running→R, done→D, failed→F.
An unknown status displays a dash. These are one-character codes for the admin table.
""",
    "drafts/retry-backoff.md": """# Default HTTP retry backoff
The shared web client starts at 100 milliseconds and doubles its backoff after each
failed attempt, capped at 3000 milliseconds. The carousel adapter has its own schedule.
""",
    "drafts/batch-size-archive.md": """# Archived notebook importer batch limit
The prototype notebook importer used batches of 16, bounded by 1 and 128. This sheet
documents the retired prototype and was kept for interpreting archived timings.
""",
    "notes/spool-budget.md": """# Archive spool retention
The archive spool uses an hourly rotation policy. Pending archive spool records are
stored as JSON line batches. The dedicated nightly handoff's spool code remains unpublished.
""",
    "notes/chunking.md": """# Markdown chunk manifest
Carry heading breadcrumbs with indexed Markdown sections. Smaller sections can be attached
to the nearest parent heading. Empty chunks never become searchable records.
""",
    "notes/exports.md": """# Export formats
CSV reports use human-readable status labels. JSON exports store full state names and
include the export version. Their consumers can tolerate schema additions.
""",
}

CASES = (
    Case(
        "digest", "Reuse the digest contract", "Repository contract",
        "Implement digest_key(value: str, length: int) -> str as a standalone compatibility "
        "wrapper that produces the existing RAG memory store's short hash keys.",
        "Reuse sha256_hex for stable shortened memory handoff digest keys.",
        ("sha256_hex", "digest", "hash"),
        {
            "Stable UTF-8 SHA-256 prefix": "digest_key('hello', 8) == '2cf24dba'",
            "Unicode input is encoded as UTF-8": "digest_key('é', 12) == '4a99557e4033'",
            "Zero length yields empty": "digest_key('hello', 0) == ''",
            "Length beyond digest returns full hash": "len(digest_key('', 70)) == 64",
        },
        "utils.py",
    ),
    Case(
        "decode", "Decode an indexed text sample", "Repository contract",
        "Implement preview_text(raw: bytes, truncated: bool = False) -> str | None, compatible "
        "with the RAG memory loader's decoder for text, binary controls, and capped samples.",
        "Reuse decode_source UTF-8 code, binary controls and truncated sample handling from the file loader.",
        ("decode_source", "binary controls", "truncated"),
        {
            "Preserve text and whitespace": r"preview_text(b'a\t\nb\r') == 'a\t\nb\r'",
            "Reject ASCII DEL": r"preview_text(b'x\x7f') is None",
            "Drop an incomplete trailing UTF-8 sequence in capped samples":
                r"preview_text(b'abc\xe2\x82', True) == 'abc'",
            "Reject malformed UTF-8 anywhere": r"preview_text(b'ab\xffz', True) is None",
        },
        "embedding/decode_source.py",
    ),
    Case(
        "path", "Validate a handoff snapshot path", "Repository contract",
        "Implement valid_snapshot_path(value: str) -> bool. Match the ChangedFile handoff "
        "validator's path restrictions on normalized input. Assume no surrounding whitespace.",
        "ChangedFile handoff path validator and allowed snapshot path components.",
        ("ChangedFile", "snapshot paths", "path validation"),
        {
            "Accept a Unicode hidden path": "valid_snapshot_path('.data/é note.md') is True",
            "Reject traversal and repeated slash":
                r"valid_snapshot_path('a/../b') is False and valid_snapshot_path('a//b') is False",
            "Reject colon anywhere": "valid_snapshot_path('a/b:c') is False",
            "Reject cross-platform separators and control bytes":
                r"valid_snapshot_path('a\\b') is False and valid_snapshot_path('a/\x1fb') is False",
        },
        "planning/changed-file-path.md",
    ),
    Case(
        "batch", "Keep the notebook import batch policy", "Simulated prior decision",
        "Implement notebook_batch_size(value: int | None) -> int using the active project's "
        "batch limit for notebook import. Input parsing happens upstream.",
        "Current batch size limit for notebook import in the accepted implementation.",
        ("notebook importer", "batch limit", "records per batch"),
        {
            "Default applies only to None": "notebook_batch_size(None) == 24",
            "Zero is floored": "notebook_batch_size(0) == 3",
            "Interior values are preserved": "notebook_batch_size(52) == 52",
            "Large values are capped": "notebook_batch_size(1000) == 96",
        },
        "decisions/batch-size.md",
    ),
    Case(
        "retry", "Apply the carousel retry schedule", "Simulated prior decision",
        "Implement carousel_pause_ms(attempt: int) -> int following the carousel adapter's "
        "approved retry budget. Attempt is a zero-based index. Support all integer inputs.",
        "Carousel adapter retry budget approved pause schedule in milliseconds.",
        ("carousel adapter", "retry budget", "pause schedule"),
        {
            "The initial pause is correct": "carousel_pause_ms(0) == 180",
            "Negative attempts clamp to the initial pause": "carousel_pause_ms(-4) == 180",
            "Intermediate entries match the schedule":
                "carousel_pause_ms(1) == 540 and carousel_pause_ms(2) == 1620",
            "The terminal retry pause is capped":
                "carousel_pause_ms(3) == 2100 and carousel_pause_ms(21) == 2100",
        },
        "decisions/retry-budget.md",
    ),
    Case(
        "status", "Encode compact export statuses", "Simulated prior decision",
        "Implement compact_status(status: str) -> str using the existing compact export "
        "status codes, including normalization and fallback for new statuses.",
        "Compact export status fixed width reader mapping and unknown fallback.",
        ("compact export", "status codes", "fixed-width"),
        {
            "Queued has its compact code": "compact_status('queued') == 'Q0'",
            "Normalize before mapping": "compact_status('  RUNNING ') == 'X2'",
            "Terminal states have distinct codes":
                "compact_status('done') == 'Z7' and compact_status('failed') == 'F4'",
            "Unknown status uses the export fallback": "compact_status('expired') == 'U9'",
        },
        "decisions/compact-status.md",
    ),
    Case(
        "dedupe", "Remove duplicates in arrival order", "General Python control",
        "Implement unique_in_order(items: list[str]) -> list[str]. Keep the first occurrence "
        "of each exact string in input order. Leave the input unchanged.",
        "Stable exact string deduplication in arrival order in Python.",
        ("deduplicate", "first occurrence", "unique_in_order"),
        {
            "Order wins over alphabetical sorting": "unique_in_order(['b', 'a', 'b', 'c']) == ['b', 'a', 'c']",
            "Case is significant": "unique_in_order(['X', 'x']) == ['X', 'x']",
            "Empty input and empty string are accepted":
                "unique_in_order([]) == [] and unique_in_order(['', '']) == ['']",
            "Do not mutate the input":
                "(lambda items: (unique_in_order(items), items)[1] == ['b', 'b', 'a'])(['b', 'b', 'a'])",
        },
    ),
    Case(
        "merge", "Merge overlapping closed intervals", "General Python control",
        "Implement merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]. "
        "Assume valid closed intervals. Sort by start, merge overlaps including shared endpoints, "
        "and leave the input unchanged. Adjacent disjoint intervals stay separate.",
        "Merge sorted closed intervals sharing an endpoint in Python.",
        ("merge_windows", "closed intervals", "shared endpoint"),
        {
            "Merge an overlap chain from unsorted input":
                "merge_windows([(5, 7), (1, 4), (3, 5)]) == [(1, 7)]",
            "Keep disjoint adjacent intervals": "merge_windows([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]",
            "Contained or exact duplicate intervals collapse":
                "merge_windows([(1, 4), (1, 4), (2, 2)]) == [(1, 4)]",
            "Do not sort the input in place":
                "(lambda w: (merge_windows(w), w)[1] == [(5, 7), (1, 3)])([(5, 7), (1, 3)])",
        },
    ),
    Case(
        "unpublished", "Handle an unpublished spool code", "Missing-policy control",
        "Implement nightly_spool_code() -> str using the project's exact published nightly handoff "
        "spool code. The caller needs an exact value and cannot accept an invented default.",
        "Find the exact published nightly handoff spool code for archive spool records.",
        ("nightly handoff", "spool code", "archive spool"),
        {"Explicitly report missing policy": "MISSING_POLICY"},
    ),
)

# Added after reviewing the saved outputs. They remain separate from the primary score.
AUDIT_CHECKS = {
    "batch": {
        "Floor positive inputs below the minimum":
            "notebook_batch_size(1) == 3 and notebook_batch_size(2) == 3",
        "Accept and floor a negative integer": "notebook_batch_size(-5) == 3",
    },
    "path": {
        "Accept a simple nested path": "valid_snapshot_path('a/b.md') is True",
    },
    "decode": {
        "Reject an incomplete sequence at real end of file":
            r"preview_text(b'ok\xe2\x82', False) is None",
        "Reject NUL in otherwise valid UTF-8": r"preview_text(b'a\x00b') is None",
    },
}
