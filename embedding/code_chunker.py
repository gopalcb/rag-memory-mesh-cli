"""
Python AST Code Chunker & Dependency Edge Extractor
===================================================
Purpose:
    Parses Python source files using Python's built-in `ast` module to extract
    self-contained top-level functions, async functions, and classes. Prepends
    module-level imports and semantic intent headers so retrieved snippets are
    runnable, and extracts directed dependency edges (`calls`, `imports`, `inherits`)
    for `CodeGraphStore`.

Call Hierarchy:
    EmbeddingPipeline.ingest()
          │
          ▼
    PythonAstChunker.chunk(record, max_chars)
          ├── ast.parse(record.content)
          │     └── [SyntaxError] ──► PythonAstChunker.build_documents(..., "module_block")
          ├── PythonAstChunker.extract_imports()
          └── For each public FunctionDef / AsyncFunctionDef / ClassDef:
                ├── PythonAstChunker.extract_edges()     -> [(target_symbol, relation), ...]
                └── PythonAstChunker.build_documents()   -> list[Document] + graph_edges
"""

import ast
import builtins
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from config import settings
from embedding.loaders import RawFileRecord
from utils import sha256_hex

_BUILTINS = set(dir(builtins))


class PythonAstChunker:
    """Splits Python modules along AST symbol boundaries and extracts dependency edges."""

    def __init__(self, max_chars: int | None = None, overlap_chars: int = 150):
        """
        Initialize the AST chunker with defaults from `default_config.yaml`.

        Args:
            max_chars: Maximum character length per chunk before splitting oversized classes.
            overlap_chars: Character overlap used when an oversized class must be split.
        """
        self.max_chars = max_chars if max_chars is not None else settings.chunking.code_max_chars
        self.overlap_chars = overlap_chars

    @staticmethod
    def extract_imports(tree: ast.Module, lines: list[str]) -> tuple[str, list[str]]:
        """
        Collect all top-level `import` and `from ... import` statements in a module.

        Args:
            tree: Parsed `ast.Module` node.
            lines: Source file split into lines.

        Returns:
            Tuple of `(formatted_import_block, imported_symbol_names)`.
        """
        import_lines, symbols = [], []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node.end_lineno:
                import_lines.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
                for alias in node.names:
                    if alias.name != "*":
                        symbols.append(alias.asname or alias.name.split(".")[-1])
        return "\n".join(import_lines), symbols

    @staticmethod
    def extract_edges(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        imported_symbols: list[str],
    ) -> list[tuple[str, str]]:
        """
        Extract deduplicated `(target_symbol, relation)` edges from an AST symbol node.

        Args:
            node: Top-level function, async function, or class node.
            imported_symbols: Symbol names imported at module scope.

        Returns:
            Deduplicated list of `(target_symbol, relation)` tuples (`inherits`, `calls`, `imports`).
        """
        edges: list[tuple[str, str]] = []
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if name and name not in _BUILTINS:
                    edges.append((name, "inherits"))

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = (
                    child.func.id
                    if isinstance(child.func, ast.Name)
                    else getattr(child.func, "attr", None)
                )
                if callee and callee not in {node.name, *_BUILTINS}:
                    edges.append((callee, "calls"))

        referenced = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        edges.extend((imp, "imports") for imp in imported_symbols if imp in referenced)
        return list(dict.fromkeys(edges))

    def build_documents(
        self,
        record: RawFileRecord,
        text: str,
        symbol_name: str,
        symbol_type: str,
        max_chars: int,
        edges: list[tuple[str, str]] | None = None,
        occurrence: int = 0,
    ) -> list[Document]:
        """Split `text` only if it exceeds `max_chars` and wrap parts as `Document` chunks."""
        parts = (
            [text]
            if len(text) <= max_chars
            else RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=max_chars,
                chunk_overlap=min(self.overlap_chars, max_chars - 1),
            ).split_text(text)
        )
        chunks = []
        for idx, part in enumerate(parts):
            identity = sha256_hex(f"{record.resolved_path}\n{occurrence}\n{part}", 24)
            chunks.append(
                Document(
                    page_content=part,
                    metadata={
                        "chunk_id": f"{record.filename}::{symbol_name}::{idx}::{identity}",
                        "source": record.filename,
                        "file_path": record.resolved_path,
                        "type": "python_code",
                        "language": "python",
                        "symbol_name": symbol_name,
                        "symbol_type": symbol_type,
                        "graph_edges": (edges or []) if idx == 0 else [],
                    },
                )
            )
        return chunks

    def chunk(self, record: RawFileRecord, max_chars: int | None = None) -> list[Document]:
        """
        Parse Python AST into self-contained symbol chunks with dependency edges.

        Args:
            record: Loaded Python `RawFileRecord`.
            max_chars: Optional per-call override for `self.max_chars`.

        Returns:
            List of `Document` chunks ready for graph registration and ChromaDB upsert.
        """
        limit = max_chars if max_chars is not None else self.max_chars
        lines = record.content.splitlines()
        try:
            tree = ast.parse(record.content)
        except SyntaxError:
            return self.build_documents(record, record.content, "module_block", "module", limit)

        imports_block, imported_symbols = self.extract_imports(tree, lines)
        chunks: list[Document] = []
        occurrences: dict[str, int] = {}

        for node in tree.body:
            if (
                not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                or node.name.startswith("_")
                or not node.end_lineno
            ):
                continue

            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            code_block = "\n".join(lines[start - 1 : node.end_lineno])
            sym_type = "class" if isinstance(node, ast.ClassDef) else "function"
            doc = ast.get_docstring(node)
            intent = doc.strip().splitlines()[0] if doc and doc.strip() else f"Python {sym_type} `{node.name}`"

            header = f"# Symbol: {node.name} ({sym_type}) | File: {record.filename}\n# Intent: {intent}\n"
            enriched = f"{header}\n{imports_block}\n\n{code_block}" if imports_block else f"{header}\n{code_block}"
            edges = self.extract_edges(node, imported_symbols)
            occurrence = occurrences.get(node.name, 0)
            occurrences[node.name] = occurrence + 1

            chunks.extend(
                self.build_documents(record, enriched, node.name, sym_type, limit, edges, occurrence)
            )

        return chunks or self.build_documents(record, record.content, "module_block", "module", limit)
