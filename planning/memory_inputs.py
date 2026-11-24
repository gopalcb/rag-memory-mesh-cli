"""Validate memory lookup and handoff requests without starting the index."""

import re
from pathlib import PurePosixPath
from typing import Annotated
from unicodedata import normalize

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


ShortText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)
]
Detail = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)
]


Keywords = Annotated[list[ShortText], Field(min_length=1, max_length=12)]


class MemoryLookup(BaseModel):
    """Inputs the main agent prepares before looking up prior work."""

    model_config = ConfigDict(extra="forbid")

    query: Detail
    keywords: Keywords

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, keywords: list[str]) -> list[str]:
        """Require searchable words or identifiers in each keyword."""
        if any(not re.search(r"\w", keyword) for keyword in keywords):
            raise ValueError(
                "Each keyword must contain a searchable word or identifier"
            )
        return keywords


class ChangedFile(BaseModel):
    """A project-relative location and an optional snapshot supplied by the caller."""

    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1024)
    ]
    change: ShortText
    content: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        """Keep caller-provided paths inside the handoff's snapshot directory."""
        if re.search(r"[\\:\x00-\x1f]", value) or any(
            part in {"", ".", ".."} for part in value.split("/")
        ):
            raise ValueError("Use a project-relative path with forward slashes")
        return value


ChangedFiles = Annotated[list[ChangedFile], Field(max_length=100)]


class TaskChanges(BaseModel):
    """Implementation, verification and file details for a completed task."""

    model_config = ConfigDict(extra="forbid")

    project: ShortText
    task: ShortText
    implementation_summary: Detail
    verification: Detail
    files: ChangedFiles = Field(default_factory=list)

    @field_validator("files")
    @classmethod
    def validate_files(cls, files: list[ChangedFile]) -> list[ChangedFile]:
        """Reject duplicate file names and file-directory collisions before staging."""
        paths: set[PurePosixPath] = set()
        snapshots: set[PurePosixPath] = set()
        for file in files:
            path = PurePosixPath(normalize("NFC", file.path.casefold()))
            if path in paths:
                raise ValueError(f"Duplicate changed file: {file.path}")
            paths.add(path)
            if file.content is not None:
                snapshots.add(path)

        if any(parent in snapshots for path in snapshots for parent in path.parents):
            raise ValueError("A snapshot path cannot also be a directory")
        return files
