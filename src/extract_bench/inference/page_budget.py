"""FIFO admission bounded by both document count and source PDF pages."""

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class PageBudget:
    """Hold document/page capacity for a complete local attempt, including retries.

    Oversized PDFs run alone; FIFO prevents later small files from overtaking
    them. Releasing a timed-out attempt does not guarantee remote cancellation.
    """

    def __init__(
        self,
        documents: Sequence[tuple[str, Path]],
        max_documents: int,
        max_pages: int,
        output_dir: Path,
    ) -> None:
        if type(max_documents) is not int or max_documents < 1:
            raise ValueError("max_concurrent must be a positive integer")
        if type(max_pages) is not int or max_pages < 1:
            raise ValueError("max_concurrent_pages must be a positive integer")
        import pymupdf

        self.max_documents = max_documents
        self.max_pages = max_pages
        self.pages: dict[str, int] = {}
        # Count every source before submitting anything. Never estimate an unreadable PDF.
        for example_id, path in documents:
            if example_id in self.pages:
                raise ValueError(f"Duplicate document ID: {example_id}")
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"Page admission requires PDF input: {path}")
            with pymupdf.open(path) as pdf:
                if pdf.needs_pass or len(pdf) < 1:
                    raise ValueError(f"Cannot count readable pages: {path}")
                self.pages[example_id] = len(pdf)
        self.active_documents = 0
        self.active_pages = 0
        self.waiting: deque[str] = deque()
        self.condition = asyncio.Condition()
        self.events = output_dir / "_admission_events.jsonl"
        # Preserve the previous session, including an interrupted JSONL tail,
        # before creating a new plan/log for a resumed run with cached results.
        previous = [path for path in (self.events, output_dir / "_admission_plan.json") if path.exists()]
        if len(previous) > 0:
            archive = output_dir / "_admission_history" / str(uuid4())
            archive.mkdir(parents=True)
            for path in previous:
                path.rename(archive / path.name)
        self.events.write_text("")
        (output_dir / "_admission_plan.json").write_text(
            json.dumps(
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "max_concurrent": max_documents,
                    "max_concurrent_pages": max_pages,
                    "oversized_document_policy": "run_alone",
                    "documents": [{"example_id": key, "pages": value} for key, value in self.pages.items()],
                },
                indent=2,
            )
            + "\n"
        )

    def _record(self, event: str, example_id: str | None = None) -> None:
        record = {
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            "example_id": example_id,
            "pages": self.pages[example_id] if example_id is not None else None,
            "active_documents": self.active_documents,
            "active_pages": self.active_pages,
        }
        with self.events.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    @asynccontextmanager
    async def slot(self, example_id: str) -> AsyncIterator[None]:
        pages = self.pages[example_id]
        async with self.condition:
            self.waiting.append(example_id)
            try:
                await self.condition.wait_for(
                    lambda: (
                        self.waiting[0] == example_id
                        and self.active_documents < self.max_documents
                        and (self.active_pages + pages <= self.max_pages or self.active_documents == 0)
                    )
                )
                self.active_documents += 1
                self.active_pages += pages
                try:
                    self._record("admitted", example_id)
                except BaseException:
                    self.active_documents -= 1
                    self.active_pages -= pages
                    raise
            finally:
                self.waiting.remove(example_id)
                self.condition.notify_all()
        try:
            yield
        finally:
            async with self.condition:
                self.active_documents -= 1
                self.active_pages -= pages
                try:
                    self._record("released", example_id)
                finally:
                    self.condition.notify_all()
