from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import replace
from datetime import UTC, date, datetime

from shu_insight.db.repositories import (
    ManualUpdateRunRepository,
    NoticeRepository,
    SourceRepository,
)
from shu_insight.db.schema import Database
from shu_insight.ingest.http_client import PoliteHttpClient
from shu_insight.ingest.pipeline import IngestionPipeline, UpdateEvent, UpdateReport
from shu_insight.processing.classify import Taxonomy
from shu_insight.services.run_lock import RunLock
from shu_insight.sources.base import SourceAdapter


class UpdateService:
    """One explicit click owns one client and run, under a database-adjacent lock."""

    MANUAL_SOURCE_IDS = ("shu_main", "undergraduate")

    def __init__(
        self,
        *,
        database: Database,
        registry: Mapping[str, SourceAdapter],
        client_factory: Callable[[], AbstractContextManager[PoliteHttpClient]],
        taxonomy: Taxonomy | None = None,
        today: Callable[[], date] = date.today,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.database = database
        self.registry = {key: registry[key] for key in self.MANUAL_SOURCE_IDS if key in registry}
        self.client_factory = client_factory
        self.taxonomy = taxonomy
        self.today = today
        self.now = now
        path = database.path.resolve()
        self.lock_path = path.with_name(path.name + ".lock")

    def update_all(
        self,
        on_event: Callable[[UpdateEvent], None] | None = None,
    ) -> UpdateReport:
        if not self.registry:
            return UpdateReport({}, (), state="failed", message="no_eligible_sources")
        run_id: str | None = None
        report: UpdateReport | None = None
        events: list[UpdateEvent] = []

        def emit(event: UpdateEvent) -> None:
            events.append(event)
            if on_event is not None:
                on_event(event)

        try:
            with RunLock(self.lock_path) as lock:
                if not lock.acquired:
                    return UpdateReport({}, (), state="concurrent_skipped")
                runs = ManualUpdateRunRepository(self.database)
                run_id = runs.start(started_at=self.now().astimezone(UTC))
                state = "interrupted"
                try:
                    # The factory may capture a session-only contact. It is
                    # never called before the lock, persisted, or logged here.
                    with self.client_factory() as client:
                        pipeline = IngestionPipeline(
                            registry=self.registry, client=client,
                            notices=NoticeRepository(self.database),
                            sources=SourceRepository(self.database), taxonomy=self.taxonomy,
                            today=self.today, now=self.now,
                        )
                        report = pipeline.run(on_event=emit, run_id=run_id)
                    results = tuple(report.by_source.values())
                    if results and all(result.state == "success" for result in results):
                        state = "success"
                    elif any(result.state in {"success", "partial"}
                             or result.saved_count or result.updated_count for result in results):
                        state = "partial"
                    else:
                        state = "failed"
                except Exception:
                    # Raw factory, close, or observer exceptions can contain
                    # request headers/contact. Expose only a fixed safe code.
                    state = "failed"
                finally:
                    runs.finish(run_id, state, finished_at=self.now().astimezone(UTC))
                return replace(
                    report or UpdateReport({}, tuple(events)), run_id=run_id, state=state,
                    message="update_failed" if state == "failed" else "",
                )
        except Exception:
            # Lock or persistence errors also remain safe; the OS lock is
            # released even if writing the terminal row fails.
            return replace(
                report or UpdateReport({}, tuple(events)), run_id=run_id,
                state="failed", message="update_failed",
            )
