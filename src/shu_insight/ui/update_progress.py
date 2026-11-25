"""Progress based on observed source steps, without estimating article totals."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from shu_insight.ingest.pipeline import UpdateEvent, UpdateReport


@dataclass
class ManualUpdateProgress:
    source_names: Mapping[str, str]
    pages_read: int = 0
    written: int = 0
    skipped: int = 0
    label: str = "准备检查公开通知…"
    status_state: str = "running"
    _current: str = ""
    _ended: set[str] = field(default_factory=set)

    @property
    def fraction(self) -> float:
        return len(self._ended) / len(self.source_names) if self.source_names else 0.0

    @property
    def progress_text(self) -> str:
        return (f"已结束 {len(self._ended)}/{len(self.source_names)} 个来源 · "
                f"已读 {self.pages_read} 个栏目页 · 写入 {self.written} 条 · "
                f"跳过 {self.skipped} 条")

    def observe(self, event: UpdateEvent) -> None:
        if event.source_id not in self.source_names:
            return
        name = self.source_names[event.source_id]
        if event.kind == "source_started":
            # A partial source has no terminal event; starting the next source
            # nevertheless proves the prior source's work has ended.
            if self._current and self._current != event.source_id:
                self._ended.add(self._current)
            self._current = event.source_id
            self.label = f"正在检查{name}…"
        elif event.kind in {"source_finished", "source_failed"}:
            self._ended.add(event.source_id)
            self.label = f"{name}：" + (
                "本轮检查结束" if event.kind == "source_finished" else "本来源检查停止")
        elif event.kind == "list_page_read":
            self.pages_read += 1
            self.label = f"正在读取{name}的通知…"
        elif event.kind == "notice_saved":
            self.written += 1
        elif event.kind == "notice_skipped":
            self.skipped += 1

    def finish(self, report: UpdateReport) -> None:
        self._ended.update(set(report.by_source) & set(self.source_names))
        if report.by_source:
            self.pages_read = sum(item.list_pages_read for item in report.by_source.values())
            self.written = report.saved_count
            self.skipped = sum(item.skipped_count for item in report.by_source.values())
        self.status_state = "error" if report.state in {"failed", "interrupted"} else "complete"
        self.label = {
            "success": "本轮更新完成",
            "partial": "本轮更新结束，部分检查未完成",
            "failed": "本轮未能完成更新，请查看下方原因",
            "interrupted": "本轮更新已中断",
            "concurrent_skipped": "已有更新正在进行，本次未重复启动",
        }.get(report.state, "本轮更新已结束")
