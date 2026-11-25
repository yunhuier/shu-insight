"""Read official HTML on explicit QA submit; cache only cleaned text in session memory."""
from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from shu_insight.compliance.policy import is_login_or_captcha_page
from shu_insight.ingest.http_client import PoliteHttpClient, SafeTransportError
from shu_insight.processing.public_preview import safe_stored_summary
from shu_insight.services.notice_qa import Evidence, rank_evidence
from shu_insight.sources.base import AttachmentOnly, canonicalize_url
from shu_insight.sources.registry import load_source_registry

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FulltextResult:
    evidence: tuple[Evidence, ...]
    warnings: tuple[str, ...]
    fulltext_count: int
    cached_count: int


def _chunks(text: str) -> list[str]:
    return [text[start:start + 900] for start in range(0, len(text), 750)]


def prepare_fulltext(
    candidates: Sequence[Evidence], question: str, cache: MutableMapping, *,
    client_factory: Callable | None = None, on_progress: Callable[[str], None] | None = None,
) -> FulltextResult:
    if not candidates:
        return FulltextResult((), (), 0, 0)
    registry = load_source_registry(ROOT / "config/sources.toml")
    factory = client_factory or (lambda: PoliteHttpClient(
        contact=os.getenv("SHU_NOTICE_CONTACT", "")))
    client = None
    groups = []
    warnings = []
    fulltext_count = cached_count = 0
    try:
        for index, original in enumerate(candidates[:3], 1):
            if on_progress:
                on_progress(f"读取官网全文 {index}/{min(len(candidates), 3)}：{original.title}")
            signature = hashlib.sha256(original.text.encode()).hexdigest()
            saved = cache.get(original.url)
            if (saved and saved.get("signature") == signature
                    and time.time() - saved["time"] < (900 if saved["text"] else 60)):
                body, reason, observed = saved["text"], saved["reason"], saved["observed"]
                cached_count += int(bool(body))
            else:
                body, reason = "", ""
                observed = datetime.now(UTC).astimezone().strftime("%m-%d %H:%M")
                try:
                    host = urlsplit(original.url).hostname
                    adapter = next(item for item in registry.values()
                                   if urlsplit(item.config.base_url).hostname == host)
                    if client is None:
                        client = factory()

                    def guard(target: str, expected: str = original.url) -> None:
                        if canonicalize_url(target) != canonicalize_url(expected):
                            raise ValueError("article_identity_changed")

                    page = client.fetch(adapter.config, original.url, purpose="detail",
                                        force_refresh=True, redirect_target_guard=guard)
                    if (page.status_code != 200 or is_login_or_captcha_page(page.body)
                            or len(page.body) > 2_000_000
                            or page.content_type.split(";", 1)[0].lower() not in {
                                "text/html", "application/xhtml+xml"}):
                        raise ValueError("article_unavailable")
                    draft = adapter.parse_detail(page)
                    guard(draft.canonical_url)
                    # Remove contact clauses without discarding all later public paragraphs.
                    body = "".join(safe_stored_summary(part) for part in re.split(
                        r"(?<=[。！？；])|\n+", draft.normalized_text)).strip()[:60000]
                    if not body:
                        raise ValueError("no_public_text")
                except AttachmentOnly:
                    reason = "正文在附件中，暂不解析附件"
                except SafeTransportError:
                    reason = "官网网络连接失败，尚未取得正文"
                except Exception:
                    reason = "官网连接、访问或正文解析未成功"
                if len(cache) >= 32 and original.url not in cache:
                    cache.pop(next(iter(cache)))
                cache[original.url] = {"signature": signature, "time": time.time(), "text": body,
                                       "reason": reason, "observed": observed}
            if body:
                fulltext_count += 1
                pieces = tuple(Evidence(
                    original.notice_id, original.title, original.url,
                    f"标题：{original.title}\n读取于：{observed}\n正文片段：{chunk}",
                    f"官网全文片段（第{number}段，读取于{observed}）",
                ) for number, chunk in enumerate(_chunks(body), 1))
                ranked = rank_evidence(pieces, question)
                groups.append(tuple(ranked[:3] or pieces[:2]) if index == 1
                              else tuple(ranked[:1] or pieces[:1]))
            else:
                warnings.append(f"{original.title}：{reason}，本次使用已收录摘要或标题。")
                groups.append((original,))
    finally:
        if client is not None:
            client.close()
    # Current notice remains first, with up to three of its best chunks.
    selected = tuple(piece for group in groups for piece in group)[:5]
    return FulltextResult(selected, tuple(warnings), fulltext_count, cached_count)
