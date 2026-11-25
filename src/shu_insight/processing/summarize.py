from __future__ import annotations

import re

_SENTENCES = re.compile(r"[^。！？!?]+[。！？!?]?")
_ACTION = ("报名", "申请", "提交", "参加", "截止", "办理", "完成", "选课")
_AUDIENCE = ("本科生", "研究生", "新生", "同学", "学生", "师生")
_DATE = re.compile(r"\d{1,2}月\d{1,2}日|\d{4}[-年]\d{1,2}")


def summarize_article(text: str, *, limit: int = 350) -> str:
    """Select source sentences, with a bounded preview for oversized paragraphs."""
    sentences = [sentence.strip() for sentence in _SENTENCES.findall(text) if sentence.strip()]

    def score(sentence: str) -> int:
        return (4 * any(word in sentence for word in _ACTION)
                + 3 * any(word in sentence for word in _AUDIENCE)
                + 2 * bool(_DATE.search(sentence)))

    chosen: list[tuple[int, str]] = []
    total = 0
    ranked = sorted(enumerate(sentences), key=lambda item: (-score(item[1]), item[0]))
    for index, sentence in ranked:
        if len(chosen) == 4:
            break
        if len(sentence) + total <= limit:
            chosen.append((index, sentence))
            total += len(sentence)
    if not chosen and ranked and limit > 0:
        return ranked[0][1][:limit - 1] + "…"
    return "".join(sentence for _, sentence in sorted(chosen))
