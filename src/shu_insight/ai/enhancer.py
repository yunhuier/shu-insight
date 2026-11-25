from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from shu_insight.ai.provider import AIProvider, AIRequest, ProviderError
from shu_insight.compliance.detector import sanitize_for_ai, scan_sensitive_text


class InvalidAIOutput(ValueError):
    """Raised when optional AI output fails its narrow schema or safety gate."""


@dataclass(frozen=True, slots=True)
class NoticeFactPacket:
    title: str
    category: str = ""
    published_text: str = ""
    deadline_text: str = ""
    location: str = ""
    audience: tuple[str, ...] = ()
    topic_tags: tuple[str, ...] = ()
    excerpts: tuple[str, ...] = ()

    def to_prompt_text(self) -> str:
        payload = {
            "title": self.title,
            "category": self.category,
            "published_text": self.published_text,
            "deadline_text": self.deadline_text,
            "location": self.location,
            "audience": self.audience,
            "topic_tags": self.topic_tags,
            "evidence_excerpts": tuple(excerpt[:240] for excerpt in self.excerpts[:3]),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True, slots=True)
class EnhancedText:
    text: str
    source: str


@dataclass(frozen=True, slots=True)
class ExpansionSuggestion:
    term: str
    enabled: bool = False
    source: str = "ai"


class FallbackEnhancer:
    def summarize(
        self,
        facts: NoticeFactPacket,
        score_evidence: Mapping[str, object],
    ) -> EnhancedText:
        del score_evidence
        details = []
        if facts.deadline_text:
            details.append(f"截止时间：{facts.deadline_text}")
        if facts.location:
            details.append(f"地点：{facts.location}")
        suffix = f"；{'；'.join(details)}" if details else ""
        return EnhancedText(f"{facts.title}{suffix}。", "fallback")

    def explain(
        self,
        facts: NoticeFactPacket,
        score_evidence: Mapping[str, object],
    ) -> EnhancedText:
        matched = score_evidence.get("matched_terms", ())
        if isinstance(matched, (list, tuple)) and matched:
            return EnhancedText(f"匹配关键词：{'、'.join(map(str, matched))}", "fallback")
        return EnhancedText(f"依据规则保留：{facts.title}", "fallback")

    def review_candidates(self, facts: Sequence[NoticeFactPacket]) -> EnhancedText:
        return EnhancedText(f"规则候选共 {len(facts)} 条，未由 AI 改动。", "fallback")

    def suggest_expansions(self, terms: tuple[str, ...]) -> tuple[ExpansionSuggestion, ...]:
        del terms
        return ()


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[。！？.!?]+", text) if part.strip()])


def _parse_object(raw: str) -> dict[str, Any]:
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidAIOutput("response is not JSON") from exc
    if not isinstance(document, dict):
        raise InvalidAIOutput("response must be a JSON object")
    return document


def _validate_text(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InvalidAIOutput(f"missing {field}")
    text = value.strip()
    if len(text) > 300 or _sentence_count(text) > 2:
        raise InvalidAIOutput(f"{field} exceeds the two-sentence limit")
    if scan_sensitive_text(text).blocked:
        raise InvalidAIOutput(f"{field} contains sensitive data")
    return text


class NoticeEnhancer:
    def __init__(
        self,
        provider: AIProvider,
        *,
        enabled: bool = True,
        fallback: FallbackEnhancer | None = None,
    ) -> None:
        self.provider = provider
        self.enabled = enabled
        self.fallback = fallback or FallbackEnhancer()

    def _safe_request(
        self,
        operation: str,
        facts: NoticeFactPacket,
        score_evidence: Mapping[str, object],
    ) -> AIRequest | None:
        del score_evidence  # Matched terms and scores may reveal private preferences.
        raw_fields = asdict(facts)
        fields = [value for value in raw_fields.values() if isinstance(value, str)]
        fields.extend(value for group in raw_fields.values() if isinstance(group, tuple)
                      for value in group)
        if any(scan_sensitive_text(value).blocked for value in fields):
            return None
        sanitized = {
            key: sanitize_for_ai(value) if isinstance(value, str)
            else tuple(sanitize_for_ai(item) for item in value)
            for key, value in raw_fields.items()
        }
        safe_facts = NoticeFactPacket(**sanitized)
        raw = json.dumps(
            {
                "operation": operation,
                "facts": json.loads(safe_facts.to_prompt_text()),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return AIRequest(operation=operation, text=raw)

    def _enhance_text(
        self,
        operation: str,
        output_field: str,
        facts: NoticeFactPacket,
        score_evidence: Mapping[str, object],
    ) -> EnhancedText:
        fallback_method = getattr(self.fallback, operation)
        if not self.enabled:
            return fallback_method(facts, score_evidence)
        request = self._safe_request(operation, facts, score_evidence)
        if request is None:
            return fallback_method(facts, score_evidence)
        try:
            response = self.provider.complete(request)
            text = _validate_text(_parse_object(response.text), output_field)
        except (ProviderError, InvalidAIOutput, TypeError, ValueError):
            return fallback_method(facts, score_evidence)
        return EnhancedText(text, "ai")

    def summarize(
        self,
        facts: NoticeFactPacket,
        score_evidence: Mapping[str, object],
    ) -> EnhancedText:
        return self._enhance_text("summarize", "summary", facts, score_evidence)

    def explain(
        self,
        facts: NoticeFactPacket,
        score_evidence: Mapping[str, object],
    ) -> EnhancedText:
        return self._enhance_text("explain", "explanation", facts, score_evidence)

    def review_candidates(self, facts: Sequence[NoticeFactPacket]) -> EnhancedText:
        if not self.enabled:
            return self.fallback.review_candidates(facts)
        safe_facts = tuple(facts[:20])
        if any(self._safe_request("review_candidates", fact, {}) is None for fact in safe_facts):
            return self.fallback.review_candidates(facts)
        synthetic = NoticeFactPacket(
            title="候选通知审查",
            excerpts=tuple(self._safe_request("review_candidates", fact, {}).text
                           for fact in safe_facts),
        )
        request = self._safe_request("review_candidates", synthetic, {})
        if request is None:
            return self.fallback.review_candidates(facts)
        try:
            response = self.provider.complete(request)
            text = _validate_text(_parse_object(response.text), "review")
        except (ProviderError, InvalidAIOutput, TypeError, ValueError):
            return self.fallback.review_candidates(facts)
        return EnhancedText(text, "ai")

    def suggest_expansions(
        self, terms: tuple[str, ...], *, consent_to_send_terms: bool = False
    ) -> tuple[ExpansionSuggestion, ...]:
        """Consent means explicitly agreeing to send these private terms to the provider."""
        if not self.enabled or not consent_to_send_terms or any(
            scan_sensitive_text(term).blocked for term in terms
        ):
            return self.fallback.suggest_expansions(terms)
        safe_terms = tuple(sanitize_for_ai(term) for term in terms if sanitize_for_ai(term))[:8]
        request = AIRequest(
            operation="suggest_expansions",
            text=json.dumps({"terms": safe_terms}, ensure_ascii=False),
        )
        try:
            document = _parse_object(self.provider.complete(request).text)
            raw_terms = document.get("terms")
            if not isinstance(raw_terms, list) or len(raw_terms) > 8:
                raise InvalidAIOutput("terms must be a list of at most eight items")
            suggestions: list[ExpansionSuggestion] = []
            for raw_term in raw_terms:
                if not isinstance(raw_term, str) or not raw_term.strip():
                    raise InvalidAIOutput("expansion term must be non-empty text")
                term = raw_term.strip()
                if scan_sensitive_text(term).blocked:
                    raise InvalidAIOutput("expansion term contains sensitive data")
                suggestions.append(ExpansionSuggestion(term=term))
            return tuple(suggestions)
        except (ProviderError, InvalidAIOutput, TypeError, ValueError):
            return self.fallback.suggest_expansions(terms)
