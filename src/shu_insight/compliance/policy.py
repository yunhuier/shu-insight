from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

from shu_insight.domain.models import SourceConfig

BLOCKED_SUFFIXES = frozenset(
    {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".png", ".jpg", ".jpeg", ".zip"}
)

BLOCKED_CONTENT_TYPES = (
    "application/pdf",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument",
    "image/",
)

_ENCODED_PATH_ESCAPE = re.compile(r"%(?:2e|2f|5c|25)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AccessDecision:
    allowed: bool
    persistable: bool
    reason: str


def _deny(reason: str) -> AccessDecision:
    return AccessDecision(allowed=False, persistable=False, reason=reason)


def evaluate_request(
    source: SourceConfig, url: str, *, robots_request: bool = False
) -> AccessDecision:
    if not url or url != url.strip() or any(ord(char) < 33 for char in url):
        return _deny("invalid_url")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return _deny("invalid_url")
    if parsed.scheme.lower() != "https":
        return _deny("https_required")
    allowed_host = urlsplit(source.base_url).hostname
    if parsed.hostname is None or parsed.hostname.lower() != (allowed_host or "").lower():
        return _deny("host_not_allowed")
    if (parsed.username is not None or parsed.password is not None or port is not None
            or "\\" in parsed.path or _ENCODED_PATH_ESCAPE.search(parsed.path)):
        return _deny("invalid_url")
    if "." in parsed.path.split("/") or ".." in parsed.path.split("/"):
        return _deny("invalid_url")
    if robots_request:
        if parsed.path != "/robots.txt" or parsed.query or parsed.fragment:
            return _deny("path_not_allowed")
        return AccessDecision(allowed=True, persistable=False, reason="allowed")
    decoded_path = unquote(parsed.path)
    if PurePosixPath(decoded_path).suffix.lower() in BLOCKED_SUFFIXES:
        return _deny("attachment_blocked")
    if not any(
        _path_matches_prefix(decoded_path, prefix) for prefix in source.allowed_path_prefixes
    ):
        return _deny("path_not_allowed")
    return AccessDecision(allowed=True, persistable=False, reason="allowed")


def _path_matches_prefix(path: str, prefix: str) -> bool:
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return (
        path == prefix
        or path.startswith(prefix + "/")
        or bool(re.fullmatch(re.escape(prefix) + r"\d*\.htm", path))
    )


def _url_scope_decision(source: SourceConfig, url: str) -> AccessDecision:
    return evaluate_request(source, url)


def is_login_or_captcha_page(body: str) -> bool:
    soup = BeautifulSoup(body, "lxml")
    if soup.select_one('input[type="password"], input[name*="captcha" i], img[src*="captcha" i]'):
        return True
    title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
    if any(marker in title for marker in ("统一身份认证", "身份验证", "验证码", "captcha")):
        return True
    auth_headings = {"用户登录", "用户登陆", "账户登录", "登录", "登录页面", "login", "sign in"}
    if title in auth_headings:
        return True
    heading = soup.select_one("h1")
    if heading is not None and heading.get_text(" ", strip=True).lower() in auth_headings:
        return True
    visible = soup.get_text(" ", strip=True).lower()
    return "请输入验证码" in visible


def evaluate_discovery(source: SourceConfig, url: str, title: str) -> AccessDecision:
    if not title.strip() or "[演示]" in title:
        return _deny("sensitive_title")
    return _url_scope_decision(source, url)


def evaluate_response(
    source: SourceConfig,
    status: int,
    content_type: str,
    final_url: str,
    body: str,
) -> AccessDecision:
    if status == 401:
        return _deny("authentication_required")
    if status == 403:
        return _deny("forbidden")
    if status == 429:
        return _deny("rate_limited")
    if 300 <= status < 400:
        return _deny("unresolved_redirect")
    if status >= 400:
        return _deny("http_error")

    normalized_content_type = content_type.lower()
    if any(marker in normalized_content_type for marker in BLOCKED_CONTENT_TYPES):
        return _deny("attachment_blocked")
    if not normalized_content_type.startswith(("text/html", "application/xhtml+xml")):
        return _deny("unsupported_content_type")

    scope = _url_scope_decision(source, final_url)
    if not scope.allowed:
        return scope

    if is_login_or_captcha_page(body):
        return _deny("login_or_captcha")
    return AccessDecision(allowed=True, persistable=True, reason="allowed")
