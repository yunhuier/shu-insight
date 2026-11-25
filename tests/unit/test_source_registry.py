from __future__ import annotations

from pathlib import Path

import pytest

from shu_insight.domain.models import DiscoveredNotice
from shu_insight.sources.base import canonicalize_url, dedupe_discoveries, normalize_space
from shu_insight.sources.registry import SourceConfigurationError, load_source_registry


@pytest.fixture
def registry():
    return load_source_registry(Path("config/sources.toml"))


def test_registry_contains_only_four_reviewed_sources(registry) -> None:
    assert set(registry) == {"shu_main", "undergraduate", "graduate", "computer"}
    assert {adapter.config.list_url for adapter in registry.values()} == {
        "https://www.shu.edu.cn/tzgg.htm",
        "https://bksy.shu.edu.cn/index/tzgg.htm",
        "https://gs.shu.edu.cn/xwlb/sy.htm",
        "https://cs.shu.edu.cn/xxgk/tzgg.htm",
    }
    assert all(
        adapter.config.reviewed_on.isoformat() == "2026-09-21"
        for adapter in registry.values()
    )


def test_duplicate_source_ids_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "sources.toml"
    config.write_text(
        """
[[source]]
id = "same"
name = "one"
base_url = "https://one.example/"
list_url = "https://one.example/notices.htm"
allowed_path_prefixes = ["/notices"]
authority_score = 1.0
minimum_interval_seconds = 2.0
reviewed_on = "2026-09-21"

[[source]]
id = "same"
name = "two"
base_url = "https://two.example/"
list_url = "https://two.example/notices.htm"
allowed_path_prefixes = ["/notices"]
authority_score = 1.0
minimum_interval_seconds = 2.0
reviewed_on = "2026-09-21"
""",
        encoding="utf-8",
    )

    with pytest.raises(SourceConfigurationError, match="duplicate source id"):
        load_source_registry(config)


@pytest.mark.parametrize(
    ("base_url", "list_url", "message"),
    [
        ("http://one.example/", "https://one.example/notices.htm", "HTTPS"),
        ("https://one.example/", "https://two.example/notices.htm", "same host"),
        ("https://one.example/", "https://one.example/news.htm", "allowed path"),
    ],
)
def test_invalid_source_scope_is_rejected(
    tmp_path: Path, base_url: str, list_url: str, message: str
) -> None:
    config = tmp_path / "sources.toml"
    config.write_text(
        f"""
[[source]]
id = "one"
name = "one"
base_url = "{base_url}"
list_url = "{list_url}"
allowed_path_prefixes = ["/notices"]
authority_score = 1.0
minimum_interval_seconds = 2.0
reviewed_on = "2026-09-21"
""",
        encoding="utf-8",
    )

    with pytest.raises(SourceConfigurationError, match=message):
        load_source_registry(config)


def test_url_and_text_helpers_are_deterministic() -> None:
    first = DiscoveredNotice(
        "computer",
        "https://CS.SHU.EDU.CN/info/1.htm?utm_source=x#top",
        "  创新赛\n通知 ",
    )
    second = DiscoveredNotice(
        "computer",
        "https://cs.shu.edu.cn/info/1.htm",
        "创新赛通知",
    )

    assert canonicalize_url(first.url) == "https://cs.shu.edu.cn/info/1.htm"
    assert normalize_space(first.title) == "创新赛 通知"
    assert dedupe_discoveries([first, second]) == [first]


def test_four_official_pagination_contracts_are_loaded(registry):
    assert all(adapter.config.pagination_reviewed_on is not None
               for adapter in registry.values())
    assert all(adapter.config.pagination_link_text == "下页"
               for adapter in registry.values())
    assert all(adapter.pagination_coverage_code == "" for adapter in registry.values())


def _synthetic_registry_config(tmp_path, pagination_fields):
    path = tmp_path / "pagination.toml"
    path.write_text('''[[source]]
id = "one"
name = "Synthetic contract only"
base_url = "https://one.example/"
list_url = "https://one.example/notices.htm"
allowed_path_prefixes = ["/notices", "/info/"]
authority_score = 1.0
minimum_interval_seconds = 2.0
reviewed_on = "2026-09-21"
''' + pagination_fields, encoding="utf-8")
    return path


def test_complete_synthetic_pagination_contract_loads(tmp_path):
    config = _synthetic_registry_config(tmp_path, r'''pagination_selector = "nav.pages > a.next"
pagination_url_pattern = '/notices(?:/\d+)?\.htm'
pagination_reviewed_on = "2026-10-06"
''')
    adapter = load_source_registry(config)["one"]
    assert adapter.config.pagination_reviewed_on.isoformat() == "2026-10-06"
    assert adapter.next_list_url(
        '<nav class="pages"><a class="next" href="/notices/2.htm">Next</a></nav>',
        "https://one.example/notices.htm",
    ) == "https://one.example/notices/2.htm"


@pytest.mark.parametrize("fields", [
    'pagination_selector = "nav.pages > a.next"',
    'pagination_url_pattern = "/notices.*"',
    'pagination_reviewed_on = "2026-10-06"',
    '''pagination_selector = ""
pagination_url_pattern = "/notices.*"
pagination_reviewed_on = "2026-10-06"''',
    '''pagination_selector = "["
pagination_url_pattern = "/notices.*"
pagination_reviewed_on = "2026-10-06"''',
    '''pagination_selector = "nav.pages > a.next"
pagination_url_pattern = "["
pagination_reviewed_on = "2026-10-06"''',
    '''pagination_selector = "nav.pages > a.next"
pagination_url_pattern = "/info/.*"
pagination_reviewed_on = "2026-10-06"''',
    '''pagination_selector = "nav.pages > a.next"
pagination_url_pattern = "/notices.*"
pagination_reviewed_on = "bad-date"''',
])
def test_incoherent_pagination_contract_is_rejected(tmp_path, fields):
    with pytest.raises(SourceConfigurationError, match="pagination"):
        load_source_registry(_synthetic_registry_config(tmp_path, fields))


@pytest.mark.parametrize("link_text_field", [
    'pagination_link_text = ""',
    'pagination_link_text = "  "',
    'pagination_link_text = 123',
])
def test_invalid_exact_pagination_link_text_is_rejected(tmp_path, link_text_field):
    config = _synthetic_registry_config(tmp_path, '''pagination_selector = "a"
pagination_url_pattern = '/notices(?:/\\d+)?\\.htm'
pagination_reviewed_on = "2026-10-13"
''' + link_text_field)
    with pytest.raises(SourceConfigurationError, match="pagination"):
        load_source_registry(config)
