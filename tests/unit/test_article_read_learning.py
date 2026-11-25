from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import (
    ArticleReadRepository,
    NoticeRepository,
    SearchHistoryRepository,
)
from shu_insight.db.schema import Database
from shu_insight.domain.models import NoticeDraft, ProofStatus, Verification
from shu_insight.processing.classify import load_taxonomy
from shu_insight.profiles.learning import ArticleReadLearner
from shu_insight.sources.registry import load_source_registry


@pytest.fixture
def readings(tmp_path):
    database = Database(tmp_path / "readings.sqlite3")
    database.initialize()
    registry = load_source_registry(Path("config/sources.toml"))
    notices = NoticeRepository(database, OfficialNoticePolicy(
        {key: adapter.config for key, adapter in registry.items()}))
    for index in range(12):
        topic = "科研论文讲座" if index < 6 else "竞赛挑战杯报名"
        notices.upsert(
            NoticeDraft("shu_main", f"https://www.shu.edu.cn/info/1051/{401000 + index}.htm",
                        f"{topic}第{index + 1}期通知", f"{topic}请按通知要求提交材料。",
                        topic_tags=("科研" if index < 6 else "竞赛",)),
            verification=Verification(ProofStatus.VERIFIED_SNAPSHOT,
                                      datetime(2026, 11, 1, tzinfo=UTC), "snapshot:read-test"),
        )
    records = sorted(notices.list_all(), key=lambda notice: notice.canonical_url)
    repository = ArticleReadRepository(database)
    learner = ArticleReadLearner(repository, load_taxonomy(Path("config/taxonomy.toml")),
                                 notice_source=lambda: records)
    return database, records, repository, learner


def test_search_history_does_not_unlock_article_profile(readings):
    database, _, _, learner = readings
    searches = SearchHistoryRepository(database)
    for index in range(12):
        searches.record(tags=("就业",), keywords=(str(index),), terms=("招聘",),
                        persona_id=None, result_count=12)
    assert searches.count() == 12
    assert learner.read_count() == 0
    assert learner.learn() is None


def test_repeat_clicks_count_one_article_and_persist_across_sessions(readings):
    database, records, repository, learner = readings
    assert learner.record_read(records[0])
    first = repository.entries()[0]
    for _ in range(12):
        assert not learner.record_read(records[0])
    assert ArticleReadRepository(Database(database.path)).count() == 1
    assert repository.entries() == [first]
    assert learner.learn() is None


def test_tenth_distinct_article_unlocks_using_article_content(readings):
    _, records, _, learner = readings
    for record in records[:9]:
        learner.record_read(record)
    assert learner.read_count() == 9
    assert learner.learn() is None
    learner.record_read(records[9])
    learned = learner.learn()
    assert learned is not None
    assert learned.article_count == 10
    assert learned.persona.label.startswith("我的阅读兴趣·")
    assert {"竞赛", "挑战杯"} & set(learned.representative_terms)
    assert "招聘" not in learned.representative_terms


def test_concurrent_clicks_only_insert_once(readings):
    _, records, repository, _ = readings
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(repository.record, [records[0].id] * 12))
    assert sum(results) == 1
    assert repository.count() == 1


def test_migration_keeps_notices_and_legacy_searches_without_inventing_reads(readings):
    database, records, _, _ = readings
    SearchHistoryRepository(database).record(tags=("就业",), keywords=(), terms=("招聘",),
                                             persona_id=None, result_count=12)
    with database.transaction() as connection:
        connection.execute("DROP TABLE article_reads")
        connection.execute("UPDATE schema_meta SET version = 6")
    database.initialize()
    with database.connect() as connection:
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 7
    assert sorted(NoticeRepository(database).list_all(),
                  key=lambda notice: notice.canonical_url) == records
    assert SearchHistoryRepository(database).count() == 1
    assert ArticleReadRepository(database).count() == 0
    assert list((database.path.parent / "backups").glob("*-v6-*.sqlite3"))
