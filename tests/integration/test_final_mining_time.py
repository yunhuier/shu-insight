from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository, SourceRepository
from shu_insight.db.schema import Database
from shu_insight.domain.models import FetchedPage, NoticeDraft, ProofStatus
from shu_insight.ingest.pipeline import IngestionPipeline
from shu_insight.ranking.score import can_strong_remind
from shu_insight.services.notices import NoticeService
from shu_insight.sources.registry import load_source_registry


class LivePathFixture:
    url = "https://gs.shu.edu.cn/info/1029/173242.htm"
    title = "上海大学2026级硕博连读研究生、直博生 申请课程免修及成绩认定通知"

    def fetch(
        self, source, url, *, force_refresh=False, redirect_target_guard=None, purpose="detail",
    ):
        del force_refresh, redirect_target_guard
        if url == source.list_url:
            body = ('<table width="755"><tr id="line_u17_0"><td><div><font>'
                    f'<a href="{self.url}">{self.title}</a>'
                    '</font></div></td><td>2026/09/11</td></tr></table>')
        else:
            body = (
                f'<h1>{self.title}</h1><div class="v_news_content">'
                '2026级硕博连读研究生、直博生符合条件者可申请相应课程免修或已修课程的学分、成绩认定。'
                '申请时间为2026年9月11日至10月11日。相关学生须先完成个人培养计划，'
                '再通过研究生综合服务平台提交申请，由导师和学院审批后交研究生院审核。'
                '</div>'
            )
        return FetchedPage(source.id, url, url, 200, "text/html", body,
                           datetime(2026, 9, 25, 8, tzinfo=UTC))


def system(tmp_path):
    db = Database(tmp_path / "mining.sqlite3")
    db.initialize()
    client = LivePathFixture()
    registry = load_source_registry(Path("config/sources.toml"))
    selected = {"graduate": registry["graduate"]}
    repo = NoticeRepository(db, OfficialNoticePolicy(
        {"graduate": registry["graduate"].config}))
    pipeline = IngestionPipeline(registry=selected, client=client, notices=repo,
                                 sources=SourceRepository(db))
    service = NoticeService(repo, authority_by_source_id={},
                            now=lambda: datetime(2026, 9, 25, 8, tzinfo=UTC))
    return repo, client, pipeline, service


def test_ingestion_to_detail_has_verified_live_proof_and_idempotent_version(tmp_path):
    repo, client, pipeline, service = system(tmp_path)
    assert pipeline.run(["graduate"]).saved_count == 1
    notice = repo.find_by_url(client.url)
    assert notice.proof_status is ProofStatus.VERIFIED_LIVE
    evidence = service.detail_evidence(notice.id)
    assert evidence["duplicates"] == []
    assert pipeline.run(["graduate"]).saved_count == 0
    assert len(service.detail_evidence(notice.id)["versions"]) == 1


def test_live_path_publication_order_and_shanghai_deadline_eligibility(tmp_path):
    repo, client, pipeline, service = system(tmp_path)
    pipeline.run(["graduate"])
    assert [n.source_id for n in service.latest()] == ["graduate"]
    notice = repo.find_by_url(client.url)
    assert notice.published_at == datetime(2026, 9, 10, 16, tzinfo=UTC)
    assert not can_strong_remind(notice, datetime(2026, 9, 25, 8, tzinfo=UTC))


def test_simhash_local_edit_and_metadata_only_version(tmp_path):
    repo, _, _, _ = system(tmp_path)
    base = NoticeDraft("computer", "https://cs.shu.edu.cn/info/1/3.htm", "科研通知",
                       "校园人工智能科研训练项目，邀请本科生参与创新研究与技术实践。" * 4)
    repo.upsert(base)
    first = repo.find_by_url(base.canonical_url)
    repo.upsert(replace(base, normalized_text=base.normalized_text + "欢迎报名。"))
    second = repo.find_by_url(base.canonical_url)
    assert (first.summary_simhash ^ second.summary_simhash).bit_count() < 12
    result = repo.upsert(replace(base, normalized_text=base.normalized_text + "欢迎报名。",
                                 published_at=datetime(2026, 9, 21)))
    assert result.updated
    assert repo.find_by_url(base.canonical_url).published_at.utcoffset() == timedelta(0)
