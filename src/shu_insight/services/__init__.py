"""Application services that coordinate repositories and domain modules."""

from shu_insight.services.notices import NoticeQuery, NoticeService
from shu_insight.services.update import UpdateService

__all__ = ["NoticeQuery", "NoticeService", "UpdateService"]
