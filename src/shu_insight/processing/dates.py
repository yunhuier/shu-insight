"""Site-local dates are Shanghai wall time; persisted instants are UTC."""
import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
PUBLICATION_DATE = re.compile(
    r"(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})日?"
    r"(?:[ T]+(\d{1,2})[:：](\d{2}))?"
)


def utc_datetime(value: datetime) -> datetime:
    return (value if value.tzinfo else value.replace(tzinfo=SHANGHAI_TZ)).astimezone(UTC)


def parse_publication(text: str | None) -> datetime | None:
    match = PUBLICATION_DATE.search(text or "")
    if not match:
        return None
    year, month, day, hour, minute = match.groups()
    try:
        return datetime(int(year), int(month), int(day), int(hour or 0),
                        int(minute or 0), tzinfo=SHANGHAI_TZ)
    except ValueError:
        return None
