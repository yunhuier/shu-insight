from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from shu_insight.db.repositories import PersonaRepository
from shu_insight.db.schema import Database
from shu_insight.domain.models import Persona
from shu_insight.profiles.service import (
    DEMO_PERSONAS,
    PersonaService,
    PersonaValidationError,
)

BASE_PERSONA = Persona(
    id="base",
    label="  计算机学院大二  ",
    education_level="本科生",
    grade="大二",
    college="计算机工程与科学学院",
    major="不限专业",
    goals=("保研", "科研"),
    interest_tags=("竞赛",),
    custom_keywords=("推免",),
)


@pytest.fixture
def repository(tmp_path: Path) -> PersonaRepository:
    database = Database(tmp_path / "personas.sqlite3")
    database.initialize()
    return PersonaRepository(database)


@pytest.fixture
def service(repository: PersonaRepository) -> PersonaService:
    return PersonaService(repository)


def test_rejects_more_than_eight_tags_or_five_keywords(service: PersonaService) -> None:
    with pytest.raises(PersonaValidationError, match="兴趣标签最多 8 个"):
        service.save(replace(BASE_PERSONA, interest_tags=tuple(map(str, range(9)))))
    with pytest.raises(PersonaValidationError, match="自定义关键词最多 5 个"):
        service.save(replace(BASE_PERSONA, custom_keywords=tuple(map(str, range(6)))))


def test_no_profile_mode_is_not_persisted_as_identity(
    service: PersonaService, repository: PersonaRepository
) -> None:
    query = service.no_profile_query(("保研", "科研"), ("推免",))

    assert query.persona_id is None
    assert query.interest_tags == ("保研", "科研")
    assert query.custom_keywords == ("推免",)
    assert repository.list_all() == []


def test_save_normalizes_text_and_rejects_invalid_level_grade_or_major(
    service: PersonaService,
) -> None:
    saved = service.save(BASE_PERSONA)

    assert saved.label == "计算机学院大二"
    with pytest.raises(PersonaValidationError, match="年级与培养层次不匹配"):
        service.save(replace(BASE_PERSONA, id="bad-grade", grade="研一"))
    with pytest.raises(PersonaValidationError, match="专业不在当前可选范围"):
        service.save(replace(BASE_PERSONA, id="bad-major", major="软件工程"))


@pytest.mark.parametrize(
    "keywords",
    [
        ("推免", "  "),
        ("AI", " ai "),
    ],
)
def test_rejects_blank_or_duplicate_normalized_keywords(
    service: PersonaService, keywords: tuple[str, ...]
) -> None:
    with pytest.raises(PersonaValidationError, match="自定义关键词"):
        service.save(replace(BASE_PERSONA, custom_keywords=keywords))


def test_update_copy_and_delete_use_distinct_local_ids(service: PersonaService) -> None:
    service.save(BASE_PERSONA)
    updated = service.update(replace(BASE_PERSONA, label="科研与升学"))
    copied = service.copy(BASE_PERSONA.id)

    assert updated.label == "科研与升学"
    assert copied.id != BASE_PERSONA.id
    assert copied.label.endswith("副本")
    service.delete(BASE_PERSONA.id)
    assert [persona.id for persona in service.list_all()] == [copied.id]


def test_delete_all_and_reset_recreate_editable_demo_personas(
    service: PersonaService,
) -> None:
    first_reset = service.reset()
    service.update(replace(first_reset[0], label="临时修改"))
    service.delete_all()

    assert service.list_all() == []

    second_reset = service.reset()
    assert second_reset == DEMO_PERSONAS
    assert {persona.id for persona in second_reset} == {
        persona.id for persona in DEMO_PERSONAS
    }
    assert any(persona.custom_keywords for persona in second_reset)
