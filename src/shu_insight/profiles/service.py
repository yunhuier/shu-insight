from __future__ import annotations

import uuid
from dataclasses import dataclass, replace

from shu_insight.db.repositories import PersonaRepository
from shu_insight.domain.models import Persona
from shu_insight.profiles.catalog import majors_for

DEMO_PERSONAS = (
    # 前四个是冻结评估集使用的评测画像，id 与 evaluation/dataset.py 的
    # REQUIRED_PERSONA_IDS 一一对应，不可改名或删除。
    Persona(
        "demo-undergrad-research",
        "计算机学院大二·保研科研",
        "本科生",
        "大二",
        "计算机工程与科学学院",
        "不限专业",
        goals=("保研", "科研"),
    ),
    Persona(
        "demo-undergrad-job",
        "计算机学院大四·实习就业",
        "本科生",
        "大四",
        "计算机工程与科学学院",
        "不限专业",
        goals=("就业",),
    ),
    Persona(
        "demo-master",
        "计算机学院研一·科研奖学金",
        "硕士生",
        "研一",
        "计算机工程与科学学院",
        "不限专业",
        goals=("科研",),
    ),
    Persona(
        "demo-freshman",
        "其他学院大一·校园事务",
        "本科生",
        "大一",
        "",
        "不限专业",
    ),
    # 以下六个把“校园事务”拆细并补充常见诉求，只用于界面演示与排序，
    # 不参与冻结评估集，因此不影响既有指标。
    Persona(
        "demo-undergrad-club",
        "其他学院大二·社团活动",
        "本科生",
        "大二",
        "",
        "不限专业",
        goals=("社团活动", "志愿服务"),
        interest_tags=("活动",),
        custom_keywords=("社团", "招新", "志愿"),
    ),
    Persona(
        "demo-undergrad-contest",
        "计算机学院大三·学科竞赛",
        "本科生",
        "大三",
        "计算机工程与科学学院",
        "不限专业",
        goals=("竞赛", "保研"),
        interest_tags=("竞赛",),
        custom_keywords=("竞赛", "挑战杯", "创新创业"),
    ),
    Persona(
        "demo-undergrad-kaoyan",
        "其他学院大四·考研深造",
        "本科生",
        "大四",
        "",
        "不限专业",
        goals=("考研", "升学"),
        interest_tags=("升学",),
        custom_keywords=("考研", "复试", "调剂"),
    ),
    Persona(
        "demo-undergrad-abroad",
        "其他学院大三·出国交流",
        "本科生",
        "大三",
        "",
        "不限专业",
        goals=("出国交流", "语言考试"),
        interest_tags=("升学",),
        custom_keywords=("交换", "交流", "留学"),
    ),
    Persona(
        "demo-undergrad-military",
        "其他学院大二·参军入伍",
        "本科生",
        "大二",
        "",
        "不限专业",
        goals=("参军入伍", "退役复学"),
        custom_keywords=("征兵", "入伍", "退役"),
    ),
    Persona(
        "demo-master-job",
        "计算机学院研三·求职实习",
        "硕士生",
        "研三",
        "计算机工程与科学学院",
        "不限专业",
        goals=("就业", "实习"),
        interest_tags=("就业",),
        custom_keywords=("招聘", "实习", "校招"),
    ),
)

_VALID_GRADES = {
    "不限": {"不限"},
    "本科生": {"不限", "大一", "大二", "大三", "大四"},
    "硕士生": {"不限", "研一", "研二", "研三"},
    "博士生": {"不限", "博一", "博二", "博三", "博四", "博五"},
}


class PersonaValidationError(ValueError):
    """Raised when an editable local persona violates a bounded option."""


@dataclass(frozen=True, slots=True)
class NoProfileQuery:
    persona_id: None
    interest_tags: tuple[str, ...]
    custom_keywords: tuple[str, ...]


def _normalize_terms(
    values: tuple[str, ...], *, label: str, maximum: int
) -> tuple[str, ...]:
    if len(values) > maximum:
        raise PersonaValidationError(f"{label}最多 {maximum} 个")
    normalized = tuple(value.strip() for value in values)
    if any(not value for value in normalized):
        raise PersonaValidationError(f"{label}不能包含空白项")
    folded = tuple(value.casefold() for value in normalized)
    if len(set(folded)) != len(folded):
        raise PersonaValidationError(f"{label}不能包含重复项")
    return normalized


def validate_persona(persona: Persona) -> Persona:
    persona_id = persona.id.strip()
    label = persona.label.strip()
    if not persona_id:
        raise PersonaValidationError("画像 ID 不能为空")
    if not label:
        raise PersonaValidationError("画像名称不能为空")
    if persona.education_level not in _VALID_GRADES:
        raise PersonaValidationError("培养层次不在可选范围")
    if persona.grade not in _VALID_GRADES[persona.education_level]:
        raise PersonaValidationError("年级与培养层次不匹配")
    if persona.major not in majors_for(persona.education_level, persona.college):
        raise PersonaValidationError("专业不在当前可选范围")

    goals = _normalize_terms(persona.goals, label="目标", maximum=8)
    interest_tags = _normalize_terms(persona.interest_tags, label="兴趣标签", maximum=8)
    custom_keywords = _normalize_terms(
        persona.custom_keywords,
        label="自定义关键词",
        maximum=5,
    )
    return replace(
        persona,
        id=persona_id,
        label=label,
        college=persona.college.strip(),
        campus=persona.campus.strip() or "不限",
        goals=goals,
        interest_tags=interest_tags,
        custom_keywords=custom_keywords,
    )


class PersonaService:
    def __init__(self, repository: PersonaRepository) -> None:
        self.repository = repository

    def list_all(self) -> list[Persona]:
        return self.repository.list_all()

    def save(self, persona: Persona) -> Persona:
        validated = validate_persona(persona)
        self.repository.save(validated)
        return validated

    def create(self, persona: Persona) -> Persona:
        created = replace(persona, id=str(uuid.uuid4()))
        return self.save(created)

    def update(self, persona: Persona) -> Persona:
        if not any(existing.id == persona.id for existing in self.repository.list_all()):
            raise KeyError(f"persona not found: {persona.id}")
        return self.save(persona)

    def copy(self, persona_id: str) -> Persona:
        original = self._require(persona_id)
        copied = replace(
            original,
            id=str(uuid.uuid4()),
            label=f"{original.label} 副本",
        )
        return self.save(copied)

    def delete(self, persona_id: str) -> None:
        if not self.repository.delete(persona_id):
            raise KeyError(f"persona not found: {persona_id}")

    def delete_all(self) -> None:
        self.repository.delete_all()

    def reset(self) -> tuple[Persona, ...]:
        self.repository.delete_all()
        return tuple(self.save(persona) for persona in DEMO_PERSONAS)

    def reset_persona(self, persona_id: str) -> Persona:
        existing = self._require(persona_id)
        default = next((persona for persona in DEMO_PERSONAS if persona.id == persona_id),
                       Persona(persona_id, existing.label))
        return self.save(default)

    def no_profile_query(
        self,
        interest_tags: tuple[str, ...],
        custom_keywords: tuple[str, ...],
    ) -> NoProfileQuery:
        return NoProfileQuery(
            persona_id=None,
            interest_tags=_normalize_terms(
                interest_tags,
                label="兴趣标签",
                maximum=8,
            ),
            custom_keywords=_normalize_terms(
                custom_keywords,
                label="自定义关键词",
                maximum=5,
            ),
        )

    def _require(self, persona_id: str) -> Persona:
        for persona in self.repository.list_all():
            if persona.id == persona_id:
                return persona
        raise KeyError(f"persona not found: {persona_id}")
