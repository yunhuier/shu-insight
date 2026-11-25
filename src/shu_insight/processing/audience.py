from __future__ import annotations

import re
from dataclasses import dataclass

_TARGET = re.compile(
    r"(?:面向|仅限|限于|(?:参赛|报名|申请|参加|适用|征集)对象\s*[:：为是]?|"
    r"对象\s*[:：])(?P<target>[^，,。；;\n]{1,60})"
)
_TEACHER = re.compile(r"教师|教职工")
_STUDENT = re.compile(r"学生|本科(?:生)?(?:新生|毕业(?:生|班))|本科生|研究生|硕士生|博士生|师生")
_NEGATED_CUE = re.compile(r"(?:不|非|未|禁止|不得|不再)$")
_TEACHER_EVENT = re.compile(
    r"(?:教师|教职工)(?:教学)?(?:竞赛|比赛|大赛)"
    r"(?:报名|申报|选拔|安排|实施|的?通知|$)"
)
_TEACHER_PARTICIPATION = re.compile(
    r"(?:推荐|选派|组织)(?:本校|我校|全校)?(?:教师|教职工)(?:参加|参与|报名|申报)"
    r"[^，,。；;\n]{0,60}(?:竞赛|比赛|大赛)"
)
_GRADUATING = re.compile(r"毕业班|本科(?:生)?(?:应届)?毕业生")
_UNDERGRADUATE_STAGES = (
    (re.compile(r"本科(?:生)?新生"), "大一"),
)


@dataclass(frozen=True, slots=True)
class AudienceScope:
    teacher_only: bool = False
    undergraduate_grades: tuple[str, ...] = ()
    graduating_students: bool = False


def explicit_audience_scope(title: str, summary: str) -> AudienceScope:
    """Read bounded audience phrases; callers must omit unverified body text.

    Titles can name the administrative audience directly. Body role mentions
    need an audience cue, so a speaker or supervisor does not become a restriction.
    """
    targets = [
        match["target"] for text in (title, summary) for match in _TARGET.finditer(text)
        if not _NEGATED_CUE.search(text[max(0, match.start() - 4):match.start()])
    ]
    student_audience = any(
        _STUDENT.search(target) for target in targets
    ) or "师生" in title or bool(re.search(r"教师[与和及、]学生|学生[与和及、]教师", title))
    teacher_only = not student_audience and (
        bool(_TEACHER_EVENT.search(title))
        or bool(_TEACHER_PARTICIPATION.search(title))
        or any(_TEACHER.search(target) and not _STUDENT.search(target) for target in targets)
    )
    grades = tuple(dict.fromkeys(
        grade for text in (title, *targets)
        for pattern, grade in _UNDERGRADUATE_STAGES if pattern.search(text)
    ))
    return AudienceScope(
        teacher_only=teacher_only, undergraduate_grades=grades,
        graduating_students=any(_GRADUATING.search(text) for text in (title, *targets)),
    )
