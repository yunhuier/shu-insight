from __future__ import annotations

import re
from dataclasses import dataclass

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "phone": re.compile(r"(?<!\d)1[3-9](?:\s*\d){9}(?!\d)"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "id_card": re.compile(r"(?<!\d)\d(?:\s*\d){16}\s*[0-9Xx](?!\d)"),
    "student_number": re.compile(r"(?:学号|报名号|准考证号)\s*[:：]?\s*\d(?:\s*\d){7,17}"),
}

RISKY_CONTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "person_list": re.compile(r"(?:人员|学生|获奖|复试|录取|推荐)名单"),
    "discipline": re.compile(r"处分|退学"),
    "health": re.compile(r"体检结果|健康信息"),
    "financial_hardship": re.compile(r"困难认定"),
}

# Ingestion uses a conservative gate for personal-result domains. Only a
# clearly procedural clause is exempt; a separate result clause still blocks
# the whole notice. AI output and frozen snapshot import keep the stricter
# scan_sensitive_text gate.
PUBLIC_NOTICE_RISK_PATTERNS: dict[str, re.Pattern[str]] = {
    "person_list": re.compile(
        r"(?:拟录取|录取|复试|考核|推免|推荐|获奖|受助|学生|人员|候选人)"
        r".{0,12}(?:名单|名册|人选)"
        r"|(?:奖学金|助学金|评选结果|评审结果|获奖|受助|录取|资助)"
        r"|授予(?:硕士|博士)学位人员"
    ),
    "discipline": re.compile(r"处分|退学"),
    "health": re.compile(r"体检结果|健康信息|健康检查结果"),
    "financial_hardship": re.compile(r"家庭经济困难|困难认定"),
}

PUBLIC_PROCEDURE_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"(?:录取|复试|考核).{0,16}(?:申请流程|办理流程|流程|申请条件|提交材料|工作安排)",
    r"(?:名单|名册|人选).{0,12}(?:填报|提交|整理|审核|管理|公布后.{0,10}申诉|公示工作安排)",
    r"(?:填报|提交|整理|审核).{0,12}(?:名单|名册|人选)",
    r"(?:名单|结果).{0,12}将在.{0,24}另行公示",
    r"(?:名单|结果).{0,12}(?:公示|公布)(?:后|前).{0,16}(?:申诉|异议|流程)",
    r"(?:名单|结果).{0,12}(?:公示|公布)(?:后|前)$",
    r"(?:名单|结果).{0,12}(?:公示|公布).{0,8}(?:工作安排|安排|异议受理|申诉流程)",
    r"(?:奖学金|助学金).{0,24}(?:申请|申报|评审办法|评选办法|评选工作|材料|截止|流程|工作安排)",
    r"(?:组织|开展).{0,24}(?:奖学金|助学金).{0,12}评选",
    r"(?:退学|处分).{0,16}(?:手续|办理流程|管理办法|处理决定流程|送达程序|申诉渠道)",
    r"(?:家庭经济困难|困难认定).{0,16}(?:申请|申报|材料|办理流程|认定工作|工作安排)",
    r"(?:体检结果|健康检查结果).{0,12}(?:查询办法|查询流程|工作安排)",
))

DIRECT_PERSONAL_RESULT = re.compile(
    r"(?:现将|现对).{0,40}(?:名单|结果|人选)"
    r"(?:[（(][^）)]{1,12}[）)])?(?:予以|进行|的)?"
    r"(?:公示|公布|公告|如下|[:：])"
    r"|(?:拟推荐|拟录取|给予|决定对).{0,40}(?:获得|处分|退学|奖学金|助学金)"
    r"|[\u4e00-\u9fff]{2,4}(?:、[\u4e00-\u9fff]{2,4})*同学.{0,24}"
    r"(?:获得|受到|通过|给予|作退学处理|被录取|获奖)"
    r"|[\u4e00-\u9fff]{2,4}(?:、[\u4e00-\u9fff]{2,4})*获得.{0,20}"
    r"(?:奖学金|助学金|资助|奖励)"
    r"|[\u4e00-\u9fff]{2,4}(?:、[\u4e00-\u9fff]{2,4}){0,4}"
    r"(?:同学|同志)?(?:被|荣获|获得|受到|通过|当选)"
    r"|[\u4e00-\u9fff]{2,4}(?:、[\u4e00-\u9fff]{2,4})*为(?:本次)?"
    r"(?:获奖|受助|拟录取|录取)"
)

DROP_LINE_PREFIXES = (
    "联系人",
    "联系方式",
    "组长",
    "副组长",
    "组员",
    "名单",
)

REPLACEMENTS = {
    "phone": "[REDACTED_PHONE]",
    "email": "[REDACTED_EMAIL]",
    "id_card": "[REDACTED_ID]",
    "student_number": "[REDACTED_STUDENT_NUMBER]",
}

# Bounded title grammar: honorific/education-role evidence, or a common surname
# in an administrative outcome. Chinese word length alone is not person evidence.
# This intentionally does not attempt general name recognition.
_TITLE_NAMES = r"[\u4e00-\u9fff]{2,4}(?:、[\u4e00-\u9fff]{2,4}){0,4}"
_TITLE_BARE_NAME = re.compile(
    r"(?:欧阳|司马|上官|诸葛|东方|皇甫|尉迟|公孙|慕容|司徒|司空)"
    r"[\u4e00-\u9fff]{1,2}"
    r"|[王李张刘陈杨黄赵吴周徐孙马朱胡郭何林高罗郑梁谢宋唐许韩冯邓曹彭曾萧肖"
    r"田董潘袁于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟]"
    r"[\u4e00-\u9fff]{1,3}"
)
_TITLE_PREFIX = r"(?:关于(?:给予|对)?|决定对|给予|对)"
_TITLE_ROLE = r"(?:优秀)?(?:研究生|本科生|学生|考生)"
_TITLE_PERSON_PATTERNS = (
    re.compile(r"(?<![\u4e00-\u9fff、])(?P<prefix>" + _TITLE_PREFIX + r"?(?P<context_role>"
               + _TITLE_ROLE + r")?)(?P<names>" + _TITLE_NAMES
               + r")(?P<role>同学|同志)(?!们)"),
    re.compile(r"(?<![\u4e00-\u9fff、])(?P<prefix>" + _TITLE_PREFIX + r"(?P<context_role>"
               + _TITLE_ROLE + r")?)(?P<names>" + _TITLE_NAMES + r")(?P<role>)"
               + r"(?=获得|荣获|受到|被录取|被记过|被处分|被退学|通过(?:复试|考核|评审))"),
)
_TITLE_GENERIC_AUDIENCE = re.compile(
    r"(?:关于)?(?:请|相关|有关|各位|各|全体|所有|本校|符合条件的)?"
    r"(?:研究生|本科生|学生|师生|新生|考生|候选人|申请人|同学|同志)?"
    r"|(?:我|本|该|各|全|贵)(?:校|院|所|系|班|组)"
)
_TITLE_GENERIC_OBJECT = re.compile(
    r"[\u4e00-\u9fff]{0,2}(?:大学|学院|学校|研究院|研究所|实验室)"
    r"|(?:科研|研究|教学|教育|实验|学术|创新|创业|校级|院级|国家"
    r"|省级|市级|重点|重大|合作|建设|课程|毕业)?"
    r"(?:项目|课题|工程|成果|计划|方案|平台|系统|课程|专业"
    r"|团队|集体|单位|部门|中心|基地|机构|课题组|项目组)"
)


@dataclass(frozen=True, slots=True)
class SensitiveScan:
    blocked: bool
    kinds: tuple[str, ...]


def scan_sensitive_text(text: str) -> SensitiveScan:
    kinds = [kind for kind, pattern in PII_PATTERNS.items() if pattern.search(text)]
    kinds.extend(
        kind for kind, pattern in RISKY_CONTENT_PATTERNS.items() if pattern.search(text)
    )
    unique_kinds = tuple(dict.fromkeys(kinds))
    return SensitiveScan(blocked=bool(unique_kinds), kinds=unique_kinds)


def scan_public_notice_text(text: str) -> SensitiveScan:
    """Fail closed on personal-result topics; admit clearly procedural clauses."""
    kinds = [kind for kind, pattern in PII_PATTERNS.items() if pattern.search(text)]
    for clause in re.split(r"[\n。！？；;，,]+", text):
        if not clause.strip():
            continue
        procedural = any(pattern.search(clause) for pattern in PUBLIC_PROCEDURE_PATTERNS)
        direct_result = bool(DIRECT_PERSONAL_RESULT.search(clause))
        for kind, pattern in PUBLIC_NOTICE_RISK_PATTERNS.items():
            if pattern.search(clause) and (direct_result or not procedural):
                kinds.append(kind)
    unique_kinds = tuple(dict.fromkeys(kinds))
    return SensitiveScan(blocked=bool(unique_kinds), kinds=unique_kinds)


def redact_public_contacts(text: str) -> str:
    """Mask public contact details without discarding the surrounding notice."""
    for kind, replacement in (("phone", "[电话已隐藏]"), ("email", "[邮箱已隐藏]")):
        text = PII_PATTERNS[kind].sub(replacement, text)
    return text


def redact_public_identifiers(text: str) -> str:
    """Mask direct identifiers in public titles and safe body previews."""
    text = redact_public_contacts(text)
    # Match full IDs before shorter labelled student numbers.
    for kind in ("id_card", "student_number"):
        text = PII_PATTERNS[kind].sub(REPLACEMENTS[kind], text)
    return text


def redact_public_title(text: str) -> str:
    """Keep public title context while masking recognizable personal subjects."""
    def mask_person(match: re.Match[str]) -> str:
        if _TITLE_GENERIC_AUDIENCE.fullmatch(match["names"]):
            return match[0]
        if not match["role"] and _TITLE_GENERIC_OBJECT.fullmatch(match["names"]):
            return match[0]
        if not (match["role"] or match["context_role"]):
            if not all(_TITLE_BARE_NAME.fullmatch(name) for name in match["names"].split("、")):
                return match[0]
        return match["prefix"] + "[姓名已隐藏]" + match["role"]

    text = redact_public_identifiers(text)
    for pattern in _TITLE_PERSON_PATTERNS:
        text = pattern.sub(mask_person, text)
    return text


def sanitize_for_ai(text: str) -> str:
    kept_lines = [
        line
        for line in text.splitlines()
        if not line.strip().startswith(DROP_LINE_PREFIXES)
    ]
    sanitized = "\n".join(kept_lines)
    for kind, pattern in PII_PATTERNS.items():
        sanitized = pattern.sub(REPLACEMENTS[kind], sanitized)
    return sanitized.strip()

