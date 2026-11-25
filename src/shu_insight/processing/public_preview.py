from __future__ import annotations

import re
from copy import deepcopy
from datetime import date

from bs4 import NavigableString, Tag

from shu_insight.compliance.detector import (
    PII_PATTERNS,
    PUBLIC_PROCEDURE_PATTERNS,
    redact_public_identifiers,
)

_PERSON_HEADERS = re.compile(r"姓名|学号|身份证|证件号|准考证|报名号|手机号|联系电话|邮箱")
_TABLE_NUMBER = re.compile(r"(?<!\d)\d{8,18}[Xx]?(?!\d)")
_STORED_PERSON_TABLE_HEADER = re.compile(
    r"姓名[\s|、，,]*(?:(?:原)?(?:学士|硕士|博士)?学位证书(?:编号|号码)"
    r"|学号|身份证(?:件)?号(?:码)?|证件号|准考证号|报名号)"
)
_STORED_PERSON_TABLE_ROW = re.compile(
    r"(?<!\d)\d{1,4}\s*[\u4e00-\u9fff·]{2,12}\s*\d{8,18}[Xx]?(?!\d)"
)
_NAME_FIELD = re.compile(r"姓名\s*[:：]|(?:获奖人|受助人|录取人)\s*[:：]")
_APPLICANT_HEADER = re.compile(r"申请人|候选人|考生|学生")
_RESULT_HEADER = re.compile(r"等级|名次|奖项|评选结果|成绩|录取结果")
_IDENTITY_LABEL = re.compile(r"(?P<subject>[\u4e00-\u9fff、]{2,40}(?:同学|同志))(?!们)")
_RESULT_ACTION_PATTERN = (
    r"荣获|获得|受到|当选|获奖|作退学处理|记过|被录取|录取为"
    r"|(?:被)?评为|(?:被)?授予|入选|认定为|推荐为"
    r"|为(?:本次)?(?:获奖|受助|拟录取|录取)"
)
_ACTION = re.compile(r"(?P<route>通过)|(?P<result>" + _RESULT_ACTION_PATTERN + r")")
_RESULT_NOUN_COMPOUND = re.compile(
    r"授予申请(?:材料|流程|手续)|获奖(?:条件|资格|要求|流程|办法)"
)
_GENERIC_SUBJECT = re.compile(
    r"(?:(?:请)?(?:相关|有关|各位|各学院|各|全体|所有|本校|符合条件的)?"
    r"(?:申请人|申请者|申报人|报名者|获奖者|获奖人|受助者"
    r"|学生|研究生|本科生|新生|师生|考生|候选人|参加者|参与者|同学|同志"
    r"|学院|学校|单位|部门|导师|本人|个人|材料|通知|本通知)"
    r"(?:同学|同志)?(?:们)?)?"
    r"(?:均|可以|可|请|应当|应|必须|须|需要|需|将|已|再|然后)*"
)
_PROCEDURAL_ROUTE = re.compile(r".{0,16}(?:平台|系统|渠道|网站|官网|窗口|邮箱).{0,16}"
                               r"(?:提交|确认|办理|查询|申请|上传|核对)")
_THROUGH_OUTCOME = re.compile(r"通过(?:复试|考核|审核|评审|评选|认定|考试|答辩|面试|初试|选拔)")
_PROCEDURAL_TAIL = re.compile(
    r"后(?P<tail>(?:必须|应当|可以|需要|须|需|应|请|可)+通过)"
)
_PROCEDURE_CONTEXT = re.compile(r"提交|材料|通知|流程|安排|时间|截止|办理|确认|核对")
_ADMINISTRATIVE_RESULT = re.compile(
    r"(?:决定对|给予|拟推荐|拟录取)[\u4e00-\u9fff]{2,4}(?:同学|同志)?"
    r".{0,12}(?:处分|退学|获得|荣获)"
)
_OUTCOME_START = re.compile(r"^(?:通过|" + _RESULT_ACTION_PATTERN + r")")
_RESULT_ANNOUNCEMENT = re.compile(
    r"(?:名单|名册|人选|评选结果|评审结果|体检结果|健康检查结果|认定结果)"
    r".{0,16}(?:公示|公布|公告|如下|[:：])"
    r"|^(?:获得|荣获|受到|被录取).{0,24}(?:奖学金|助学金|处分|奖励)"
)
_BLOCKS = {"p", "div", "li", "tr", "section", "h1", "h2", "h3", "h4", "br"}
_CONTACT_TAIL = re.compile(
    r"联系人|联系方式|联系电话|联系邮箱|办公电话|(?:副)?组长\s*[:：]|组员\s*[:：]"
)
_LANDLINE = re.compile(r"(?<!\d)(?:0\d{2,3}[- ]?)?[2-9]\d{7}(?:[-*转]\d{1,6})?(?!\d)")


def redact_public_landlines(text: str) -> str:
    def redact(match: re.Match[str]) -> str:
        token = match[0]
        if re.fullmatch(r"20\d{6}", token):
            try:
                date.fromisoformat(f"{token[:4]}-{token[4:6]}-{token[6:]}")
                return token
            except ValueError:
                pass
        return "[电话已隐藏]"

    return _LANDLINE.sub(redact, text)


def safe_stored_summary(text: str) -> str:
    """Project historical tables and contact tails without rewriting stored proof."""
    header = _STORED_PERSON_TABLE_HEADER.search(text)
    if header:
        rows = text[header.end():]
        if (_STORED_PERSON_TABLE_ROW.search(rows)
                or len(_TABLE_NUMBER.findall(rows)) >= 2):
            return ""
    # Flattened contact/roster tails cannot safely be separated back into names
    # and public prose. The same projection protects cards and version evidence.
    text = _CONTACT_TAIL.split(text, maxsplit=1)[0].rstrip()
    return redact_public_identifiers(redact_public_landlines(text))


def _personal_identity_or_outcome(text: str) -> bool:
    # Identity is checked independently: role prefixes do not hide a named
    # person, even when that person is following an otherwise public procedure.
    if any(not _GENERIC_SUBJECT.fullmatch(match["subject"])
           for match in _IDENTITY_LABEL.finditer(text)):
        return True
    for clause in re.split(r"[，,：:；;。！？]", text):
        # Match only the action, never its whole object: later result spans
        # must still be visited after an otherwise public platform route.
        audience = _GENERIC_SUBJECT.match(clause)
        # A result word inside a generic role noun (e.g. 获奖者) is not a verb.
        for action in _ACTION.finditer(clause, audience.end() if audience else 0):
            if action["result"] is not None:
                # A nominal complement changes this token's role, not the
                # safety of the whole clause; later predicates still run.
                if _RESULT_NOUN_COMPOUND.match(clause, action.start()):
                    continue
                return True
            generic = _GENERIC_SUBJECT.fullmatch(clause[:action.start()].strip())
            result = _THROUGH_OUTCOME.match(clause, action.start())
            if result:
                if generic and re.match(r"后|者|方可|即可|需|须|应|可", clause[result.end():]):
                    continue
                return True
            if generic and _PROCEDURAL_ROUTE.search(clause[action.end():]):
                continue
            # An unresolved subject or non-procedural action is unsafe.
            return True
    return False


def _identity_fragment(text: str) -> bool:
    """Keep bounded unresolved subject context without guessing a name shape."""
    candidate = text.rstrip("。.")
    return bool(
        re.fullmatch(r"[\w·、 -]{2,40}", candidate)
        and not _GENERIC_SUBJECT.fullmatch(candidate)
        and not _PROCEDURE_CONTEXT.search(candidate)
    )


def _personal_results(text: str) -> bool:
    """Require identity/result evidence; admission or award topics alone are safe."""
    return bool(
        _NAME_FIELD.search(text)
        or _ADMINISTRATIVE_RESULT.search(text)
        or _personal_identity_or_outcome(text)
        or (_RESULT_ANNOUNCEMENT.search(text)
            and not any(pattern.search(text) for pattern in PUBLIC_PROCEDURE_PATTERNS))
    )


def _safe_result_clause(text: str) -> str:
    if not _personal_results(text):
        return text
    audience = _GENERIC_SUBJECT.match(text)
    first = _ACTION.search(text, audience.end() if audience else 0)
    if (first is None or not _GENERIC_SUBJECT.fullmatch(text[:first.start()].strip())
            or (first["result"] is None and not _THROUGH_OUTCOME.match(text, first.start()))):
        return ""
    # Only a generic conditional result prefix may yield an independent tail.
    # Rescan the entire candidate so a later personal result cannot be rescued.
    for boundary in _PROCEDURAL_TAIL.finditer(text, first.end()):
        tail = text[boundary.start("tail"):]
        if _PROCEDURAL_ROUTE.search(tail) and not _personal_results(tail):
            return tail
    return ""


def safe_public_preview(content_node: Tag) -> str:
    """Flatten a disposable DOM copy only after removing personal-result material."""
    node = deepcopy(content_node)
    for unsafe in node.select("script, style, iframe, object, embed, noscript"):
        unsafe.decompose()
    for table in list(node.select("table")):
        if table.parent is None:
            continue
        text = table.get_text("", strip=True)
        if (_PERSON_HEADERS.search(text) or _TABLE_NUMBER.search(text)
                or any(pattern.search(text) for pattern in PII_PATTERNS.values())
                or (_APPLICANT_HEADER.search(text) and _RESULT_HEADER.search(text))
                or _personal_results(text)):
            table.decompose()
    for listing in list(node.select("ul, ol, dl")):
        if listing.parent is None:
            continue
        previous = listing.find_previous_sibling()
        if (_personal_results(listing.get_text("", strip=True))
                or (previous is not None
                    and _personal_results(previous.get_text("", strip=True)))):
            listing.decompose()

    chunks: list[str] = []

    def flatten(current: Tag) -> None:
        for child in current.children:
            if isinstance(child, NavigableString):
                chunks.append(str(child))
            elif isinstance(child, Tag):
                block = child.name in _BLOCKS
                if block:
                    chunks.append("\n")
                flatten(child)
                if block:
                    chunks.append("\n")
                elif child.name in {"td", "th"}:
                    chunks.append(" ")

    flatten(node)
    # Redact while labels and values still share context across DOM blocks.
    lines = [" ".join(line.split()) for line in
             redact_public_identifiers("".join(chunks)).splitlines() if line.strip()]
    personal = [_personal_results(line) for line in lines]
    joined_private: set[int] = set()
    for index in range(len(lines) - 1):
        if (_identity_fragment(lines[index]) and _OUTCOME_START.search(lines[index + 1])
                and _personal_results(lines[index] + lines[index + 1])):
            joined_private.update((index, index + 1))
            personal[index] = personal[index + 1] = True
    safe: list[str] = []
    for index, line in enumerate(lines):
        if index in joined_private:
            continue
        if _identity_fragment(line):
            neighbors = []
            for step in (-1, 1):
                neighbor = index + step
                while (0 <= neighbor < len(lines) and _identity_fragment(lines[neighbor])
                       and not personal[neighbor]):
                    neighbor += step
                if 0 <= neighbor < len(lines):
                    neighbors.append(personal[neighbor])
            if any(neighbors):
                continue
        kept: list[str] = []
        for clause in re.split(r"(?<=[。！？!?；;])", line):
            clause = " ".join(clause.split())
            if not clause:
                continue
            cleaned = _safe_result_clause(clause)
            if cleaned:
                kept.append(redact_public_identifiers(cleaned))
        if kept:
            safe.append("".join(kept))
    return " ".join(safe)
