from shu_insight.domain.models import Recommendation


def explain(recommendation: Recommendation) -> str:
    if not recommendation.reasons:
        return "暂无明确匹配证据"
    return "；".join(recommendation.reasons)
