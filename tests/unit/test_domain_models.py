from dataclasses import fields

from shu_insight.domain.models import DeadlineConfidence, Persona


def test_persona_does_not_expose_identity_fields() -> None:
    assert {field.name for field in fields(Persona)}.isdisjoint(
        {"name", "student_id", "phone", "score"}
    )


def test_persona_collection_defaults_are_immutable_tuples() -> None:
    persona = Persona(id="demo", label="演示角色")

    assert persona.goals == ()
    assert persona.interest_tags == ()
    assert persona.custom_keywords == ()


def test_deadline_confidence_has_explicit_unknown_state() -> None:
    assert DeadlineConfidence.UNKNOWN.value == "unknown"
