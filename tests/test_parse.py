"""Natural-language query parsing into (colour, garment) clauses + scene/formality."""
from retriever.parse import parse


def test_two_clause_and_formality():
    q = parse("A red tie and a white shirt in a formal setting.")
    assert q.formality == "formal"
    pairs = {(c.color, c.garment) for c in q.clauses}
    assert ("red", "tie") in pairs
    assert ("white", "shirt") in pairs


def test_scene_and_single_clause():
    q = parse("Someone wearing a blue shirt sitting on a park bench.")
    assert q.scene == "park"
    assert any(c.color == "blue" and c.garment == "shirt" for c in q.clauses)


def test_scene_only_no_clauses():
    q = parse("Casual weekend outfit for a city walk.")
    assert q.clauses == []
    assert q.scene == "street"
    assert q.formality == "casual"


def test_intensity_modifier_stripped():
    q = parse("A person in a bright yellow raincoat.")
    assert len(q.clauses) == 1
    assert q.clauses[0].color == "yellow"
    assert q.clauses[0].garment == "raincoat"
