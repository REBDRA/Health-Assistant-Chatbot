import pytest
from ai_service import Doctor, HealthResponse, QueryIntent, sanitize_search_term
from app import get_stars, doctor_completeness_score, clean_html


def test_sanitize_search_term():
    raw = "What is the best treatment for migraine in Kolkata?!?"
    cleaned = sanitize_search_term(raw)
    assert len(cleaned.split()) <= 6
    assert "?" not in cleaned
    assert "!" not in cleaned


def test_doctor_model():
    doc = Doctor(
        name="Dr. Jane Doe",
        phone="+91 9876543210",
        location="Salt Lake, Kolkata",
        rating="4.8/5",
        link="https://example.com/dr-jane",
    )
    assert doc.name == "Dr. Jane Doe"
    assert doc.rating == "4.8/5"
    assert doc.phone == "+91 9876543210"


def test_health_response_schema():
    resp = HealthResponse(
        is_valid_query=True,
        query_type="symptom_triage",
        direct_answer="Overview",
        remedies=["Drink fluids", "Rest"],
        advice="Seek medical care if fever persists",
        doctors=[],
    )
    assert resp.is_valid_query is True
    assert len(resp.remedies) == 2


def test_get_stars():
    assert "⭐" in get_stars("4.5/5")
    assert "⭐⭐⭐⭐" in get_stars("invalid_rating")


def test_doctor_completeness_score():
    complete_doc = {
        "name": "Dr. Smith (Cardiologist)",
        "phone": "+1 555-1234",
        "location": "Downtown Clinic",
        "rating": "4.9/5",
        "link": "https://clinic.example.com",
    }
    empty_doc = {
        "name": "Unknown",
        "phone": "Visit Website",
        "location": "Not specified",
        "rating": "",
        "link": "",
    }
    assert doctor_completeness_score(complete_doc) > doctor_completeness_score(empty_doc)


def test_clean_html_strips_leading_whitespace():
    indented_html = (
        "    <div class=\"doctor-card\">\n"
        "        <div class=\"doctor-details\">\n"
        "            <div class=\"doctor-detail-item\">📍 Location</div>\n"
        "        </div>\n"
        "    </div>\n"
    )
    cleaned = clean_html(indented_html)
    for line in cleaned.splitlines():
        assert not line.startswith(" "), f"Line still has leading space: {line}"
    assert "<div class=\"doctor-card\">" in cleaned
    assert "📍 Location" in cleaned
