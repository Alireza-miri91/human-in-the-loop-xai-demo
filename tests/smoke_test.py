"""Smoke checks for the public HiL XAI demo."""

from src.router_app import CONTROL_GROUP, TREATMENT_GROUP, router


def test_router_health() -> None:
    client = router.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.text == "OK"


def test_public_group_names() -> None:
    assert CONTROL_GROUP == "control"
    assert TREATMENT_GROUP == "treatment"
