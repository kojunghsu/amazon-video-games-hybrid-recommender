from recsys.demo import DEMO_HTML


def test_demo_exposes_primary_modes_and_metrics() -> None:
    assert "For a user" in DEMO_HTML
    assert "Similar items" in DEMO_HTML
    assert "0.6428" in DEMO_HTML
    assert 'fetch("/health")' in DEMO_HTML
