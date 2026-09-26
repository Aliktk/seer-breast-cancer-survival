"""Smoke test: every Streamlit page renders without an exception."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
st_testing = pytest.importorskip("streamlit.testing.v1")
PAGES = sorted((ROOT / "app" / "views").glob("*.py"))


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_renders(page):
    at = st_testing.AppTest.from_file(str(page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]


def test_live_verification_matches_stored_metrics():
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "views" / "verify.py"), default_timeout=180).run()
    assert "❌" not in str(at.dataframe[0].value.to_dict()), "recomputed metrics differ from stored ones"
