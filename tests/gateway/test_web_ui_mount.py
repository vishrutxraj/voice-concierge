"""
The React UI is served by the gateway at /app, and must never be able to take
the service (or the Gradio fallback UI) down with it.

These tests deliberately don't depend on a real `npm run build` having been
run -- CI for the Python side has no Node. find_web_dist() is tested against
temp directories instead.
"""

from __future__ import annotations

import pytest
from app.main import app, find_web_dist
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _fresh_settings():
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_find_web_dist_uses_the_configured_directory(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>")
    monkeypatch.setenv("WEB_DIST_DIR", str(tmp_path))
    assert find_web_dist() == tmp_path


def test_find_web_dist_ignores_a_configured_directory_without_index_html(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_DIST_DIR", str(tmp_path))  # exists, but nothing built in it
    monkeypatch.setattr("app.main._DEFAULT_WEB_DIST", tmp_path / "nothing-here")
    assert find_web_dist() is None


def test_find_web_dist_falls_back_to_the_repo_build_output(tmp_path, monkeypatch):
    default = tmp_path / "dist"
    default.mkdir()
    (default / "index.html").write_text("<html></html>")
    monkeypatch.setenv("WEB_DIST_DIR", "")
    monkeypatch.setattr("app.main._DEFAULT_WEB_DIST", default)
    assert find_web_dist() == default


def test_root_redirects_to_whichever_ui_is_available():
    with TestClient(app, follow_redirects=False) as client:
        resp = client.get("/")
    assert resp.status_code in (302, 307)
    expected = "/app/" if find_web_dist() is not None else "/ui/"
    assert resp.headers["location"] == expected


def test_gradio_fallback_ui_is_still_mounted():
    with TestClient(app) as client:
        assert client.get("/ui/").status_code == 200


def test_web_ui_is_served_when_built():
    if find_web_dist() is None:
        pytest.skip("frontend not built (npm run build in services/web)")
    with TestClient(app) as client:
        resp = client.get("/app/")
        assert resp.status_code == 200
        assert '<div id="root">' in resp.text
        # Asset URLs must be rooted at /app/ or the page loads blank.
        assert 'src="/app/assets/' in resp.text
