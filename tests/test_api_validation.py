"""API validation tests — reject non-PDF / oversized uploads with 422."""

from __future__ import annotations

from io import BytesIO


def test_rejects_non_pdf_extension(client, policy_pdf):
    files = {
        "policy_pdf": ("policy.txt", policy_pdf.read_bytes(), "application/pdf"),
        "estimate_pdf": ("estimate.pdf", policy_pdf.read_bytes(), "application/pdf"),
    }
    resp = client.post("/claims", files=files)
    assert resp.status_code == 422
    assert "policy_pdf" in resp.json()["detail"]


def test_rejects_bad_magic_bytes(client, estimate_pdf):
    files = {
        "policy_pdf": ("policy.pdf", b"NOT_A_PDF_CONTENT", "application/pdf"),
        "estimate_pdf": ("estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    resp = client.post("/claims", files=files)
    assert resp.status_code == 422
    assert "valid PDF" in resp.json()["detail"]


def test_rejects_oversized_upload(client, policy_pdf, estimate_pdf, monkeypatch):
    from api.config import get_settings

    monkeypatch.setenv("MAX_UPLOAD_MB", "0")  # effectively 0 MB → any file oversized
    get_settings.cache_clear()

    # Rebuild settings on the claims module dependency — TestClient still uses
    # get_settings(); clearing cache is enough because Depends calls it per request.
    files = {
        "policy_pdf": ("policy.pdf", policy_pdf.read_bytes(), "application/pdf"),
        "estimate_pdf": ("estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    resp = client.post("/claims", files=files)
    # max_bytes = 0 * 1024 * 1024 = 0, first chunk exceeds immediately
    assert resp.status_code == 422
    assert "max size" in resp.json()["detail"]
    get_settings.cache_clear()


def test_rejects_missing_file(client, policy_pdf):
    files = {
        "policy_pdf": ("policy.pdf", policy_pdf.read_bytes(), "application/pdf"),
    }
    resp = client.post("/claims", files=files)
    assert resp.status_code == 422


def test_rejects_wrong_content_type(client, policy_pdf, estimate_pdf):
    files = {
        "policy_pdf": ("policy.pdf", policy_pdf.read_bytes(), "text/plain"),
        "estimate_pdf": ("estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    resp = client.post("/claims", files=files)
    assert resp.status_code == 422
    assert "content type" in resp.json()["detail"]
