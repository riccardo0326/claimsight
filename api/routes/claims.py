"""Claim submission and status endpoints."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from api.config import Settings, get_settings
from api.schemas import ClaimCreateResponse, ClaimDetailResponse
from db.models import Claim, ClaimStatus
from db.session import get_db
from worker.tasks import process_claim

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claims", tags=["claims"])

PDF_MAGIC = b"%PDF-"
JPEG_MAGIC = (b"\xff\xd8\xff",)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "application/octet-stream",
}
CHUNK_SIZE = 1024 * 64
# Use literal 422 — Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY.
UNPROCESSABLE = 422


def _image_ext(filename: str) -> str:
    return Path(filename).suffix.lower()


async def _save_validated_image(
    upload: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
    field_name: str,
) -> None:
    """Stream an image upload to disk, validating type/size/magic. Raises 422 on reject."""
    filename = upload.filename or ""
    ext = _image_ext(filename)
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=UNPROCESSABLE,
            detail=f"{field_name}: filename must end with .jpg, .jpeg, or .png",
        )

    content_type = (upload.content_type or "").lower()
    if content_type and content_type not in IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=UNPROCESSABLE,
            detail=(
                f"{field_name}: content type must be image/jpeg or image/png "
                f"(got {content_type})"
            ),
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    magic_checked = False

    try:
        with dest.open("wb") as out:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                if not magic_checked:
                    is_jpeg = any(chunk.startswith(m) for m in JPEG_MAGIC)
                    is_png = chunk.startswith(PNG_MAGIC)
                    if ext in {".jpg", ".jpeg"} and not is_jpeg:
                        raise HTTPException(
                            status_code=UNPROCESSABLE,
                            detail=f"{field_name}: file is not a valid JPEG",
                        )
                    if ext == ".png" and not is_png:
                        raise HTTPException(
                            status_code=UNPROCESSABLE,
                            detail=f"{field_name}: file is not a valid PNG",
                        )
                    if not is_jpeg and not is_png:
                        raise HTTPException(
                            status_code=UNPROCESSABLE,
                            detail=f"{field_name}: file is not a valid image",
                        )
                    magic_checked = True
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=UNPROCESSABLE,
                        detail=f"{field_name}: file exceeds max size of {max_bytes} bytes",
                    )
                out.write(chunk)

        if not magic_checked or total == 0:
            raise HTTPException(
                status_code=UNPROCESSABLE,
                detail=f"{field_name}: empty file",
            )
    except HTTPException:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


async def _save_validated_pdf(
    upload: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
    field_name: str,
) -> None:
    """Stream an upload to disk, validating type/size/magic bytes. Raises 422 on reject."""
    filename = upload.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=UNPROCESSABLE,
            detail=f"{field_name}: filename must end with .pdf",
        )

    content_type = (upload.content_type or "").lower()
    if content_type and content_type not in {
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    }:
        raise HTTPException(
            status_code=UNPROCESSABLE,
            detail=f"{field_name}: content type must be application/pdf (got {content_type})",
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    magic_checked = False

    try:
        with dest.open("wb") as out:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                if not magic_checked:
                    if not chunk.startswith(PDF_MAGIC):
                        raise HTTPException(
                            status_code=UNPROCESSABLE,
                            detail=f"{field_name}: file is not a valid PDF (missing %PDF- header)",
                        )
                    magic_checked = True
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=UNPROCESSABLE,
                        detail=f"{field_name}: file exceeds max size of {max_bytes} bytes",
                    )
                out.write(chunk)

        if not magic_checked or total == 0:
            raise HTTPException(
                status_code=UNPROCESSABLE,
                detail=f"{field_name}: empty file",
            )
    except HTTPException:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


@router.post(
    "",
    response_model=ClaimCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_claim(
    policy_pdf: UploadFile = File(...),
    estimate_pdf: UploadFile = File(...),
    # Real claims should require a narrative; empty allowed for Slice 1 compatibility.
    narrative: str = Form(""),
    # Optional free-text location for NOAA weather (Slice 4). Missing → skip weather.
    incident_location: str | None = Form(None),
    damage_photos: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ClaimCreateResponse:
    claim_id = uuid.uuid4()
    storage_root = Path(settings.storage_dir)
    claim_dir = storage_root / str(claim_id)
    policy_path = claim_dir / "policy.pdf"
    estimate_path = claim_dir / "estimate.pdf"
    max_bytes = settings.max_upload_mb * 1024 * 1024

    # FastAPI may pass a single UploadFile when only one file is sent.
    if isinstance(damage_photos, UploadFile):
        photo_uploads: list[UploadFile] = [damage_photos]
    else:
        photo_uploads = list(damage_photos or [])

    saved_photo_paths: list[Path] = []
    try:
        await _save_validated_pdf(
            policy_pdf, policy_path, max_bytes=max_bytes, field_name="policy_pdf"
        )
        await _save_validated_pdf(
            estimate_pdf, estimate_path, max_bytes=max_bytes, field_name="estimate_pdf"
        )
        for i, photo in enumerate(photo_uploads):
            ext = _image_ext(photo.filename or "") or ".jpg"
            if ext == ".jpeg":
                ext = ".jpg"
            dest = claim_dir / f"damage_photo_{i}{ext}"
            await _save_validated_image(
                photo, dest, max_bytes=max_bytes, field_name="damage_photos"
            )
            saved_photo_paths.append(dest)
    except HTTPException:
        # Clean up any partial claim directory.
        if claim_dir.exists():
            for child in claim_dir.iterdir():
                child.unlink(missing_ok=True)
            claim_dir.rmdir()
        raise

    location = (incident_location or "").strip() or None

    claim = Claim(
        id=claim_id,
        status=ClaimStatus.pending,
        input_paths={
            "policy_pdf": str(policy_path),
            "estimate_pdf": str(estimate_path),
            "damage_photos": [str(p) for p in saved_photo_paths],
        },
        narrative=narrative or "",
        incident_location=location,
        result=None,
    )
    db.add(claim)
    db.commit()
    db.refresh(claim)

    process_claim.delay(str(claim.id))
    logger.info("Enqueued process_claim for claim_id=%s", claim.id)

    return ClaimCreateResponse(claim_id=claim.id, status=claim.status)


@router.get("/{claim_id}", response_model=ClaimDetailResponse)
def get_claim(claim_id: uuid.UUID, db: Session = Depends(get_db)) -> ClaimDetailResponse:
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found")
    return ClaimDetailResponse.model_validate(claim)
