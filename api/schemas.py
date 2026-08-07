from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from db.models import ClaimStatus


class ClaimCreateResponse(BaseModel):
    claim_id: uuid.UUID
    status: ClaimStatus


class ClaimDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    claim_id: uuid.UUID = Field(validation_alias="id")
    status: ClaimStatus
    result: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
