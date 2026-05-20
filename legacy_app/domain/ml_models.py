from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, Float, DateTime, func

from legacy_app.db.models import Base


class MLRequest(Base):
    __tablename__ = "ml_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    secid: Mapped[str] = mapped_column(String(16))
    horizon: Mapped[int] = mapped_column(Integer)

    payload_len: Mapped[int] = mapped_column(Integer)
    processing_ms: Mapped[float] = mapped_column(Float)

    status_code: Mapped[int] = mapped_column(Integer)
    error: Mapped[Optional[str]] = mapped_column(String(255))
