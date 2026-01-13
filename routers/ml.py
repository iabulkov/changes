import time
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from domain.schemas import ForwardRequest, ForwardResponse
from factories.ml_factory import MLRequestFactory
from repositories.ml_repository import MLRequestRepository
from services.ml_service import MLService
from domain.schemas import HistoryItem
from services.history_service import HistoryService
from fastapi import Query
from domain.schemas import StatsResponse
from services.stats_service import StatsService


router = APIRouter(tags=["ML"])


@router.post("/forward", response_model=ForwardResponse)
async def forward(
    request: Request,
    dto: ForwardRequest,
    session: AsyncSession = Depends(get_session),
):
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(status_code=400, detail="bad request")

    try:
        payload = await request.json()
        dto = ForwardRequest.model_validate(payload)
    except Exception:
        raise HTTPException(status_code=400, detail="bad request")

    service = MLService("artifacts")
    repo = MLRequestRepository(session)

    start = time.perf_counter()
    status_code = 200
    error = None

    try:
        forecast = await service.predict(dto.secid, dto.horizon)
    except Exception:
        status_code = 403
        error = "модель не смогла обработать данные"
        raise HTTPException(status_code=403, detail=error)
    finally:
        processing_ms = (time.perf_counter() - start) * 1000
        payload_len = len(await request.body())

        log = MLRequestFactory.create_request(
            dto=dto,
            payload_len=payload_len,
            processing_ms=processing_ms,
            status_code=status_code,
            error=error,
        )
        await repo.add(log)

    return ForwardResponse(
        secid=dto.secid,
        horizon=dto.horizon,
        target="LOG_RETURN",
        forecast=forecast,
        model="SARIMAX(1,0,1)",
    )

@router.get("/history", response_model=list[HistoryItem])
async def history(
    limit: int = Query(default=100, ge=1, le=1000),
    session: AsyncSession = Depends(get_session),
):
    repo = MLRequestRepository(session)
    service = HistoryService(repo)
    return await service.get_history(limit=limit)

@router.get("/stats", response_model=StatsResponse)
async def stats(
    session: AsyncSession = Depends(get_session),
):
    repo = MLRequestRepository(session)
    service = StatsService(repo)
    return await service.get_stats()
