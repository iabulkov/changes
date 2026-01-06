from domain.schemas import HistoryItem
from repositories.ml_repository import MLRequestRepository

class HistoryService:
    def __init__(self, repo: MLRequestRepository):
        self.repo = repo

    async def get_history(self, limit: int = 100) -> list[HistoryItem]:
        rows = await self.repo.list_all(limit=limit)
        return [
            HistoryItem(
                id=r.id,
                created_at=r.created_at.isoformat() if r.created_at else "",
                secid=r.secid,
                horizon=r.horizon,
                payload_len=r.payload_len,
                processing_ms=r.processing_ms,
                status_code=r.status_code,
                error=r.error,
            )
            for r in rows
        ]
