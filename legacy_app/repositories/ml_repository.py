# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import select

# from domain.models import MLRequest


# class MLRequestRepository:
#     def __init__(self, session: AsyncSession):
#         self.session = session

#     async def add(self, request: MLRequest) -> None:
#         self.session.add(request)
#         await self.session.commit()

#     async def list_all(self) -> list[MLRequest]:
#         res = await self.session.execute(
#             select(MLRequest).order_by(MLRequest.created_at.desc())
#         )
#         return res.scalars().all()

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from legacy_app.domain.ml_models import MLRequest


class MLRequestRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, request: MLRequest) -> None:
        self.session.add(request)
        await self.session.commit()

    async def list_all(self, limit: int = 100) -> list[MLRequest]:
        res = await self.session.execute(
            select(MLRequest).order_by(MLRequest.created_at.desc()).limit(limit)
        )
        return res.scalars().all()

    async def get_stats_rows(self) -> list[tuple[float, int, int]]:
        """
        Returns list of tuples: (processing_ms, payload_len, horizon)
        """
        res = await self.session.execute(
            select(MLRequest.processing_ms, MLRequest.payload_len, MLRequest.horizon)
        )
        return res.all()
