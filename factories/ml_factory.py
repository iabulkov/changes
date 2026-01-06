from domain.models import MLRequest
from domain.schemas import ForwardRequest


class MLRequestFactory:
    """
    Factory for MLRequest domain entity
    """

    @staticmethod
    def create_request(
        dto: ForwardRequest,
        payload_len: int,
        processing_ms: float,
        status_code: int,
        error: str | None,
    ) -> MLRequest:
        return MLRequest(
            secid=dto.secid,
            horizon=dto.horizon,
            payload_len=payload_len,
            processing_ms=processing_ms,
            status_code=status_code,
            error=error,
        )
