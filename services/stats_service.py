from repositories.ml_repository import MLRequestRepository

def _quantile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])

class StatsService:
    def __init__(self, repo: MLRequestRepository):
        self.repo = repo

    async def get_stats(self) -> dict:
        rows = await self.repo.get_stats_rows()

        times = [float(t) for (t, _, _) in rows if t is not None]
        payload_lens = [int(pl) for (_, pl, _) in rows if pl is not None]
        horizons = [int(h) for (_, _, h) in rows if h is not None]

        times_sorted = sorted(times)

        timings = {
            "mean": (sum(times_sorted) / len(times_sorted)) if times_sorted else None,
            "p50": _quantile(times_sorted, 50),
            "p95": _quantile(times_sorted, 95),
            "p99": _quantile(times_sorted, 99),
        }

        inputs = {
            "payload_len": {
                "count": len(payload_lens),
                "mean": (sum(payload_lens) / len(payload_lens)) if payload_lens else None,
                "min": min(payload_lens) if payload_lens else None,
                "max": max(payload_lens) if payload_lens else None,
            },
            "horizon": {
                "count": len(horizons),
                "mean": (sum(horizons) / len(horizons)) if horizons else None,
                "min": min(horizons) if horizons else None,
                "max": max(horizons) if horizons else None,
            },
        }

        return {
            "count": len(times_sorted),
            "timings_ms": timings,
            "inputs": inputs,
        }
