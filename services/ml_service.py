import os
import asyncio
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

class MLService:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self._cache = {}  # secid -> model

    async def _load(self, secid: str):
        if secid in self._cache:
            return self._cache[secid]

        path = os.path.join(self.artifacts_dir, f"sarimax_{secid}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"model not found for {secid}")

        model = await asyncio.to_thread(SARIMAXResults.load, path)
        self._cache[secid] = model
        return model

    async def predict(self, secid: str, horizon: int) -> list[float]:
        model = await self._load(secid)

        def _predict():
            return model.get_forecast(steps=horizon).predicted_mean.astype(float).tolist()

        return await asyncio.to_thread(_predict)
