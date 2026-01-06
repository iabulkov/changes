# import os
# import asyncio
# from statsmodels.tsa.statespace.sarimax import SARIMAXResults


# class MLService:
#     def __init__(self, model_path: str = "artifacts/sarimax_sber.pkl"):
#         self.model_path = model_path
#         self._model = None

#     async def load(self) -> None:
#         if self._model is not None:
#             return

#         def _load_sync():
#             if not os.path.exists(self.model_path):
#                 raise FileNotFoundError(self.model_path)
#             return SARIMAXResults.load(self.model_path)

#         self._model = await asyncio.to_thread(_load_sync)

#     async def forecast(self, secid: str, horizon: int) -> list[float]:
#         # Модель обучена под SBER на данный момент
#         if secid != "SBER":
#             raise ValueError("unsupported secid")

#         await self.load()

#         def _predict_sync():
#             pred = self._model.get_forecast(steps=horizon).predicted_mean
#             return [float(x) for x in pred]

#         return await asyncio.to_thread(_predict_sync)
import asyncio
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


class MLService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    async def _load_model(self):
        if self._model is None:
            self._model = await asyncio.to_thread(
                SARIMAXResults.load, self.model_path
            )

    async def predict(self, secid: str, horizon: int) -> list[float]:
        if secid != "SBER":
            raise ValueError("unsupported secid")

        await self._load_model()

        def _predict():
            return (
                self._model
                .get_forecast(steps=horizon)
                .predicted_mean
                .astype(float)
                .tolist()
            )

        return await asyncio.to_thread(_predict)
