# Эксперименты с LSTM-архитектурами и наборами фичей.

# Данные приходят из API, напрямую к Postgres никто не лезет.
# Метрики каждого прогона пишутся отдельной строкой в журнал dl_experiments.csv.

# Параметры передаются через CLI:
#   python DL_Experiments.py --ticker SBER --features quotes_macro_fund_sent --scheme attention --horizon 1 --lookback 60


import argparse
import csv
import os
import sys
import time

import httpx
import math
import torch
import numpy as np
import pandas as pd

from datetime import date, timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --ticker   Тикер
TICKER = "SBER"

# --features   Набор фичей
FEATURES = "quotes"

# --scheme   Архитектура
SCHEME = "vanilla"

# --lookback   Длина rolling window (для horizon = 1 пробую 15/30/60/90, для horizon = 20 - 60/120/240)
LOOKBACK = 30

# --horizon   Горизонт прогноза в торговых днях
#   1 - прогноз на следующий день (один скаляр)
#   20 - прогноз на месяц вперед (вектор из 20 log-return)
HORIZON = 1


# Размер скрытого слоя LSTM и количество LSTM-слоев
HIDDEN = None
LAYERS = None

# Пресеты по каждой схеме (если HIDDEN или LAYERS = None), чтобы не править два значения при каждой смене SCHEME
SCHEME_PRESETS = {
    "vanilla": {"hidden": 64,  "layers": 1}, # один слой LSTM + Linear
    "stacked": {"hidden": 128, "layers": 2}, # LSTM из LAYERS слоев + Dropout + Linear
    "attention": {"hidden": 64,  "layers": 1}, # LSTM + self attention поверх hidden states + Linear
    "attn_seq2seq": {"hidden": 128, "layers": 1}, # encoder-decoder LSTM с Bahdanau attention (для horizon=20)
}

DEVICE = "cuda"
SEED = 42

LR = 1e-3 # Adam learning rate
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOP_PATIENCE = 10

# Дата начала и конца выборки
DATE_FROM: date = date.today() - timedelta(days=365*5) # По умолчанию - последние 5 лет, при FEATURES != "quotes" реальный диапазон сужается до пересечения с доп фичами.
DATE_TO: date = date.today()

API_TOKEN = os.environ.get("APP_API_TOKEN", "some_token")
API_URL = "http://localhost:8000"

# Как разбиваем датасет (Test = 1 - TRAIN_FRAC - VAL_FRAC)
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15

# Журнал экспериментов
CSV_PATH = Path(__file__).parent / "dl_experiments.csv"
CSV_FIELDS = [
    "runid", "secid", "scheme", "features",
    "lookback", "horizon", "hidden", "layers", "dropout", "lr", "batch_size", "epochs_trained",
    "mae_train", "mae_val", "mae_test",
    "rmse_train", "rmse_val", "rmse_test",
    "mase_train", "mase_val", "mase_test",
    "dir_accuracy_train", "dir_accuracy_val", "dir_accuracy_test",
    "duration_sec", "device",
]

def _resolve_hidden_layers() -> tuple[int, int]:
    preset = SCHEME_PRESETS.get(SCHEME, {"hidden": 64, "layers": 1})
    hidden = HIDDEN if HIDDEN is not None else preset["hidden"]
    layers = LAYERS if LAYERS is not None else preset["layers"]
    return hidden, layers


def log_to_csv(row: dict) -> None:
    file_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})



# ---------------------------------------------------------------------------------
# МОДЕЛИ
# ---------------------------------------------------------------------------------

# Все принимают (batch, lookback, n_features) и возвращают (batch, horizon)

class VanillaLSTM(nn.Module):
    """Один LSTM-слой + Linear. Самый простой baseline"""

    def __init__(self, n_features: int, hidden: int, horizon: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))


class StackedLSTM(nn.Module):
    """LSTM из LAYERS слоев с Dropout между ними + Linear на горизонт"""

    def __init__(
        self,
        n_features: int,
        hidden: int,
        horizon: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, # применяет dropout между слоями только если num_layers > 1
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))


class AttentionLSTM(nn.Module):
    """LSTM + self attention пуллинг по всем hidden states + Linear.
    Модель сама взвешивает, какие дни в окне важнее (например, на дни с отчетами или новостями attention дает большие веса).
    """

    def __init__(self, n_features: int, hidden: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (batch, T, hidden)
        scores = self.attn(out)  # (batch, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = (out * weights).sum(dim=1)  # (batch, hidden)
        return self.head(self.dropout(context))


class AttentionSeq2Seq(nn.Module):
    """Seq2Seq LSTM с Bahdanau attention и teacher forcing, для длинного горизонта"""

    def __init__(self, n_features: int, hidden: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.hidden = hidden
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        # decoder получает на вход [y_prev (1), context (hidden)]
        self.decoder = nn.LSTMCell(1 + hidden, hidden)
        # Bahdanau-attention: e_i = v^T tanh(W h_t + U enc_out_i)
        self.attn_W = nn.Linear(hidden, hidden)
        self.attn_U = nn.Linear(hidden, hidden)
        self.attn_v = nn.Linear(hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        y_teacher: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        enc_out, (h_n, c_n) = self.encoder(x)  # (batch, T, hidden)
        # Стартовое состояние decoder - последнее состояние encoder
        h_t = h_n[-1]
        c_t = c_n[-1]
        # Стартовый y_prev - ноль (нейтральный log_return)
        y_prev = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(self.horizon):
            # Attention: для текущего скрытого состояния decoder считается context-вектор как взвешенная сумма encoder hidden states
            ht_proj = self.attn_W(h_t).unsqueeze(1)  # (batch, 1, hidden)
            enc_proj = self.attn_U(enc_out)  # (batch, T, hidden)
            scores = self.attn_v(torch.tanh(ht_proj + enc_proj))  # (batch, T, 1)
            weights = torch.softmax(scores, dim=1)
            context = (enc_out * weights).sum(dim=1)  # (batch, hidden)

            dec_in = torch.cat([y_prev, context], dim=1)  # (batch, 1 + hidden)
            h_t, c_t = self.decoder(dec_in, (h_t, c_t))
            y_t = self.head(self.dropout(h_t))  # (batch, 1)
            outputs.append(y_t)

            # Teacher forcing на train, авторегрессия на eval/inference
            if y_teacher is not None and self.training:
                y_prev = y_teacher[:, t : t + 1]
            else:
                y_prev = y_t

        return torch.cat(outputs, dim=1)  # (batch, horizon)


def build_model(scheme: str, n_features: int) -> nn.Module:
    hidden, layers = _resolve_hidden_layers()
    if scheme == "vanilla":
        return VanillaLSTM(n_features, hidden, HORIZON, dropout=DROPOUT)
    if scheme == "stacked":
        return StackedLSTM(n_features, hidden, HORIZON, num_layers=layers, dropout=DROPOUT)
    if scheme == "attention":
        return AttentionLSTM(n_features, hidden, HORIZON, dropout=DROPOUT)
    if scheme == "attn_seq2seq":
        if HORIZON == 1:
            print("❗️ Внимание: attn_seq2seq на horizon=1 избыточен, но запускается.")
        return AttentionSeq2Seq(n_features, hidden, HORIZON, dropout=DROPOUT)
    raise ValueError(f"❌ Неизвестная схема: {scheme!r}")



# ---------------------------------------------------------------------------------
# HTTP-клиент: тонкие обертки над GET /data/*
# ---------------------------------------------------------------------------------Z

def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_TOKEN}"}


def fetch_quotes(secid: str, date_from: date, date_to: date) -> pd.DataFrame:
    """Дневные котировки одного тикера за диапазон дат"""
    params = [
        ("secid", secid),
        ("date_from", date_from.isoformat()),
        ("date_to", date_to.isoformat()),
    ]
    resp = httpx.get(
        f"{API_URL}/data/quotes",
        params=params,
        headers=_headers(),
        timeout=60.0,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        return df
    df["tradedate"] = pd.to_datetime(df["tradedate"]).dt.date
    # Перевод в float, т.к. decimal колонки после JSON приходят как строки
    for col in ["open", "high", "low", "close", "legalcloseprice", "waprice", "value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("tradedate").reset_index(drop=True)


def fetch_fundamentals(secid: str) -> pd.DataFrame:
    """Фундаменталка по тикеру, отсортированная по published_at"""
    resp = httpx.get(
        f"{API_URL}/data/fundamentals",
        params=[("secid", secid)],
        headers=_headers(),
        timeout=60.0,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        return df
    df["published_at"] = pd.to_datetime(df["published_at"]).dt.date
    # Колонки с метриками - в float
    metric_cols = [
        "net_profit", "assets", "equity", "dividend_payout", "dividend_per_share", "dividend_yield", "revenue",
        "ebitda", "ebitda_margin", "operating_profit", "fcf", "capex", "cogs", "net_debt", "net_debt_to_ebitda",
        "interest_income", "fee_income", "securities_income", "loan_portfolio", "loan_loss_provision", "clients_count",
        "personnel_expenses", "operating_expenses", "interest_expenses", "pe_ratio", "pb_ratio", "roe", "roa",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("published_at").reset_index(drop=True)


def fetch_macro(date_from: date, date_to: date) -> pd.DataFrame:
    """Дневные макро - курс USD/RUB и ключевая ставка за диапазон дат"""
    params = [
        ("date_from", date_from.isoformat()),
        ("date_to", date_to.isoformat()),
    ]
    resp = httpx.get(
        f"{API_URL}/data/macro",
        params=params,
        headers=_headers(),
        timeout=60.0,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["usd_rub", "key_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fetch_sentiment(secid: str, date_from: date, date_to: date) -> pd.DataFrame:
    """Дневной агрегат сантиментов по тикеру за диапазон дат"""
    params = [
        ("secid", secid),
        ("date_from", date_from.isoformat()),
        ("date_to", date_to.isoformat()),
    ]
    resp = httpx.get(
        f"{API_URL}/data/sentiment",
        params=params,
        headers=_headers(),
        timeout=60.0,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df.sort_values("trade_date").reset_index(drop=True)



# ---------------------------------------------------------------------------------
# Технические индикаторы
# ---------------------------------------------------------------------------------

# Считаются на сыром ряду до split, где rolling только по предыдущим дням

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["sma_5"] = out["close"].rolling(5).mean()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["rsi_14"] = _rsi(out["close"], 14)
    out["atr_14"] = _atr(out["high"], out["low"], out["close"], 14)
    # log от объема, чтобы избежать огромных хвостов
    out["log_volume"] = np.log1p(out["volume"].fillna(0.0))
    return out



# ---------------------------------------------------------------------------------
# Сборка датафрейма фичей под выбранную стадию
# ---------------------------------------------------------------------------------


# Названия колонок, которые попадают в X (без таргета)
QUOTE_FEATURES = [
    "open", "high", "low", "close", "value", "numtrades", "waprice", "log_return", "sma_5", "sma_20", "rsi_14", "atr_14", "log_volume",
]
MACRO_FEATURES = [
    "macro_usd_rub", "macro_key_rate", "macro_usd_rub_log_return", "macro_key_rate_diff",
]
FUND_FEATURES = [
    "fund_net_profit", "fund_assets", "fund_equity", "fund_pe_ratio", "fund_pb_ratio", "fund_roe",
]
SENT_FEATURES = [
    "sent_mean_score", "sent_news_count", "sent_pos_mean", "sent_neg_mean", "sent_market_share",
]
# Склеивание в группы
FEATURE_COLUMNS = {
    "quotes": QUOTE_FEATURES,
    "quotes_macro": QUOTE_FEATURES + MACRO_FEATURES,
    "quotes_macro_fund": QUOTE_FEATURES + MACRO_FEATURES + FUND_FEATURES,
    "quotes_macro_fund_sent": QUOTE_FEATURES + MACRO_FEATURES + FUND_FEATURES + SENT_FEATURES,
}


def _attach_fundamentals(df_daily: pd.DataFrame, df_fund: pd.DataFrame) -> pd.DataFrame:
    """Добавление фундаментала в дневной ряд без заглядывания в будущее"""
    fund_cols = ["fund_net_profit", "fund_assets", "fund_equity", "fund_pe_ratio", "fund_pb_ratio", "fund_roe"]
    if df_fund.empty:
        for col in fund_cols:
            df_daily[col] = 0.0
        return df_daily

    # Готовится узкий фрейм фундаментала с переименованными колонками
    fund = df_fund[[
        "published_at", "net_profit", "assets", "equity", "pe_ratio", "pb_ratio", "roe",
    ]].rename(columns={
        "net_profit": "fund_net_profit",
        "assets": "fund_assets",
        "equity": "fund_equity",
        "pe_ratio": "fund_pe_ratio",
        "pb_ratio": "fund_pb_ratio",
        "roe": "fund_roe",
    })
    df_daily = df_daily.copy()
    # merge_asof требует datetime на ключе сортировки
    df_daily["__td"] = pd.to_datetime(df_daily["tradedate"])
    fund = fund.sort_values("published_at")
    fund["__pa"] = pd.to_datetime(fund["published_at"])
    merged = pd.merge_asof(
        df_daily.sort_values("__td"),
        fund.sort_values("__pa"),
        left_on="__td",
        right_on="__pa",
        direction="backward", # Важно!
        allow_exact_matches=True,
    )
    merged = merged.drop(columns=["__td", "__pa", "published_at"], errors="ignore")
    # Дни до первой публикации и пропуски отдельных метрик у банков -> 0, чтобы dropna не выкосил
    for col in fund_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
        else:
            merged[col] = 0.0
    return merged


def _attach_macro(df_daily: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """Добавление дневных макро по торговой дате.
        macro_usd_rub_log_return - день-ко-дню log-изменение курса (стационарная версия для LSTM)
        macro_key_rate_diff - изменение ставки, почти всегда 0 со спайками в дни решений
    """
    macro_cols = ["macro_usd_rub", "macro_key_rate", "macro_usd_rub_log_return", "macro_key_rate_diff"]
    if df_macro.empty:
        for col in macro_cols:
            df_daily[col] = 0.0
        return df_daily

    macro = df_macro.rename(columns={
        "date": "macro_date",
        "usd_rub": "macro_usd_rub",
        "key_rate": "macro_key_rate",
    })
    merged = df_daily.merge(macro, left_on="tradedate", right_on="macro_date", how="left")
    merged = merged.drop(columns=["macro_date"], errors="ignore")
    # .ffill().bfill() страховка, чтобы не сломать тензор (хотя по значениям из БД пропусков быть не должно)
    merged["macro_usd_rub"] = merged["macro_usd_rub"].ffill().bfill()
    merged["macro_key_rate"] = merged["macro_key_rate"].ffill().bfill()
    merged["macro_usd_rub_log_return"] = np.log(merged["macro_usd_rub"] / merged["macro_usd_rub"].shift(1))
    merged["macro_key_rate_diff"] = merged["macro_key_rate"].diff()
    return merged


def _attach_sentiment(df_daily: pd.DataFrame, df_sent: pd.DataFrame) -> pd.DataFrame:
    """Подмешивает дневной агрегат сантиментов. Дни без новостей - нули"""
    sent_cols = {
        "mean_score": "sent_mean_score",
        "news_count": "sent_news_count",
        "pos_mean": "sent_pos_mean",
        "neg_mean": "sent_neg_mean",
        "market_news_share": "sent_market_share",
    }
    if df_sent.empty:
        for new_col in sent_cols.values():
            df_daily[new_col] = 0.0
        return df_daily

    sent = df_sent[["trade_date"] + list(sent_cols.keys())].rename(columns=sent_cols)
    merged = df_daily.merge(sent, left_on="tradedate", right_on="trade_date", how="left")
    merged = merged.drop(columns=["trade_date"], errors="ignore")
    # Дни без новостей -> 0
    for col in sent_cols.values():
        merged[col] = merged[col].fillna(0.0)
    return merged


def build_dataset(secid: str) -> tuple[pd.DataFrame, list[str]]:
    """Готовит датафрейм фичей по одной бумаге под выбранную стадию"""
    print(f"📥 Загрузка котировок {secid} за {DATE_FROM} - {DATE_TO} ...")
    df = fetch_quotes(secid, DATE_FROM, DATE_TO)
    if df.empty:
        raise RuntimeError(f"❌ Нет котировок по {secid}")
    print(f"  Строк котировок: {len(df)}, диапазон: {df['tradedate'].min()} - {df['tradedate'].max()}")

    df = add_technicals(df)

    if FEATURES in ("quotes_macro", "quotes_macro_fund", "quotes_macro_fund_sent"):
        print(f"📥 Загрузка макро за {DATE_FROM} - {DATE_TO} ...")
        df_macro = fetch_macro(DATE_FROM, DATE_TO)
        if not df_macro.empty:
            print(f"  Дней макро: {len(df_macro)}, диапазон: {df_macro['date'].min()} - {df_macro['date'].max()}")
        else:
            print(f"  ❗️ макро пуст - колонки будут нулями")
        df = _attach_macro(df, df_macro)

    if FEATURES in ("quotes_macro_fund", "quotes_macro_fund_sent"):
        print(f"📥 Загрузка фундаменталок {secid} ...")
        df_fund = fetch_fundamentals(secid)
        if not df_fund.empty:
            print(f"  Отчетов: {len(df_fund)}, диапазон published_at: {df_fund['published_at'].min()} - {df_fund['published_at'].max()}")
        else:
            print(f"  ❗️ фундаментал по {secid} пуст - колонки будут NaN")
        df = _attach_fundamentals(df, df_fund)

    if FEATURES == "quotes_macro_fund_sent":
        print(f"📥 Загрузка sentiment {secid} ...")
        df_sent = fetch_sentiment(secid, DATE_FROM, DATE_TO)
        if not df_sent.empty:
            print(f"  Дней с новостями: {len(df_sent)}, диапазон: {df_sent['trade_date'].min()} - {df_sent['trade_date'].max()}")
        else:
            print(f"  ❗️ sentiment по {secid} пуст - агрегаты будут нулями")
        df = _attach_sentiment(df, df_sent)

    # Таргет: log_return на t+1 (или вектор log_return на следующие HORIZON шагов)
    if HORIZON == 1:
        df["y"] = df["log_return"].shift(-1)
    else:
        # Вектор будущих log_return склеивается в HORIZON отдельных колонок (для direct multi-output и для loss)
        for k in range(1, HORIZON + 1):
            df[f"y_{k}"] = df["log_return"].shift(-k)

    feature_cols = FEATURE_COLUMNS[FEATURES]
    needed = feature_cols + (["y"] if HORIZON == 1 else [f"y_{k}" for k in range(1, HORIZON + 1)])
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    after = len(df)
    print(f"  отброшено неполных строк: {before - after}, осталось {after}")

    return df, feature_cols



# ---------------------------------------------------------------------------------
# Нарезка rolling окон и split
# ---------------------------------------------------------------------------------


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Строит rolling-окна X, y и индекс анкоров"""
    x_feats = df[feature_cols].to_numpy(dtype=np.float32)
    if horizon == 1:
        y_arr = df["y"].to_numpy(dtype=np.float32)
    else:
        y_cols = [f"y_{k}" for k in range(1, horizon + 1)]
        y_arr = df[y_cols].to_numpy(dtype=np.float32)

    anchor_dates = df["tradedate"].to_numpy() # Анкорная дата - это последняя дата в окне (дата, после которой прогнозируется будущее)

    n = len(df)
    last = n - lookback + 1
    if last <= 0:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))

    x_out = np.empty((last, lookback, x_feats.shape[1]), dtype=np.float32)
    y_out = np.empty((last, horizon) if horizon > 1 else (last,), dtype=np.float32)
    anchors = np.empty(last, dtype=object)

    for i in range(last):
        x_out[i] = x_feats[i : i + lookback]
        idx = i + lookback - 1
        y_out[i] = y_arr[idx]
        anchors[i] = anchor_dates[idx]

    return x_out, y_out, anchors


def time_split(
    anchors: np.ndarray,
    train_frac: float,
    val_frac: float,
    embargo: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает три булевых маски (train, val, test) по позициям окон"""
    n = len(anchors)
    if n == 0:
        return np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, bool)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    mask_train = np.zeros(n, bool)
    mask_val = np.zeros(n, bool)
    mask_test = np.zeros(n, bool)
    mask_train[:train_end] = True
    mask_val[train_end + embargo:val_end] = True
    mask_test[val_end + embargo:] = True
    return mask_train, mask_val, mask_test


def fit_scaler(x_train: np.ndarray) -> StandardScaler:
    """Фитит StandardScaler по train (после reshape в 2D) для дальнейшего использования в val и test"""
    flat = x_train.reshape(-1, x_train.shape[-1])
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def apply_scaler(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Применяет StandardScaler к 3D-тензору (batch, lookback, n_features)"""
    flat = x.reshape(-1, x.shape[-1])
    scaled = scaler.transform(flat)
    return scaled.reshape(x.shape).astype(np.float32)



# ---------------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------------

# Все считаются на тензорах (batch, horizon) или (batch,)

def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """Приводит и target и pred к форме (batch, horizon)"""
    if t.dim() == 1:
        return t.unsqueeze(1)
    return t


def naive_scale_from_train(y_train: np.ndarray) -> float:
    """Знаменатель MASE"""
    series = y_train if y_train.ndim == 1 else y_train[:, 0]
    if len(series) < 2:
        return float("nan")
    diffs = np.abs(series[1:] - series[:-1])
    return float(diffs.mean())


def calc_metrics(pred: torch.Tensor, true: torch.Tensor, naive_scale: float) -> dict[str, float]:
    """Считает MAE, RMSE, MASE и directional accuracy"""
    pred = _to_2d(pred.detach().cpu())
    true = _to_2d(true.detach().cpu())

    abs_err = (pred - true).abs()
    mae = abs_err.mean().item()
    rmse = torch.sqrt(((pred - true) ** 2).mean()).item()

    mase = mae / naive_scale if naive_scale and naive_scale > 0 else float("nan")

    # Знак: > 0 как 1, иначе 0. Совпадения усредняются.
    sign_pred = (pred > 0).float()
    sign_true = (true > 0).float()
    dir_acc = (sign_pred == sign_true).float().mean().item() # доля совпадений знака предсказания и факта (среднее по batch и по шагам горизонта)

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mase": round(mase, 6),
        "dir_acc": round(dir_acc, 6),
    }



# ---------------------------------------------------------------------------------
# Тренировка одной модели на одном тикете
# ---------------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_secid(secid: str, runid: int) -> dict[str, str | int | float]:
    """Обучает модель на указанном тикете, возвращает словарь"""
    _set_seed(SEED)
    started_at = time.time()

    #  Собираем датасет
    df, feature_cols = build_dataset(secid)
    if len(df) < LOOKBACK + HORIZON + 50:
        raise RuntimeError(f"❌ Слишком мало данных по {secid}: {len(df)} строк, нужно минимум {LOOKBACK + HORIZON + 50}")

    # Rolling окна
    x_all, y_all, anchors = make_windows(df, feature_cols, LOOKBACK, HORIZON)
    if len(x_all) == 0:
        raise RuntimeError(f"❌ Не удалось построить окна для {secid}")
    print(f"  окон всего: {len(x_all)}, форма X: {x_all.shape}")

    # Split с embargo, чтобы окна соседних сплитов не пересекались по дням
    embargo = LOOKBACK + HORIZON - 1
    mask_train, mask_val, mask_test = time_split(anchors, TRAIN_FRAC, VAL_FRAC, embargo=embargo)
    x_train, y_train = x_all[mask_train], y_all[mask_train]
    x_val, y_val = x_all[mask_val], y_all[mask_val]
    x_test, y_test = x_all[mask_test], y_all[mask_test]
    print(f"  split (train/val/test): {len(x_train)} / {len(x_val)} / {len(x_test)}, embargo={embargo} окон")
    if len(x_val) == 0 or len(x_test) == 0:
        raise RuntimeError(f"❌ embargo={embargo} съел val или test. Увеличь диапазон дат, либо уменьши LOOKBACK / HORIZON")

    # Знаменатель MASE считаем один раз по y_train (naive).
    naive_scale = naive_scale_from_train(y_train)
    print(f"  naive MASE (mean |y_t - y_t-1| на train): {naive_scale:.6f}")

    # StandardScaler по train, потом применяется ко всем
    scaler = fit_scaler(x_train)
    x_train = apply_scaler(x_train, scaler)
    x_val = apply_scaler(x_val, scaler)
    x_test = apply_scaler(x_test, scaler)

    # Тензоры и DataLoader
    device = torch.device(DEVICE)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_x_t = torch.from_numpy(x_val).to(device)
    val_y_t = torch.from_numpy(y_val).to(device)
    test_x_t = torch.from_numpy(x_test).to(device)
    test_y_t = torch.from_numpy(y_test).to(device)

    # Модель и оптимизатор
    model = build_model(SCHEME, x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    patience = 0
    epochs_trained = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            if SCHEME == "attn_seq2seq" and HORIZON > 1:
                pred = model(x_batch, y_teacher=y_batch)
            else:
                pred = model(x_batch)
            # Приводим к одинаковой форме перед loss
            if pred.shape != y_batch.shape:
                pred = pred.squeeze(-1) if pred.shape[-1] == 1 else pred
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Val
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_t)
            if val_pred.shape != val_y_t.shape and val_pred.shape[-1] == 1:
                val_pred = val_pred.squeeze(-1)
            val_metrics = calc_metrics(val_pred, val_y_t, naive_scale)

        epochs_trained = epoch
        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(f"эпоха {epoch:3d}: train_loss={mean_train_loss:.5f}, val_mae={val_metrics['mae']:.5f}, val_mase={val_metrics['mase']:.3f}, val_dir_acc={val_metrics['dir_acc']:.3f}")

        # Early stop
        if val_metrics["mae"] < best_val_mae - 1e-6:
            best_val_mae = val_metrics["mae"]
            patience = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"  Early stop: лучший val_mae={best_val_mae:.5f}")
                break

    # Восстановление лучших весов и финальные метрики
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_x_t = torch.from_numpy(x_train).to(device)
        train_y_t = torch.from_numpy(y_train).to(device)
        train_pred = model(train_x_t)
        if train_pred.shape != train_y_t.shape and train_pred.shape[-1] == 1:
            train_pred = train_pred.squeeze(-1)
        train_m = calc_metrics(train_pred, train_y_t, naive_scale)

        val_pred = model(val_x_t)
        if val_pred.shape != val_y_t.shape and val_pred.shape[-1] == 1:
            val_pred = val_pred.squeeze(-1)
        val_m = calc_metrics(val_pred, val_y_t, naive_scale)

        test_pred = model(test_x_t)
        if test_pred.shape != test_y_t.shape and test_pred.shape[-1] == 1:
            test_pred = test_pred.squeeze(-1)
        test_m = calc_metrics(test_pred, test_y_t, naive_scale)

    duration = time.time() - started_at
    print(f"  {secid}: train_mae={train_m['mae']:.5f}, val_mae={val_m['mae']:.5f}  test_mae={test_m['mae']:.5f}, за {duration:.1f} сек")

    hidden, layers = _resolve_hidden_layers()
    return {
        "runid": runid,
        "secid": secid,
        "scheme": SCHEME,
        "features": FEATURES,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "hidden": hidden,
        "layers": layers,
        "dropout": DROPOUT,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs_trained": epochs_trained,
        "mae_train": train_m["mae"],
        "mae_val": val_m["mae"],
        "mae_test": test_m["mae"],
        "rmse_train": train_m["rmse"],
        "rmse_val": val_m["rmse"],
        "rmse_test": test_m["rmse"],
        "mase_train": train_m["mase"],
        "mase_val": val_m["mase"],
        "mase_test": test_m["mase"],
        "dir_accuracy_train": train_m["dir_acc"],
        "dir_accuracy_val": val_m["dir_acc"],
        "dir_accuracy_test": test_m["dir_acc"],
        "duration_sec": math.ceil(duration),
        "device": DEVICE,
    }



# ---------------------------------------------------------------------------------
# ГЛАВНЫЙ ЦИКЛ
# ---------------------------------------------------------------------------------


def _check_device() -> None:
    """Печатает информацию об устройстве и фейлит запуск, если cuda не работает"""
    print("PyTorch:", torch.__version__)
    print("CUDA доступна:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version (torch):", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))

    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("❌ DEVICE='cuda', но torch.cuda.is_available() = False. Надо чинить установку torch.")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Тикер MOEX, например SBER")
    parser.add_argument(
        "--features",
        required=True,
        choices=list(FEATURE_COLUMNS.keys()),
        help="Набор фичей (quotes / quotes_macro / quotes_macro_fund / quotes_macro_fund_sent)",
    )
    parser.add_argument(
        "--scheme",
        required=True,
        choices=list(SCHEME_PRESETS.keys()),
        help="Архитектура LSTM (vanilla / stacked / attention / attn_seq2seq)",
    )
    parser.add_argument(
        "--horizon",
        required=True,
        type=int,
        choices=[1, 20],
        help="Горизонт прогноза в торговых днях",
    )
    parser.add_argument("--lookback", required=True, type=int, help="Длина rolling window")
    return parser.parse_args()


def main() -> None:
    global TICKER, FEATURES, SCHEME, HORIZON, LOOKBACK
    args = parse_args()
    TICKER = args.ticker
    FEATURES = args.features
    SCHEME = args.scheme
    HORIZON = args.horizon
    LOOKBACK = args.lookback

    runid = int(time.time() * 1000) # целочисленный unix-timestamp в миллисекундах
    print("-" * 60)
    print(f"ЗАПУСК ЭКСПЕРИМЕНА {runid}!")
    print(f"  TICKER={TICKER}, SCHEME={SCHEME}, FEATURES={FEATURES}, LOOKBACK={LOOKBACK}, HORIZON={HORIZON}")
    print(f"  DEVICE={DEVICE}")
    print("-" * 60)

    _check_device()

    print()
    print(f"--- {TICKER} ---")
    ok = True
    try:
        row = train_one_secid(TICKER, runid)
        log_to_csv(row)
        print(f"  строка дописана в {CSV_PATH.name}")
    except Exception as exc:
        print(f"  ❌ {TICKER}: {type(exc).__name__}: {exc}")
        ok = False

    print()
    print("-" * 60)
    if ok:
        print(f"✅ {TICKER} обработан")
    else:
        print(f"❗️ {TICKER} завершился с ошибкой")
    print(f"Журнал: {CSV_PATH}")
    print("-" * 60)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
