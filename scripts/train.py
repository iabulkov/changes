import os
import argparse
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

# python scripts/train.py --data-dir data/moex --secid SBER --out artifacts/sarimax_sber.pkl

# Тикеты
SECIDS = ['GAZP', 'GMKN', 'LKOH', 'NVTK', 'PIKK', 'ROSN', 'SBER', 'T', 'VTBR', 'YDEX']


def merge_csv_files(data_dir: str, output_file: str | None = None) -> pd.DataFrame:
    """
    Объединяет CSV файлы для заданных тикеров в один DataFrame
    Ожидает файлы вида: data_dir/moex_<SECID>.csv
    """
    dfs = []
    missing_files = []

    for secid in SECIDS:
        filename = os.path.join(data_dir, f"moex_{secid}.csv")
        if os.path.exists(filename):
            try:
                print(f"Читаю файл: {filename}")
                df = pd.read_csv(filename)
                dfs.append(fill_missing_dates(df))
            except Exception as e:
                print(f"Ошибка при чтении файла {filename}: {e}")
        else:
            print(f"Файл не найден: {filename}")
            missing_files.append(filename)

    if not dfs:
        raise ValueError("Ничего не удалось прочитать! Проверь data_dir и имена файлов")

    result_df = pd.concat(dfs, ignore_index=True)
    print(f"- Размер DataFrame: {result_df.shape}")

    if missing_files:
        print(f"- Не найдено файлов: {len(missing_files)}")
        for f in missing_files:
            print("  -", f)

    if output_file:
        out_dir = os.path.dirname(output_file)
        # на случай если output_file = "file.csv" без папки
        if out_dir: 
            os.makedirs(out_dir, exist_ok=True)
        result_df.to_csv(output_file, index=False)
        print(f"Объединённый датасет сохранён в: {output_file}")
    
    print("=" * 50)
    return result_df


def fill_missing_dates(df, column="TRADEDATE") -> pd.DataFrame:
    """
    Заполняет пропуски и добавляет флаг выходного дня.
    Forward fill (заполнение последним значением)
    """
    # Преобразуем колонку с датами в datetime
    df = df.copy()
    df[column] = pd.to_datetime(df[column])

    # Сортируем по дате
    df = df.sort_values(by=column).reset_index(drop=True)

    # Создаем полный диапазон дат от минимальной до максимальной
    min_date = df[column].min()
    max_date = df[column].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    # Создаем DataFrame с полным диапазоном дат
    full_df = pd.DataFrame({column: full_date_range})

    # Объединяем с исходными данными
    df_filled = full_df.merge(df, on=column, how='left')

    # Добавляем признак IS_WORK_DAY (1 для существующих строк, 0 для вставленных) и проверяем, есть ли хотя бы одна непустая колонка кроме даты
    other_columns = [col for col in df.columns if col != column]
    df_filled['IS_WORK_DAY'] = df_filled[other_columns].notna().any(axis=1).astype(int)

    # Заполняем пропуски forward fill для всех колонок кроме даты
    df_filled[other_columns] = df_filled[other_columns].ffill()

    # Сортируем по дате
    df_filled = df_filled.sort_values(by=column).reset_index(drop=True)

    return df_filled

def build_series(df: pd.DataFrame, secid: str) -> pd.DataFrame:
    """
    Фильтрует по тикеру и строит целевую переменную LOG_RETURN.
    """
    if "SECID" not in df.columns:
        raise ValueError("В данных нет колонки SECID")

    df_series = df[df["SECID"] == secid].copy()
    if df_series.empty:
        raise ValueError(f"Нет строк для тикера {secid}")

    # ensure types
    df_series["TRADEDATE"] = pd.to_datetime(df_series["TRADEDATE"])
    df_series = df_series.sort_values("TRADEDATE").reset_index(drop=True)

    # LOG_RETURN — исправлено: считаем на df_series (в ноутбуке была ошибка)
    df_series["LOG_RETURN"] = np.log(
        df_series["LEGALCLOSEPRICE"] / df_series["LEGALCLOSEPRICE"].shift(1)
    )

    df_series = df_series[["TRADEDATE", "LEGALCLOSEPRICE", "LOG_RETURN", "IS_WORK_DAY"]]
    df_series = df_series.dropna(subset=["LOG_RETURN"]).reset_index(drop=True)
    return df_series


def train_sarimax(log_returns: pd.Series, order=(1, 0, 1)):
    model = SARIMAX(
        log_returns,
        order=order,
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    model_fit = model.fit(disp=False)
    return model_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/moex", help="Папка с CSV: moex_<SECID>.csv")
    parser.add_argument("--secid", type=str, default="SBER", help="Тикер, например SBER")
    parser.add_argument("--out", type=str, default="artifacts/sarimax_sber.pkl", help="Куда сохранить модель")
    parser.add_argument("--order", type=str, default="1,0,1", help="ARIMA order p,d,q например 1,0,1")
    parser.add_argument("--save-merged", type=str, default="", help="(опц.) сохранить объединённый CSV")
    args = parser.parse_args()

    p, d, q = (int(x.strip()) for x in args.order.split(","))

    merged_out = args.save_merged.strip() or None
    df = merge_csv_files(args.data_dir, output_file=merged_out)

    df_series = build_series(df, args.secid)
    print(f"Ряд для {args.secid}: {df_series.shape}, период {df_series['TRADEDATE'].min()} — {df_series['TRADEDATE'].max()}")

    model_fit = train_sarimax(df_series["LOG_RETURN"], order=(p, d, q))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model_fit.save(args.out)
    print(f"✅ Модель сохранена: {args.out}")


if __name__ == "__main__":
    main()