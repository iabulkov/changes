"""
BOARDID	            {"type": "string", "bytes": 12, "max_size": 0}      Идентификатор режима торгов	TQBR	Основной рынок акций (Т+). Может быть TQBR, TQOB, EQNE и т.д.
TRADEDATE	        {"type": "date", "bytes": 10, "max_size": 0}        Дата торгов	2024-11-20	Ключевой временной признак (ось времени для анализа).
SHORTNAME	        {"type": "string", "bytes": 189, "max_size": 0}     Краткое название бумаги	Сбербанк-п	Удобно для отображения и проверки.
SECID	            {"type": "string", "bytes": 36, "max_size": 0}      Код бумаги (тикер)	SBERP	Уникальный идентификатор инструмента.
NUMTRADES	        {"type": "double"}                                  Количество сделок за день	34251	Показывает ликвидность и активность торгов.
VALUE	            {"type": "double"}                                  Общий оборот торгов (₽)	4828573945.43	Сумма всех сделок за день (в рублях).
OPEN	            {"type": "double"}                                  Цена открытия	205.10	Цена первой сделки дня.
LOW	                {"type": "double"}                                  Минимальная цена дня	203.50	Самая низкая цена за день.
HIGH	            {"type": "double"}                                  Максимальная цена дня	206.80	Самая высокая цена за день.
LEGALCLOSEPRICE	    {"type": "double"}                                  Цена закрытия	205.95	Официальная цена закрытия по стандартам MOEX (важнее, чем CLOSE).
WAPRICE	            {"type": "double"}                                  Средневзвешенная цена (Weighted Average Price)	205.47	Средняя цена, взвешенная по объёму сделок — часто используют для анализа.
CLOSE	            {"type": "double"}                                  Цена последней сделки дня	205.90	Последняя зарегистрированная сделка; может отличаться от официального close.
VOLUME	            {"type": "double"}                                  Объём торгов (в штуках)	2_345_000	Общее количество акций, сменивших владельца.
MARKETPRICE2	    {"type": "double"}                                  Рыночная цена по методике №2	205.50	Используется биржей для расчёта индексов; близка к средневзвешенной.
MARKETPRICE3	    {"type": "double"}                                  Рыночная цена по методике №3	205.55	Альтернативный способ расчёта рыночной цены (внутренние нужды MOEX).
ADMITTEDQUOTE	    {"type": "double"}                                  Допущенная цена	205.47	Средняя цена, по которой бумага допускается к торгам на бирже.
MP2VALTRD	        {"type": "double"}                                  Оборот по MARKETPRICE2 (₽)	4828573945.43	Рыночная стоимость объёма торгов, рассчитанная по MARKETPRICE2.
MARKETPRICE3TRADESVALUE	    {"type": "double"}                          Оборот по MARKETPRICE3 (₽)	4828573945.43	То же самое, но по MARKETPRICE3.
ADMITTEDVALUE	    {"type": "double"}                                  Допущенный оборот	4828573945.43	Оборот, который учтён в расчётах по допущенным торгам.
WAVAL	            {"type": "double"}                                  Средневзвешенная стоимость всех сделок	205.47	Иногда дублирует WAPRICE.
TRADINGSESSION	    {"type": "int32"}                                   Номер торговой сессии	1	1 — основная, 2 — вечерняя, 3 — постторговая.
CURRENCYID	        {"type": "string", "bytes": 9, "max_size": 0}       Валюта торгов	RUB	Почти всегда RUB, но может быть USD, EUR.
TRENDCLSPR	        {"type": "double"}                                  Тренд закрытия (изменение к предыдущему дню)	0.35	Показывает, насколько изменилась цена закрытия относительно предыдущей.
TRADE_SESSION_DATE	{"type": "date", "bytes": 10, "max_size": 0}        Дата торговой сессии	2024-11-20	Иногда дублирует TRADEDATE (служебное поле для индексов).
"""

import pandas as pd
import requests
import time
from dateutil import parser


def moex_read_securities(path='data/moex/moex_securities.csv'):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"❌ Файл {path} не найден. Сначала запусти moex_securities.py")


def input_read_dates():
    try:
        print("❓ Введите дату начала сбора и дату окончания через запятую:")
        start_str, end_str = [s.strip() for s in input().strip().split(",")]
        start_date = parser.parse(start_str, dayfirst=True)
        end_date = parser.parse(end_str, dayfirst=True)

        if start_date > end_date:
            start_date, end_date = end_date, start_date

        return start_date.date(), end_date.date()
    except Exception as e:
        print("❌ Ошибка в формате дат.")


def moex_get_data_day(secid, date_start, date_end, board='TQBR', start=0, host='https://iss.moex.com'):
    try:
        print(f"📝 Получаю данные по {secid} (start={start}) с MOEX...")
        url = host + f'/iss/history/engines/stock/markets/shares/boards/{board}/securities/{secid}.json'
        params = {
            "from": date_start,
            "till": date_end,
            "start": start
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'history' not in data:
            raise TypeError('history')

        df = pd.DataFrame(data['history']['data'], columns=data['history']['columns'])
        df.index = df.index + 1
        print("✅ Ответ получен и проверен!")

        return df

    except requests.exceptions.ConnectionError as e:
        print("❌ Ошибка соединения:", e)
    except requests.exceptions.HTTPError as e:
        print("❌ HTTP ошибка:", e)
    except requests.exceptions.RequestException as e:
        print("❌ Ошибка при запросе:", e)
    except TypeError as e:
        print("❌ Ошибка дынных:", e)


def moex_get_data_all(secid, date_start, date_end, board='TQBR', host='https://iss.moex.com'):
    data_all = []
    start = 0
    step = 100

    while True:
        df = moex_get_data_day(secid, date_start, date_end, board, start, host)
        if df is None or df.empty:
            print("❗️ Достигнут конец данных.")
            break
        data_all.append(df)
        start += step
        time.sleep(0.5)

    if data_all:
        return pd.concat(data_all, ignore_index=True)
    else:
        return pd.DataFrame()


def moex_save_data_all(secid, df, path='data/moex/moex_secid.csv'):
    if not df.empty:
        path = path.replace("secid", secid)
        df = df.drop_duplicates().sort_values("TRADEDATE")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ Данные сохранены в {path}")
        return df
    else:
        print(f"❌ Не удалось получить данные (возможно {secid} не торговалась)")



df1 = moex_read_securities()
if df1 is not None:
    print("❓ Введите тикер (код бумаги):")
    secid = input().strip().upper()

    if secid in df1["SECID"].values:
        date_start, date_end = input_read_dates()

        if date_start is not None and date_end is not None:
            df2 = moex_get_data_all(secid, date_start, date_end)

            if df2 is not None:
                moex_save_data_all(secid, df2)
                print("\n", df2.head())
    else:
        print(f"❌ Тикер '{secid}' в списке не найден!")
