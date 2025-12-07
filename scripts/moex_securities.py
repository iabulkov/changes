'''
SECID          {"type": "string", "bytes": 36, "max_size": 0},
BOARDID        {"type": "string", "bytes": 12, "max_size": 0},
SHORTNAME      {"type": "string", "bytes": 30, "max_size": 0},
PREVPRICE       {"type": "double"},
LOTSIZE         {"type": "int32"},
FACEVALUE       {"type": "double"},
STATUS          {"type": "string", "bytes": 3, "max_size": 0},
BOARDNAME       {"type": "string", "bytes": 381, "max_size": 0},
DECIMALS        {"type": "int32"},
SECNAME         {"type": "string", "bytes": 90, "max_size": 0},
REMARKS         {"type": "string", "bytes": 24, "max_size": 0},
MARKETCODE      {"type": "string", "bytes": 12, "max_size": 0},
INSTRID         {"type": "string", "bytes": 12, "max_size": 0},
SECTORID        {"type": "string", "bytes": 12, "max_size": 0},
MINSTEP         {"type": "double"},
PREVWAPRICE     {"type": "double"},
FACEUNIT        {"type": "string", "bytes": 12, "max_size": 0},
PREVDATE        {"type": "date", "bytes": 10, "max_size": 0},
ISSUESIZE       {"type": "int64"},
ISIN            {"type": "string", "bytes": 36, "max_size": 0},
LATNAME         {"type": "string", "bytes": 90, "max_size": 0},
REGNUMBER       {"type": "string", "bytes": 90, "max_size": 0},
PREVLEGALCLOSEPRICE     {"type": "double"},
CURRENCYID      {"type": "string", "bytes": 12, "max_size": 0},
SECTYPE         {"type": "string", "bytes": 3, "max_size": 0},
LISTLEVEL       {"type": "int32"},
SETTLEDATE      {"type": "date", "bytes": 10, "max_size": 0}
'''

import requests
import pandas as pd


def moex_get_securities(host='https://iss.moex.com'):
    try:
        print("📝 Получаю список акций с MOEX...")
        url = host + '/iss/engines/stock/markets/shares/securities.json'
        response = requests.get(url)
        response.raise_for_status()
        print("✅ Ответ успешно получен!")
        return response.json()
    except requests.exceptions.ConnectionError as e:
        print("❌ Ошибка соединения:", e)
    except requests.exceptions.HTTPError as e:
        print("❌ HTTP ошибка:", e)
    except requests.exceptions.RequestException as e:
        print("❌ Ошибка при запросе:", e)


def moex_save_securities(data, path='data/moex/moex_securities.csv'):
    try:
        print("🔎 Проверяю и сохраняю данные...")
        if 'securities' in data and data['securities']['data']:
            df = pd.DataFrame(data["securities"]["data"], columns=data["securities"]["columns"])
            df.to_csv(path, index=True, encoding="utf-8-sig")
            print(f"✅ Данные сохранены в {path}")
            return df
        else:
            raise TypeError('securities')
    except TypeError as e:
        print("❌ Ошибка дынных:", e)


data = moex_get_securities()
if data is not None:
    df = moex_save_securities(data)
    if df is not None:
        print("\n", df.head())

