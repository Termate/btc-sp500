import os
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


def update_data_and_models():
    # -------------------------------------------------------------------------
    # ШАГ 1. Сбор и предобработка данных Bitcoin
    # -------------------------------------------------------------------------

    os.makedirs("output", exist_ok=True)

    print("\n=== ШАГ 1: Загрузка и агрегация данных Bitcoin ===")
    bitcoin_data_path = "data/btcusd_1-min_data.csv"
    if not os.path.isfile(bitcoin_data_path):
        print(f"ОШИБКА: Не найден файл {bitcoin_data_path}")
        return

    btc_data = pd.read_csv(bitcoin_data_path)

    # Проверка: может быть Timestamp в миллисекундах
    max_ts = btc_data["Timestamp"].max()
    if max_ts > 1e10:
        print("Похоже, что Timestamp в миллисекундах. Делим на 1000 для секунд.")
        btc_data["Timestamp"] = btc_data["Timestamp"] // 1000

    # Преобразую в datetime
    btc_data["Date"] = pd.to_datetime(btc_data["Timestamp"], unit="s", errors="coerce")
    btc_data.dropna(subset=["Date", "Close"], inplace=True)

    # Агрегирую в дневные
    btc_daily = (
        btc_data
        .set_index("Date")
        .resample("D")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
        .dropna()
        .reset_index()
    )
    if btc_daily.empty:
        print("ОШИБКА: После агрегации нет дневных строк.")
        return

    # Переименовываю Close -> BTC_Close
    btc_daily.rename(columns={"Close": "BTC_Close"}, inplace=True)
    print(f"BTC дневных строк: {len(btc_daily)}")

    # -------------------------------------------------------------------------
    # ШАГ 2. Скачивание S&P 500 и объединение
    # -------------------------------------------------------------------------
    start_date = btc_daily["Date"].min()
    end_date = btc_daily["Date"].max()
    print(f"\n=== Скачиваем S&P 500 ({start_date.date()} — {end_date.date()}) ===")
    sp500_df = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    sp500_df.reset_index(inplace=True)

    # Сплющиваю, если мультииндекс
    if isinstance(sp500_df.columns, pd.MultiIndex):
        sp500_df.columns = [
            "_".join([str(x) for x in col if x])
            for col in sp500_df.columns
        ]
    # Переименовываю Close -> SP500_Close
    if "Close" in sp500_df.columns:
        sp500_df.rename(columns={"Close": "SP500_Close"}, inplace=True)
    elif "Close_^GSPC" in sp500_df.columns:
        sp500_df.rename(columns={"Close_^GSPC": "SP500_Close"}, inplace=True)

    if "Date" not in sp500_df.columns or "SP500_Close" not in sp500_df.columns:
        print("ОШИБКА: не нашли Date / SP500_Close в SP500.")
        return

    sp500_df = sp500_df[["Date", "SP500_Close"]].dropna(subset=["SP500_Close"])

    # merdge
    merged_data = pd.merge(btc_daily, sp500_df, on="Date", how="inner")
    if merged_data.empty:
        print("ОШИБКА: После merge нет общих дат.")
        return

    # -------------------------------------------------------------------------
    # ШАГ 3. Модель BTC
    # -------------------------------------------------------------------------
    # (например, лаг=1, чтобы не потерять слишком много)
    btc_model_data = merged_data.copy()
    btc_model_data["BTC_Lag1"] = btc_model_data["BTC_Close"].shift(1)
    btc_model_data.dropna(subset=["BTC_Close", "BTC_Lag1"], inplace=True)

    if len(btc_model_data) < 2:
        print("BTC-модель: слишком мало строк после shift(1).")
        return

    X_btc = btc_model_data[["BTC_Lag1"]]
    y_btc = btc_model_data["BTC_Close"]

    train_btc = int(0.8 * len(X_btc))
    if train_btc == 0:
        print("BTC-модель: нет train или test выборки.")
        return
    X_btc_train, X_btc_test = X_btc[:train_btc], X_btc[train_btc:]
    y_btc_train, y_btc_test = y_btc[:train_btc], y_btc[train_btc:]

    # создётся объект модели
    model_btc = LinearRegression()
    # Вызываем метод fit, передавая признаки и цели тренировочной выборки:
    model_btc.fit(X_btc_train, y_btc_train)
    y_btc_pred = model_btc.predict(X_btc_test)
    print(f"BTC-модель обучена. Всего строк: {len(X_btc)}, train={len(X_btc_train)}, test={len(X_btc_test)}")

    # Запишем предсказания в merged_data
    idx_btc_test = btc_model_data.index[train_btc:]  # индексы строк, которые вошли в тест (это последние 20%).
    merged_data["BTC_Pred"] = None
    merged_data.loc[idx_btc_test, "BTC_Pred"] = y_btc_pred

    # -------------------------------------------------------------------------
    # ШАГ 4. Модель S&P 500
    # -------------------------------------------------------------------------
    sp500_model_data = merged_data.copy()
    sp500_model_data["SP500_Lag1"] = sp500_model_data["SP500_Close"].shift(1)
    sp500_model_data.dropna(subset=["SP500_Close", "SP500_Lag1"], inplace=True)

    if len(sp500_model_data) < 2:
        print("SP500-модель: слишком мало строк после shift(1).")
        return

    X_sp500 = sp500_model_data[["SP500_Lag1"]]
    y_sp500 = sp500_model_data["SP500_Close"]

    train_sp500 = int(0.8 * len(X_sp500))
    if train_sp500 == 0:
        print("SP500-модель: нет train/test.")
        return
    X_sp500_train, X_sp500_test = X_sp500[:train_sp500], X_sp500[train_sp500:]
    y_sp500_train, y_sp500_test = y_sp500[:train_sp500], y_sp500[train_sp500:]

    model_sp500 = LinearRegression()
    model_sp500.fit(X_sp500_train, y_sp500_train)
    y_sp500_pred = model_sp500.predict(X_sp500_test)
    print(f"SP500-модель обучена. Всего строк: {len(X_sp500)}, train={len(X_sp500_train)}, test={len(X_sp500_test)}")

    # Запишем предсказания
    idx_sp500_test = sp500_model_data.index[train_sp500:]
    merged_data["SP500_Pred"] = None
    merged_data.loc[idx_sp500_test, "SP500_Pred"] = y_sp500_pred

    # -------------------------------------------------------------------------
    
  # итог
    merged_data.to_csv("output/merged_bitcoin_sp500_data.csv", index=False)
    print("merged_bitcoin_sp500_data.csv обновлен с BTC_Pred, SP500_Pred.")

    with open("output/btc_model.pkl", "wb") as f:
        pickle.dump(model_btc, f)
    with open("output/sp500_model.pkl", "wb") as f:
        pickle.dump(model_sp500, f)
    print("Сохранены btc_model.pkl, sp500_model.pkl")

    print("\n=== Готово! Две модели обучены, merged_data c BTC_Pred и SP500_Pred. ===")


if __name__ == "__main__":
    update_data_and_models()
