import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os

app = dash.Dash(__name__)

def read_merged_data():
    path = "output/merged_bitcoin_sp500_data.csv"
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)

app.layout = html.Div([
    html.H1("Bitcoin & S&P 500 Dashboard"),
    dcc.Graph(id="btc-graph"),
    dcc.Graph(id="sp500-graph"),

    html.Button("Обновить данные", id="update-button", n_clicks=0),
    html.Div(id="status-text"),
])

@app.callback(
    Output("btc-graph", "figure"),
    Output("sp500-graph", "figure"),
    Output("status-text", "children"),
    Input("update-button", "n_clicks")
)
def update_dashboard(n):


    df = read_merged_data()
    if df is None:
        return {}, {}, "Нет данных (merged_bitcoin_sp500_data.csv не найден)."

    # ========== График BTC ==========
    # Проверка есть ли BTC_Pred
    if "BTC_Pred" in df.columns:
        # Две линии: BTC_Close и BTC_Pred
        df_btc = df[["Date","BTC_Close","BTC_Pred"]].copy()
        df_btc.dropna(subset=["BTC_Close","BTC_Pred"], inplace=True)
        if df_btc.empty:
            fig_btc = px.line(title="BTC: нет данных (после dropna).")
        else:
            fig_btc = px.line(df_btc, x="Date", y=["BTC_Close","BTC_Pred"],
                              title="BTC: Real vs Predicted")
    else:
        # только одна линия
        if "BTC_Close" not in df.columns:
            fig_btc = px.line(title="BTC: нет колонки BTC_Close / BTC_Pred.")
        else:
            df_btc = df[["Date","BTC_Close"]].dropna(subset=["BTC_Close"])
            if df_btc.empty:
                fig_btc = px.line(title="BTC: нет данных (после dropna).")
            else:
                fig_btc = px.line(df_btc, x="Date", y="BTC_Close",
                                  title="BTC: Real Only")

    # ========== График S&P 500 ==========
    # Проверка, есть ли SP500_Pred
    if "SP500_Pred" in df.columns:
        # Две линии: SP500_Close, SP500_Pred
        df_sp = df[["Date","SP500_Close","SP500_Pred"]].copy()
        df_sp.dropna(subset=["SP500_Close","SP500_Pred"], inplace=True)
        if df_sp.empty:
            fig_sp500 = px.line(title="S&P 500: нет данных (после dropna).")
        else:
            fig_sp500 = px.line(df_sp, x="Date", y=["SP500_Close","SP500_Pred"],
                                title="S&P 500: Real vs Predicted")
    else:
        # только одна линия
        if "SP500_Close" not in df.columns:
            fig_sp500 = px.line(title="S&P 500: нет колонки SP500_Close / SP500_Pred.")
        else:
            df_sp = df[["Date","SP500_Close"]].dropna(subset=["SP500_Close"])
            if df_sp.empty:
                fig_sp500 = px.line(title="S&P 500: нет данных (после dropna).")
            else:
                fig_sp500 = px.line(df_sp, x="Date", y="SP500_Close",
                                    title="S&P 500: Real Only")

    msg = f"Данные обновлены {n} раз." if n else "Готово!"
    return fig_btc, fig_sp500, msg


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)