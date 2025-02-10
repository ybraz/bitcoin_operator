# Importar as bibliotecas necessárias
import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import time
import yfinance as yf


# Função para baixar dados de OHLCV do par BTC/USDT
# OHLCV = Open, High, Low, Close, Volume
def fetch_all_binance_data(symbol="BTC/USDT", timeframe="1d"):
    exchange = ccxt.binance()
    since = exchange.parse8601("2017-01-01T00:00:00Z")
    all_data = []
    max_limit = 1000

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=max_limit)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        print(f"Baixados {len(all_data)} registros...")
        time.sleep(1)

    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values(
        "timestamp"
    )
    return df


# Coletar os dados de cotação
btc_data = fetch_all_binance_data()

# Incluindo colunas de média móvel (moving average) de 3 dias
btc_data["open_ma3"] = btc_data["open"].rolling(window=3).mean()
btc_data["close_ma3"] = btc_data["close"].rolling(window=3).mean()
btc_data["volume_ma3"] = btc_data["volume"].rolling(window=3).mean()
btc_data["high_ma3"] = btc_data["high"].rolling(window=3).mean()
btc_data["low_ma3"] = btc_data["low"].rolling(window=3).mean()

# Criando colunas de shift para o preço de abertura e fechamento do dia anterior
btc_data["open_shift"] = btc_data["open"].shift(1)
btc_data["close_shift"] = btc_data["close"].shift(1)

# Criando uma feature de indicacao (target) onde:
# se a variacao percentual entre a aberura e o fechamento for positiva (> 0.5%), a variavel indicacao recebe 1
# se a variacao percentual entre a aberura e o fechamento for menor que 0.5%, a variavel indicacao recebe 0
btc_data["variation"] = (btc_data["close"] - btc_data["open"]) / btc_data["open"]
btc_data["indication"] = (btc_data["variation"] > 0.005).astype(int)


# Baixar dados do VIX usando yfinance
def fetch_vix_data():
    start_date = btc_data["timestamp"].min().strftime("%Y-%m-%d")
    end_date = btc_data["timestamp"].max().strftime("%Y-%m-%d")
    vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d")
    vix = vix[["Open", "High", "Low", "Close"]].reset_index()
    vix.columns = ["timestamp", "vix_open", "vix_high", "vix_low", "vix_close"]

    # Inserir uma coluna da variação diária e a média diária do VIX
    vix["vix_variation"] = vix["vix_high"] - vix["vix_low"]
    vix["vix_mean"] = (vix["vix_high"] + vix["vix_low"]) / 2

    return vix


vix_data = fetch_vix_data()

# Média móvel (moving average) de 3 dias
vix_data["vix_open_ma3"] = vix_data["vix_open"].rolling(window=3).mean()
vix_data["vix_close_ma3"] = vix_data["vix_close"].rolling(window=3).mean()
vix_data["vix_variation_ma3"] = vix_data["vix_variation"].rolling(window=3).mean()
vix_data["vix_mean_ma3"] = vix_data["vix_mean"].rolling(window=3).mean()

# Unir os dados de BTC e VIX pelo timestamp
btc_data["date"] = btc_data["timestamp"].dt.date
vix_data["date"] = vix_data["timestamp"].dt.date
merged_data = pd.merge(btc_data, vix_data, on="date", how="inner")

# Remover a coluna duplicada de date
merged_data.drop(["date"], axis=1, inplace=True)

# Excluindo as linhas com valores nulos
merged_data.dropna(inplace=True)

# salvando o dataframe em um arquivo CSV
merged_data.to_csv("merged_data.csv", index=False)
