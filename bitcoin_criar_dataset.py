import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import time
import requests
from io import StringIO

# =============================================================================
# Função para baixar dados de OHLCV do par BTC/USDT (Binance)
# OHLCV = Open, High, Low, Close, Volume
# =============================================================================
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
        since = ohlcv[-1][0] + 1  # atualiza "since" para continuar baixando
        print(f"Baixados {len(all_data)} registros...")
        time.sleep(1)  # pequeno intervalo para evitar limite de requisições

    # Log de debug: vamos ver as primeiras 5 linhas brutas de all_data
    print("\nExemplo de registros de 'all_data' (primeiros 5):")
    print(all_data[:5])

    # Monta o DataFrame
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Ordena e mantém apenas as colunas desejadas
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")
    return df

# -----------------------------------------------------------------------------
# Função para baixar dados do VIX diretamente do CSV oficial da CBOE
# -----------------------------------------------------------------------------
def fetch_vix_data(btc_data):
    """
    Faz o download dos dados históricos do VIX a partir do CSV oficial da CBOE.
    Filtra o período com base no DataFrame 'btc_data'.
    """
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    response = requests.get(url)
    response.raise_for_status()  # Lança exceção se houver erro na requisição

    csv_data = StringIO(response.text)
    vix_raw = pd.read_csv(csv_data)

    print("\nColunas originais do CSV da CBOE (VIX):", vix_raw.columns.tolist())
    print("Exemplo das primeiras 5 linhas (brutas) do VIX:")
    print(vix_raw.head())

    # Normaliza todas as colunas para maiúsculas
    vix_raw.columns = [col.strip().upper() for col in vix_raw.columns]

    # Dicionário para renomear
    rename_dict = {
        "DATE": "timestamp",
        "OPEN": "vix_open",
        "HIGH": "vix_high",
        "LOW": "vix_low",
        "CLOSE": "vix_close",
    }
    # Renomeia apenas se a coluna existir
    for original, destino in rename_dict.items():
        if original in vix_raw.columns:
            vix_raw.rename(columns={original: destino}, inplace=True)

    # Verifica se a coluna "timestamp" realmente existe depois do rename
    if "timestamp" not in vix_raw.columns:
        raise ValueError(
            f"Coluna 'timestamp' não encontrada após o rename. "
            f"Colunas disponíveis: {vix_raw.columns.tolist()}"
        )

    # Converte para datetime
    vix_raw["timestamp"] = pd.to_datetime(vix_raw["timestamp"])

    # Filtra o período com base na data mínima e máxima do DataFrame de BTC
    start_date = btc_data["timestamp"].min()
    end_date = btc_data["timestamp"].max()
    print("\n[DEBUG] Período do BTC para filtragem do VIX:")
    print("  data mínima:", start_date, "| data máxima:", end_date)

    mask = (vix_raw["timestamp"] >= start_date) & (vix_raw["timestamp"] <= end_date)
    vix = vix_raw.loc[mask].copy()

    # Ordena por timestamp
    vix.sort_values("timestamp", inplace=True)

    # Cria colunas de variação e média
    vix["vix_variation"] = vix["vix_high"] - vix["vix_low"]
    vix["vix_mean"] = (vix["vix_high"] + vix["vix_low"]) / 2

    return vix.reset_index(drop=True)

# =============================================================================
# INÍCIO DO SCRIPT PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # 1) Baixar dados BTC/USDT
    btc_data = fetch_all_binance_data()

    # Logs de debug após baixar BTC
    print("\n[DEBUG] btc_data.shape:", btc_data.shape)
    print("[DEBUG] btc_data.head():")
    print(btc_data.head())
    print("[DEBUG] btc_data.dtypes:")
    print(btc_data.dtypes)
    print("Min data BTC:", btc_data["timestamp"].min(), 
          "| Max data BTC:", btc_data["timestamp"].max())

    # 2) Criar colunas de média móvel (moving average) de 3 dias
    btc_data["open_ma3"] = btc_data["open"].rolling(window=3).mean()
    btc_data["close_ma3"] = btc_data["close"].rolling(window=3).mean()
    btc_data["volume_ma3"] = btc_data["volume"].rolling(window=3).mean()
    btc_data["high_ma3"] = btc_data["high"].rolling(window=3).mean()
    btc_data["low_ma3"] = btc_data["low"].rolling(window=3).mean()

    # 3) Criar colunas de shift (dia anterior) para open e close
    btc_data["open_shift"] = btc_data["open"].shift(1)
    btc_data["close_shift"] = btc_data["close"].shift(1)

    # 4) Criar a feature 'indication' (target) se a variação percentual do dia for > 0.5%
    btc_data["variation"] = (btc_data["close"] - btc_data["open"]) / btc_data["open"]
    btc_data["indication"] = (btc_data["variation"] > 0.005).astype(int)

    # 5) Baixar dados do VIX (filtrado pelo período do BTC)
    vix_data = fetch_vix_data(btc_data)

    # Logs de debug após baixar/filtrar VIX
    print("\n[DEBUG] vix_data.shape:", vix_data.shape)
    print("[DEBUG] vix_data.head():")
    print(vix_data.head())
    print("[DEBUG] vix_data.dtypes:")
    print(vix_data.dtypes)
    if not vix_data.empty:
        print("Min data VIX:", vix_data["timestamp"].min(), 
              "| Max data VIX:", vix_data["timestamp"].max())
    else:
        print("vix_data está vazio (shape == 0 linhas)!")

    # 6) Calcular médias móveis para o VIX
    vix_data["vix_open_ma3"] = vix_data["vix_open"].rolling(window=3).mean()
    vix_data["vix_close_ma3"] = vix_data["vix_close"].rolling(window=3).mean()
    vix_data["vix_variation_ma3"] = vix_data["vix_variation"].rolling(window=3).mean()
    vix_data["vix_mean_ma3"] = vix_data["vix_mean"].rolling(window=3).mean()

    # 7) Fazer o merge dos dados BTC e VIX pela data (dia)
    # Cria colunas 'date' a partir do timestamp
    btc_data["date"] = btc_data["timestamp"].dt.date
    vix_data["date"] = vix_data["timestamp"].dt.date

    merged_data = pd.merge(btc_data, vix_data, on="date", how="inner")

    print("\n[DEBUG] merged_data.shape:", merged_data.shape)
    print("[DEBUG] merged_data.head():")
    print(merged_data.head())
    if not merged_data.empty:
        print("Min data merged:", merged_data['timestamp_x'].min(), 
              "| Max data merged:", merged_data['timestamp_x'].max())
    else:
        print("merged_data está vazio (0 linhas)!")

    # 8) Remover coluna duplicada "date"
    merged_data.drop(["date"], axis=1, inplace=True)

    # 9) Excluir linhas com valores nulos
    merged_data.dropna(inplace=True)

    # 10) Salvar resultado final em CSV
    merged_data.to_csv("merged_data.csv", index=False)
    print("\nDataset final salvo em 'merged_data.csv'!\n")

    print("[FIM DO SCRIPT] Verifique os logs acima para entender onde as datas podem estar se perdendo.")
