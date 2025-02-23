import os
import time
import joblib
import logging
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ccxt
import requests
from io import StringIO

# Configurações de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nome do arquivo de cache
CACHE_FILE = "market_data_cache.pkl"

# Objetos globais
cached_market_data = None  # DataFrame (BTC + VIX)
cached_current_prices = {"btc": None, "vix": None}
last_fetched_time = {"btc": None, "vix": None}

# Intervalo de tempo mínimo (em minutos) para atualizar cotações "ao vivo"
LIVE_PRICE_INTERVAL_MINUTES = 60

# Carrega seu pipeline (modelo) treinado
logger.info("Carregando o pipeline treinado...")
pipeline = joblib.load("bitcoin_model.pkl")

# Colunas usadas no modelo
FEATURE_COLUMNS = [
    "open_ma3",
    "close_ma3",
    "volume_ma3",
    "high_ma3",
    "low_ma3",
    "open_shift",
    "close_shift",
    "vix_open_ma3",
    "vix_close_ma3",
    "vix_variation_ma3",
    "vix_mean_ma3",
]

# Criação do aplicativo FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# FUNÇÕES DE APOIO PARA BTC
# ======================================

def fetch_btc_ohlcv_daily_until_yesterday(symbol="BTC/USDT", days=10):
    """
    Busca candles 1d da Binance até o dia anterior (exclui hoje).
    Ex: se hoje é 2025-02-23, pega até 2025-02-22.
    Tudo em timestamps naive (sem timezone).
    """
    logger.info("fetch_btc_ohlcv_daily_until_yesterday...")
    exchange = ccxt.binance()

    # data "de hoje" em naive
    end_dt = datetime.utcnow().date()  
    start_dt = end_dt - timedelta(days=days)

    since_ts = exchange.parse8601(start_dt.strftime("%Y-%m-%dT00:00:00Z"))
    ohlcv = exchange.fetch_ohlcv(symbol, "1d", since=since_ts, limit=days + 5)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # naive
    df.sort_values("timestamp", inplace=True)

    # filtra tudo que seja < end_dt (ou seja, até ontem)
    mask = df["timestamp"].dt.date < end_dt
    df = df.loc[mask].copy()

    logger.info(f"fetch_btc_ohlcv_daily_until_yesterday -> retornando {len(df)} candles diários.")
    return df


def fetch_btc_partial_candle_today(symbol="BTC/USDT"):
    """
    Constrói um "candle parcial" para o dia atual (naive):
      - open/high/low: agregado dos candles intraday (1h) desde 00:00 UTC até agora
      - close: igual ao close do dia anterior (candle 1d)
    Retorna um dicionário com { timestamp, open, high, low, close, volume }.
    Se não existir candle intraday, retorna open=None, etc.
    """
    logger.info("fetch_btc_partial_candle_today: construindo candle parcial do dia atual...")

    exchange = ccxt.binance()

    # data de hoje (naive)
    today_date = datetime.utcnow().date()
    yesterday_date = today_date - timedelta(days=1)

    # 1) candle 1d de ontem (para pegar o close de ontem)
    since_ts_yest = exchange.parse8601(
        (yesterday_date - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    )
    daily = exchange.fetch_ohlcv(symbol, "1d", since=since_ts_yest, limit=5)
    df_daily = pd.DataFrame(daily, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"], unit="ms")  # naive
    df_daily.sort_values("timestamp", inplace=True)

    mask_yest = df_daily["timestamp"].dt.date == yesterday_date
    df_yest = df_daily[mask_yest]
    if df_yest.empty:
        logger.warning(f"Nenhum candle 1d encontrado para ontem = {yesterday_date}.")
        return None

    close_yesterday = df_yest.iloc[-1]["close"]

    # 2) candles 1h do dia atual (00:00 UTC até agora)
    # Montamos um datetime naive para 00:00 do dia atual
    since_today = datetime.combine(today_date, datetime.min.time())  # naive
    since_ts_today = exchange.parse8601(since_today.strftime("%Y-%m-%dT00:00:00Z"))

    intraday = exchange.fetch_ohlcv(symbol, "1h", since=since_ts_today, limit=48)
    df_intraday = pd.DataFrame(intraday, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df_intraday["timestamp"] = pd.to_datetime(df_intraday["timestamp"], unit="ms")  # naive
    df_intraday.sort_values("timestamp", inplace=True)

    mask_today = df_intraday["timestamp"].dt.date == today_date
    df_today = df_intraday[mask_today]

    if df_today.empty:
        # Se não há candles de hoje, retornamos um "candle" com open=None e close=ontem
        partial = {
            "timestamp": pd.Timestamp(datetime(today_date.year, today_date.month, today_date.day)),  # naive
            "open": None,
            "high": None,
            "low": None,
            "close": close_yesterday,
            "volume": 0.0,
        }
        return partial

    open_today = df_today.iloc[0]["open"]
    high_today = df_today["high"].max()
    low_today = df_today["low"].min()
    volume_today = df_today["volume"].sum()

    partial = {
        "timestamp": pd.Timestamp(datetime(today_date.year, today_date.month, today_date.day)),  # naive
        "open": open_today,
        "high": high_today,
        "low": low_today,
        "close": close_yesterday,  # "close" do candle parcial = close de ontem
        "volume": volume_today,
    }
    return partial


def fetch_live_btc_price():
    """
    Retorna a cotação atual (last) do BTC/USDT na Binance,
    mas só faz fetch se já passou LIVE_PRICE_INTERVAL_MINUTES desde a última vez.
    """
    global cached_current_prices, last_fetched_time

    now = datetime.utcnow()  # naive
    if last_fetched_time["btc"] is not None:
        diff = now - last_fetched_time["btc"]
        if diff.total_seconds() < LIVE_PRICE_INTERVAL_MINUTES * 60 and cached_current_prices["btc"] is not None:
            logger.info("fetch_live_btc_price: Retornando preço BTC do cache interno (fresco).")
            return cached_current_prices["btc"]

    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker("BTC/USDT")
        last_price = ticker.get("last")
        logger.info(f"fetch_live_btc_price => ticker = {ticker}")
        if last_price is not None:
            cached_current_prices["btc"] = float(last_price)
            last_fetched_time["btc"] = now
            return cached_current_prices["btc"]
    except Exception as e:
        logger.error(f"Erro ao buscar ticker do BTC: {e}")

    # retorna o que tiver em cache, se existir
    return cached_current_prices["btc"]


# ======================================
# FUNÇÕES DE APOIO PARA VIX
# ======================================
def fetch_vix_data(days=30):
    """
    Busca dados do VIX (CSV oficial da CBOE) e filtra pelos últimos X dias.
    Tudo naive. Calcula colunas vix_open_ma3 etc. e insere 'date'.
    """
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    response = requests.get(url)
    response.raise_for_status()

    csv_data = StringIO(response.text)
    vix_raw = pd.read_csv(csv_data)

    # Ajusta colunas
    vix_raw.columns = [col.strip().upper() for col in vix_raw.columns]
    rename_dict = {
        "DATE": "timestamp",
        "OPEN": "vix_open",
        "HIGH": "vix_high",
        "LOW": "vix_low",
        "CLOSE": "vix_close",
    }
    for original, destino in rename_dict.items():
        if original in vix_raw.columns:
            vix_raw.rename(columns={original: destino}, inplace=True)

    vix_raw["timestamp"] = pd.to_datetime(vix_raw["timestamp"])  # naive
    vix_raw["vix_variation"] = vix_raw["vix_high"] - vix_raw["vix_low"]
    vix_raw["vix_mean"] = (vix_raw["vix_high"] + vix_raw["vix_low"]) / 2

    end_dt = datetime.utcnow().date()  
    start_dt = end_dt - timedelta(days=days)
    mask = (vix_raw["timestamp"].dt.date >= start_dt) & (vix_raw["timestamp"].dt.date <= end_dt)
    vix = vix_raw.loc[mask].copy()
    vix.sort_values("timestamp", inplace=True)

    # Calcula médias móveis
    vix["vix_open_ma3"] = vix["vix_open"].rolling(3).mean()
    vix["vix_close_ma3"] = vix["vix_close"].rolling(3).mean()
    vix["vix_variation_ma3"] = vix["vix_variation"].rolling(3).mean()
    vix["vix_mean_ma3"] = vix["vix_mean"].rolling(3).mean()

    # Coluna date
    vix["date"] = vix["timestamp"].dt.date
    return vix


def fetch_live_vix_price():
    """
    Usa o 'vix_close' do último dia do cache como “preço atual” do VIX.
    """
    global cached_current_prices, last_fetched_time, cached_market_data

    now = datetime.utcnow()  # naive
    if last_fetched_time["vix"] is not None:
        diff = now - last_fetched_time["vix"]
        if diff.total_seconds() < LIVE_PRICE_INTERVAL_MINUTES * 60 and cached_current_prices["vix"] is not None:
            logger.info("fetch_live_vix_price: Retornando preço VIX do cache interno (fresco).")
            return cached_current_prices["vix"]

    if cached_market_data is None or cached_market_data.empty:
        logger.warning("cached_market_data indisponível para extrair VIX.")
        return None

    last_row = cached_market_data.iloc[-1]
    vix_close = last_row.get("vix_close", None)
    if vix_close is not None:
        cached_current_prices["vix"] = float(vix_close)
        last_fetched_time["vix"] = now
        return cached_current_prices["vix"]

    return None


# ======================================
# CONSTRUÇÃO DO DATASET (BTC + VIX)
# ======================================
def process_and_merge_data(days=10):
    logger.info("process_and_merge_data => Iniciando montagem do dataset...")

    # 1) Dados diários do BTC até ontem
    df_1d = fetch_btc_ohlcv_daily_until_yesterday("BTC/USDT", days=days)

    # 2) Candle parcial do dia atual
    partial_today = fetch_btc_partial_candle_today("BTC/USDT")
    if partial_today:
        df_partial = pd.DataFrame([partial_today])
        logger.info("Candle parcial de hoje obtido.")
    else:
        df_partial = pd.DataFrame()

    # Concatena e ordena os dados do BTC
    df_btc = pd.concat([df_1d, df_partial], ignore_index=True).sort_values("timestamp")

    # Cálculos de médias móveis e indicadores
    df_btc["open_ma3"] = df_btc["open"].rolling(3).mean()
    df_btc["close_ma3"] = df_btc["close"].rolling(3).mean()
    df_btc["volume_ma3"] = df_btc["volume"].rolling(3).mean()
    df_btc["high_ma3"] = df_btc["high"].rolling(3).mean()
    df_btc["low_ma3"] = df_btc["low"].rolling(3).mean()

    df_btc["variation"] = (df_btc["close"] - df_btc["open"]) / df_btc["open"]
    df_btc["indication"] = (df_btc["variation"] > 0.005).astype(int)
    df_btc["open_shift"] = df_btc["open"].shift(1)
    df_btc["close_shift"] = df_btc["close"].shift(1)
    df_btc["date"] = df_btc["timestamp"].dt.date

    # 3) Dados do VIX
    df_vix = fetch_vix_data(days=30)

    # 4) Merge: usa left join para preservar os dados do BTC mesmo se não houver VIX para hoje
    merged = pd.merge(df_btc, df_vix, on="date", how="left", suffixes=("", "_vix"))
    merged.sort_values("timestamp", inplace=True)

    # Preenche somente as colunas do VIX com forward fill
    vix_cols = [
        "vix_open", "vix_close", "vix_variation", "vix_mean",
        "vix_open_ma3", "vix_close_ma3", "vix_variation_ma3", "vix_mean_ma3"
    ]
    merged[vix_cols] = merged[vix_cols].ffill()

    # Remove apenas linhas sem dados críticos do BTC (por exemplo, 'close')
    merged = merged[merged["close"].notna()]

    logger.info(f"process_and_merge_data => shape final = {merged.shape}")
    return merged


def load_cache_from_file():
    global cached_market_data
    if os.path.exists(CACHE_FILE):
        logger.info("Carregando cache existente do disco.")
        with open(CACHE_FILE, "rb") as f:
            cached_market_data = joblib.load(f)
    else:
        logger.info("Cache não encontrado. Gerando dados iniciais.")
        cached_market_data = process_and_merge_data(days=10)
        with open(CACHE_FILE, "wb") as f:
            joblib.dump(cached_market_data, f)


# ======================================
# ENDPOINTS FASTAPI
# ======================================
@app.on_event("startup")
def startup_event():
    # Carrega cache ou processa inicial
    load_cache_from_file()
    # Já busca cotações "ao vivo"
    fetch_live_btc_price()
    fetch_live_vix_price()


@app.get("/market-data")
def market_data():
    """
    Retorna dados do último candle (que pode ser parcial do dia atual),
    mais a cotação atual do BTC e do VIX (simulado).
    """
    try:
        global cached_market_data
        if cached_market_data is None or cached_market_data.empty:
            raise ValueError("Dados de mercado indisponíveis.")

        last_row = cached_market_data.iloc[-1]
        btc_current = fetch_live_btc_price()
        vix_current = fetch_live_vix_price()

        logger.info(f"/market-data => Candle final: open={last_row['open']}, close={last_row['close']}")
        logger.info(f"   => Candle date = {last_row['timestamp']}")
        logger.info(f"   => btc_current = {btc_current}")

        # Exemplo: devolvendo open, close, current e etc.
        return {
            "date": str(last_row["date"]),
            "btc_open": float(last_row["open"]) if last_row["open"] is not None else None,
            "btc_close": float(last_row["close"]),
            "btc_high": float(last_row["high"]) if last_row["high"] is not None else None,
            "btc_low": float(last_row["low"]) if last_row["low"] is not None else None,
            "btc_close_ma3": float(last_row["close_ma3"]),
            "btc_current": float(btc_current) if btc_current else None,
            "vix_open": float(last_row["vix_open"]),
            "vix_close": float(last_row["vix_close"]),
            "vix_current": float(vix_current) if vix_current else None,
            "vix_close_ma3": float(last_row["vix_close_ma3"]),
        }
    except Exception as e:
        logger.error(f"Erro em /market-data: {e}")
        return {"error": str(e)}


@app.get("/vix-current-price")
def vix_current_price():
    """
    Retorna o 'preço atual' do VIX (simulado pelo último vix_close).
    """
    try:
        current_price = fetch_live_vix_price()
        logger.info(f"/vix-current-price => Devolvendo {current_price}")
        return {"current_price": float(current_price) if current_price else None}
    except Exception as e:
        logger.error(f"Erro ao obter o preço do VIX: {e}")
        return {"error": str(e)}


@app.post("/refresh-cache")
def refresh_cache():
    """
    Regera o dataset e zera o cache de preços ao vivo
    """
    global cached_market_data, cached_current_prices, last_fetched_time
    try:
        logger.info("refresh_cache => Recriando dataset.")
        new_data = process_and_merge_data(days=10)
        cached_market_data = new_data

        with open(CACHE_FILE, "wb") as f:
            joblib.dump(cached_market_data, f)

        # Zera cache
        cached_current_prices = {"btc": None, "vix": None}
        last_fetched_time = {"btc": None, "vix": None}
        fetch_live_btc_price()
        fetch_live_vix_price()

        return {"message": "Cache atualizado com sucesso."}
    except Exception as e:
        logger.error(f"Erro em /refresh-cache: {e}")
        return {"error": str(e)}


@app.post("/predict")
def predict():
    """
    Faz a previsão usando o pipeline treinado,
    usando os dados da última linha do dataset
    (que pode ser parcial do dia atual).
    """
    global cached_market_data
    try:
        if cached_market_data is None or cached_market_data.empty:
            raise ValueError("Dados indisponíveis para previsão.")

        missing_cols = [c for c in FEATURE_COLUMNS if c not in cached_market_data.columns]
        if missing_cols:
            raise ValueError(f"Colunas ausentes: {missing_cols}")

        X = cached_market_data[FEATURE_COLUMNS].iloc[[-1]].fillna(0)
        prediction = pipeline.predict(X)[0]
        return {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),  # naive
            "predicted_class": int(prediction),
        }
    except Exception as e:
        logger.error(f"Erro em /predict: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
