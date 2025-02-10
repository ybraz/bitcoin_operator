import os
import time
import joblib
import logging
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ccxt
import yfinance as yf

# Mostrar todas as colunas
pd.set_option("display.max_columns", None)

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Arquivo de cache para os dados de mercado
CACHE_FILE = "market_data_cache.pkl"
cached_market_data = None

# Carrega o pipeline treinado (LogisticRegression com max_iter=10000, class_weight='balanced',
# C=100.0 e solver='lbfgs'.)
# StandardScaler() para escalonar os dados
logger.info("Carregando o pipeline treinado (StandardScaler + PCA + Classifier)...")
pipeline = joblib.load("bitcoin_model.pkl")

# Lista das features utilizadas no treinamento (dados brutos)
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

# Criação do aplicativo FastAPI e configuração de CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_all_binance_data(symbol="BTC/USDT", timeframe="1d", days_limit=None):
    logger.info("Buscando dados do Binance...")
    exchange = ccxt.binance()
    start_dt = (
        datetime.now() - timedelta(days=days_limit)
        if days_limit
        else datetime(2017, 1, 1)
    )
    since = exchange.parse8601(start_dt.strftime("%Y-%m-%dT00:00:00Z"))
    all_data = []
    max_limit = 1000
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=max_limit
            )
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(1)
        except Exception as e:
            logger.error(f"Erro ao buscar dados do Binance: {e}")
            break
    if not all_data:
        raise ValueError("Não foi possível coletar dados do Binance.")
    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values(
        "timestamp"
    )
    return df


def fetch_vix_data():
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d")
    vix = vix[["Open", "High", "Low", "Close"]].reset_index()
    vix.columns = ["timestamp", "vix_open", "vix_high", "vix_low", "vix_close"]
    vix["vix_variation"] = vix["vix_high"] - vix["vix_low"]
    vix["vix_mean"] = (vix["vix_high"] + vix["vix_low"]) / 2
    return vix


def process_and_merge_data():
    btc_data = fetch_all_binance_data(days_limit=10).copy()
    btc_data["open_ma3"] = btc_data["open"].rolling(window=3).mean()
    btc_data["close_ma3"] = btc_data["close"].rolling(window=3).mean()
    btc_data["volume_ma3"] = btc_data["volume"].rolling(window=3).mean()
    btc_data["high_ma3"] = btc_data["high"].rolling(window=3).mean()
    btc_data["low_ma3"] = btc_data["low"].rolling(window=3).mean()
    btc_data["variation"] = (btc_data["close"] - btc_data["open"]) / btc_data["open"]
    btc_data["indication"] = (btc_data["variation"] > 0.005).astype(int)
    btc_data["date"] = btc_data["timestamp"].dt.date
    btc_data["open_shift"] = btc_data["open"].shift(1)
    btc_data["close_shift"] = btc_data["close"].shift(1)

    vix_data = fetch_vix_data().copy()
    vix_data["date"] = pd.to_datetime(vix_data["timestamp"]).dt.date
    vix_data["vix_open_ma3"] = vix_data["vix_open"].rolling(window=3).mean()
    vix_data["vix_close_ma3"] = vix_data["vix_close"].rolling(window=3).mean()
    vix_data["vix_variation_ma3"] = vix_data["vix_variation"].rolling(window=3).mean()
    vix_data["vix_mean_ma3"] = vix_data["vix_mean"].rolling(window=3).mean()

    btc_data.reset_index(drop=False, inplace=True)
    merged_data = pd.merge(
        btc_data, vix_data, on="date", how="inner", suffixes=("", "_vix")
    )
    merged_data = merged_data.ffill()
    merged_data.dropna(inplace=True)
    if "index" in merged_data.columns:
        merged_data.drop("index", axis=1, inplace=True)
    return merged_data


def load_cache_from_file():
    global cached_market_data
    if os.path.exists(CACHE_FILE):
        logger.info("Carregando cache de dados do arquivo.")
        with open(CACHE_FILE, "rb") as f:
            cached_market_data = joblib.load(f)
    else:
        logger.info("Cache não encontrado. Coletando dados iniciais.")
        cached_market_data = process_and_merge_data()
        with open(CACHE_FILE, "wb") as f:
            joblib.dump(cached_market_data, f)


@app.on_event("startup")
def startup_event():
    load_cache_from_file()


@app.get("/market-data")
def market_data():
    """
    Retorna indicadores da última linha dos dados de mercado e a cotação atual do BTC (em tempo real).
    """
    try:
        if cached_market_data is None or cached_market_data.empty:
            raise ValueError("Dados de mercado não carregados corretamente.")
        last_row = cached_market_data.iloc[-1]
        # Coleta o preço atual do BTC em tempo real
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker("BTC/USDT")
        btc_current = ticker.get("last")
        return {
            "date": str(last_row["date"]),
            "btc_open": float(last_row["open"]),
            "btc_close": float(
                last_row["close"]
            ),  # Preço de fechamento do candle diário
            "btc_current": round(float(btc_current), 2)
            if btc_current is not None
            else None,
            "btc_close_ma3": float(last_row["close_ma3"]),
            "vix_open": float(last_row["vix_open"]),
            "vix_close_ma3": float(last_row["vix_close_ma3"]),
        }
    except Exception as e:
        logger.error(f"Erro ao obter dados de mercado: {e}")
        return {"error": str(e)}


@app.get("/btc-current-price")
def btc_current_price():
    """
    Retorna a cotação atual (last) do par BTC/USDT em tempo real.
    """
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker("BTC/USDT")
        btc_current = ticker.get("last")
        return {
            "btc_current": round(float(btc_current), 2)
            if btc_current is not None
            else None
        }
    except Exception as e:
        logger.error(f"Erro ao obter a cotação atual do BTC: {e}")
        return {"btc_current": "Indisponível"}


@app.get("/vix-current-price")
def vix_current_price():
    """
    Retorna a cotação atual (fechamento) do VIX.
    """
    try:
        # Utiliza o método Ticker do yfinance para obter o histórico
        ticker = yf.Ticker("^VIX")
        df = ticker.history(period="5d")
        if not df.empty:
            current_price = df["Close"].values[-1]
            return {"current_price": round(float(current_price), 2)}
        raise ValueError("Nenhum dado do VIX disponível.")
    except Exception as e:
        logger.error(f"Erro ao obter o preço do VIX: {e}")
        return {"current_price": "Indisponível"}


@app.post("/refresh-cache")
def refresh_cache():
    """
    Atualiza o cache de dados de mercado.
    """
    global cached_market_data
    try:
        new_data = process_and_merge_data()
        cached_market_data = new_data
        with open(CACHE_FILE, "wb") as f:
            joblib.dump(cached_market_data, f)
        return {"message": "Cache de dados atualizado com sucesso."}
    except Exception as e:
        logger.error(f"Erro ao atualizar o cache: {e}")
        return {"error": str(e)}


@app.post("/predict")
def predict():
    """
    Gera a predição usando o pipeline treinado.
    Seleciona as colunas conforme a lista FEATURE_COLUMNS e utiliza a última linha para predição.
    """
    try:
        if cached_market_data is None or cached_market_data.empty:
            raise ValueError("Dados de mercado indisponíveis para previsão.")
        missing_cols = [
            col for col in FEATURE_COLUMNS if col not in cached_market_data.columns
        ]
        if missing_cols:
            raise ValueError(f"Colunas de features ausentes: {missing_cols}")
        feature_data = cached_market_data[FEATURE_COLUMNS].iloc[[-1]].fillna(0)

        # Dados que estão sendo considerados para previsão
        logger.info(f"Feature data: {feature_data}")

        prediction = pipeline.predict(feature_data)[0]
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "predicted_class": int(prediction),
        }
    except Exception as e:
        logger.error(f"Erro ao gerar a previsão: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
