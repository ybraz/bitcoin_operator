// ======================
// CONFIGURAÇÕES GERAIS
// ======================
const baseURL = "http://127.0.0.1:8080"; // Ajuste se sua API estiver noutro endpoint

// Quando a página carrega, definimos a data atual e tentamos carregar dados
document.addEventListener("DOMContentLoaded", () => {
  const currentDateEl = document.getElementById("current-date");
  currentDateEl.textContent = "Data: " + new Date().toLocaleDateString("pt-BR");
  loadMarketData();
});

// ======================
// FUNÇÕES DE UTILIDADE
// ======================
function formatNumber(num) {
  if (num === null || num === undefined || isNaN(num)) return "--";
  // Formato local "pt-BR" com duas casas decimais
  return Number(num).toLocaleString("pt-BR", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function showError(message) {
  const errorContainer = document.getElementById("error-container");
  errorContainer.textContent = message;
  errorContainer.style.display = "block";
}

function hideError() {
  const errorContainer = document.getElementById("error-container");
  errorContainer.style.display = "none";
}

function showLoading() {
  const overlay = document.getElementById("loading-overlay");
  overlay.style.display = "flex";
}

function hideLoading() {
  const overlay = document.getElementById("loading-overlay");
  overlay.style.display = "none";
}

// ======================
// CHAMADAS À API
// ======================
async function loadMarketData() {
  try {
    hideError();
    showLoading();

    // Pega /market-data
    const responseMarket = await fetch(`${baseURL}/market-data`, { cache: "no-cache" });
    if (!responseMarket.ok) {
      throw new Error("Erro ao obter dados de mercado (/market-data).");
    }
    const marketData = await responseMarket.json();
    console.log("marketData:", marketData);

    // Se vier erro
    if (marketData.error) {
      throw new Error(marketData.error);
    }

    // Preenche no dashboard
    document.getElementById("btcOpen").textContent = `\$${formatNumber(marketData.btc_open)}`;
    document.getElementById("btcCloseMa3").textContent = `\$${formatNumber(marketData.btc_close_ma3)}`;
    document.getElementById("btcCurrent").textContent = `\$${formatNumber(marketData.btc_current)}`;

    const btcOpen = marketData.btc_open;
    const btcCurrent = marketData.btc_current;
    const variationEl = document.getElementById("btcVariation");
    let variation = 0;
    if (btcOpen && btcCurrent) {
      variation = ((btcCurrent - btcOpen) / btcOpen) * 100;
    }
    // Exibe com seta up/down
    if (variation >= 0) {
      variationEl.innerHTML = `↑ ${formatNumber(variation)}%`;
      variationEl.classList.add("positive");
      variationEl.classList.remove("negative");
      document.getElementById("card-btc-current").classList.add("positive");
      document.getElementById("card-btc-current").classList.remove("negative");
    } else {
      variationEl.innerHTML = `↓ ${formatNumber(variation)}%`;
      variationEl.classList.add("negative");
      variationEl.classList.remove("positive");
      document.getElementById("card-btc-current").classList.add("negative");
      document.getElementById("card-btc-current").classList.remove("positive");
    }

    document.getElementById("vixOpen").textContent = formatNumber(marketData.vix_open);
    document.getElementById("vixCloseMa3").textContent = formatNumber(marketData.vix_close_ma3);

    // Agora pegamos a cotação "atual" do VIX (/vix-current-price)
    const vixResponse = await fetch(`${baseURL}/vix-current-price`, { cache: "no-cache" });
    if (!vixResponse.ok) {
      throw new Error("Erro ao obter a cotação do VIX (/vix-current-price).");
    }
    const vixData = await vixResponse.json();
    if (vixData.error) {
      throw new Error(vixData.error);
    }
    document.getElementById("vixCurrentPrice").textContent = formatNumber(vixData.current_price);

    // Exibe o container de market-data
    document.getElementById("market-data").style.display = "grid";

    // Atualiza a "última atualização"
    const lastUpdatedEl = document.getElementById("last-updated");
    const lastUpdatedValueEl = document.getElementById("last-updated-value");
    lastUpdatedValueEl.textContent = new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
    lastUpdatedEl.style.display = "block";
  } catch (err) {
    console.error(err);
    showError("Falha ao carregar dados de mercado: " + err.message);
  } finally {
    hideLoading();
  }
}

async function refreshMarketCache() {
  try {
    hideError();
    showLoading();
    const response = await fetch(`${baseURL}/refresh-cache`, { method: "POST", cache: "no-cache" });
    if (!response.ok) {
      throw new Error("Erro ao atualizar o cache de mercado (/refresh-cache).");
    }
    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }
    alert(data.message || "Cache atualizado com sucesso.");
    // Recarrega os dados de mercado
    await loadMarketData();
  } catch (err) {
    console.error(err);
    showError("Erro ao atualizar cache de mercado: " + err.message);
  } finally {
    hideLoading();
  }
}

async function getPrediction() {
  try {
    hideError();
    showLoading();
    const response = await fetch(`${baseURL}/predict`, { method: "POST", cache: "no-cache" });
    if (!response.ok) {
      throw new Error("Erro ao obter a previsão (/predict).");
    }
    const data = await response.json();
    console.log("Prediction data:", data);

    if (data.error) {
      throw new Error(data.error);
    }

    document.getElementById("predictDate").textContent = data.date || "--";
    const predictRecommendationEl = document.getElementById("predictRecommendation");
    if (data.predicted_class === 1) {
      predictRecommendationEl.textContent = "Operar (possível alta)";
      predictRecommendationEl.className = "recommendation success";
    } else {
      predictRecommendationEl.textContent = "Não operar (possível queda)";
      predictRecommendationEl.className = "recommendation danger";
    }

    document.getElementById("prediction-result").style.display = "block";
  } catch (err) {
    console.error(err);
    showError("Erro ao obter a previsão: " + err.message);
  } finally {
    hideLoading();
  }
}
