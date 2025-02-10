# Bitcoin Price Prediction Dashboard

Este é um projeto avançado de machine learning que recomenda se é recomendável ou não operar em Bitcoin (BTC) no dia atual. Ele utiliza uma arquitetura de coleta de dados, pré-processamento, treinamento de modelos preditivos e uma interface web interativa para exibição dos resultados.

---

## 🔧 Estrutura do Repositório

> ├── bitcoin_criar_dataset.py      # Script de coleta e criação do dataset  
> ├── bitcoin_treinar_modelo.py     # Script para treinamento do modelo preditivo  
> ├── main.py                       # Código principal da API FastAPI  
> ├── index.html                    # Interface do usuário (dashboard)  
> └── README.md                     # Documentação do projeto  

---

## 🛠️ Explicação de cada arquivo

### **1. bitcoin_criar_dataset.py**  
Este arquivo é responsável por criar o dataset que será usado para treinar o modelo. Ele busca dados históricos de preços do Bitcoin utilizando as bibliotecas `ccxt` (para exchanges como Binance) e `yfinance` (para índices econômicos como o VIX). A lógica do processo é a seguinte:

- **Passo 1**: Coleta os preços de abertura, fechamento, máximas e mínimas do BTC em uma determinada exchange (ex.: Binance).  
- **Passo 2**: Obtém dados históricos de volatilidade, como o índice VIX, para correlacionar eventos de alta volatilidade com movimentações do BTC.  
- **Passo 3**: Calcula indicadores financeiros, como médias móveis (3 dias), variações percentuais e outros sinais.  
- **Passo 4**: Normaliza e salva o dataset em um arquivo CSV para ser usado no treinamento.

### **2. bitcoin_treinar_modelo.py**  
Este script realiza o treinamento do modelo preditivo. Ele utiliza a biblioteca `scikit-learn` para aplicar classificadores binários (como Random Forest ou XGBoost). Aqui está o fluxo:

- **Passo 1**: Carrega o dataset criado pelo script anterior.  
- **Passo 2**: Divide o dataset em conjuntos de treino e teste.  
- **Passo 3**: Cria features de entrada com base nos indicadores calculados (médias móveis, volatilidade, etc.).  
- **Passo 4**: Treina o modelo utilizando um classificador binário. O objetivo é prever se o BTC terá alta ou queda no fechamento do dia (considerando alta apenas a partir de 5% positivo, para reduzir riscos de operação).  
- **Passo 5**: Avalia a acurácia usando métricas como F1-score, precisão e recall, e salva o modelo treinado.

### **3. main.py**  
O arquivo principal da API, desenvolvido com FastAPI, expõe os seguintes endpoints:

- **GET /market-data**  
  Retorna os dados de mercado em tempo real, incluindo os preços de abertura, variações, médias móveis e o índice de volatilidade VIX.

- **POST /predict**  
  Utiliza o modelo treinado para prever se o BTC terá uma alta ou queda. O retorno inclui uma recomendação (Operar ou Não Operar).

- **POST /refresh-cache**  
  Atualiza o cache de dados de mercado, buscando informações mais recentes nas exchanges e fontes externas.

---

## 🌐 Como funciona a API 

1. **Iniciar a API**  
   Execute o comando:  
   > uvicorn main:app --reload

2. **Interagir com os endpoints**  
   Acesse `http://127.0.0.1:8080` e utilize os endpoints:

   - **GET /market-data**  
     Este endpoint retorna os dados do BTC e do VIX atualizados, que são exibidos no dashboard.

   - **POST /predict**  
     Envia os dados atuais de mercado ao modelo de machine learning e retorna uma recomendação:  
     - **Operar (possível alta)**  
     - **Não operar (possível queda)**  

   - **POST /refresh-cache**  
     Este endpoint é útil para forçar a atualização dos dados sem precisar reiniciar o sistema.

---

## 🎨 Interface do Dashboard

A interface foi projetada para ser limpa e interativa. Ao abrir o arquivo `index.html`, você verá:

- **Indicadores principais:**  
  - Preço atual do BTC e comparação com a abertura do dia  
  - Médias móveis e variações percentuais  
  - Preço do índice VIX  

- **Seção de previsão:**  
  Mostra a recomendação gerada pelo modelo de machine learning com base nas condições de mercado.

---

## 🛠️ Requisitos

- Python 3.8+  
- FastAPI  
- pandas  
- scikit-learn  
- ccxt  
- yfinance  
- uvicorn  

Instale as dependências executando:  
> pip install -r requirements.txt

---

## 📜 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.