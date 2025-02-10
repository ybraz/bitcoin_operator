import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Carregar os dados gerados (merged_data.csv)
data = pd.read_csv("merged_data.csv")

# Seleciona apenas as colunas numéricas (incluindo as 21 features)
data = data.select_dtypes(include=[np.number])

# Excluir colunas indesejadas
colunas_excluir = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "variation",
    "vix_open",
    "vix_high",
    "vix_low",
    "vix_close",
    "vix_variation",
    "vix_mean",
]
data = data.drop(columns=colunas_excluir)

# Separar variáveis de entrada e saída
X = data.drop("indication", axis=1)
y = data["indication"]

# Cria o pipeline com pré-processamento e classificador
# Para o modelo vencedor, usamos:
#   - StandardScaler() para escalonar os dados;
#   - "passthrough" no lugar do PCA (ou seja, não aplica PCA);
#   - LogisticRegression com max_iter=10000, class_weight='balanced', C=100.0 e solver='lbfgs'.
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", "passthrough"),
        (
            "clf",
            LogisticRegression(
                max_iter=10000, class_weight="balanced", C=100.0, solver="lbfgs"
            ),
        ),
    ]
)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Treinar o pipeline
pipeline.fit(X_train, y_train)

# Avaliar o modelo
predictions = pipeline.predict(X_test)
print("Matriz de confusão:")
print(confusion_matrix(y_test, predictions))
print("\nRelatório de classificação:")
print(classification_report(y_test, predictions))

# Salvar o pipeline treinado para uso na API
joblib.dump(pipeline, "bitcoin_model.pkl")
print("Modelo treinado e salvo como 'bitcoin_model.pkl'")
