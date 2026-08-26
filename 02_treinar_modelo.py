# ═══════════════════════════════════════════════════════════════════
# SCRIPT 02 — TREINAMENTO DO MODELO
# ═══════════════════════════════════════════════════════════════════
# Objetivo: Treinar o ensemble soft voting (SVM + NB + LR) sobre
#           os 50 crashes de treino do Firebase Crashlytics.
#
# Alinhamento com o TAP e Cap. 2/3:
# - Ensemble soft voting SVM + NB + LR (NAMDAR et al., 2025)
# - TF-IDF com bigramas (NICE, 2025; TASKIRAN et al., 2025)
# - SMOTE apenas no treino (CHAWLA et al., 2002; AHMED et al., 2023)
# - Features de logs mobile (JORAYEVA et al., 2022)
# - Categoria de exceção do BLAME_FRAME_OWNER (GHADESI et al., 2023)
# ═══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix, issparse


# 1. CARREGA OS DADOS DE TREINO
ARQUIVO = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026.xlsx"
df = pd.read_excel(ARQUIVO, sheet_name="Crashes Treino")

print("=" * 65)
print("TREINAMENTO DO ENSEMBLE SOFT VOTING")
print("=" * 65)
print(f"Dataset de treino: {df.shape[0]} crashes rotulados")


# 2. SEPARA FEATURES (X) E RÓTULO (y)
y = df["severity"]


# 3. PRÉ-PROCESSAMENTO: FEATURES TEXTUAIS COM TF-IDF + BIGRAMAS
# Justificativa (NICE, 2025): TF-IDF com bigramas eh suficiente para
# boa predicao com scikit-learn, sem necessidade de DistilBERT.
# Bigramas capturam padroes como "NullPointer Exception" ou
# "OutOfMemory Error" que unigramas isolados perderiam.

print("\n[1/5] Pre-processando features textuais com TF-IDF + bigramas...")

texto = (
    df["exceptions_type"].astype(str) + " " +
    df["exceptions_exception_message"].astype(str) + " " +
    df["blame_frame_file"].astype(str) + " " +
    df["blame_frame_symbol"].astype(str)
)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=150,
    lowercase=True,
    token_pattern=r"[a-zA-Z]+"
)
X_texto = tfidf.fit_transform(texto)
print(f"   Features textuais geradas: {X_texto.shape[1]}")


# 4. PRÉ-PROCESSAMENTO: FEATURES CATEGÓRICAS COM ONE-HOT
# Justificativa (JORAYEVA et al., 2022): features de plataforma sao
# validadas para apps mobile.
# Justificativa (GHADESI et al., 2023): blame_frame_owner classifica
# a origem do crash (DEVELOPER, VENDOR, RUNTIME, PLATFORM, SYSTEM).

print("\n[2/5] Pre-processando features categoricas com one-hot encoding...")

colunas_categoricas = [
    "platform",
    "operating_system_name",
    "device_manufacturer",
    "error_type",
    "process_state",
    "blame_frame_owner"
]

onehot = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_categorico = onehot.fit_transform(df[colunas_categoricas])
print(f"   Features categoricas geradas: {X_categorico.shape[1]}")


# 5. PRÉ-PROCESSAMENTO: FEATURES NUMÉRICAS
# occurrences_count e users_affected sao fatores confirmados no
# questionario como determinantes da severidade.

print("\n[3/5] Preparando features numericas...")

colunas_numericas = [
    "blame_frame_line",
    "occurrences_count",
    "users_affected"
]

X_numerico = csr_matrix(df[colunas_numericas].values.astype(float))
print(f"   Features numericas: {X_numerico.shape[1]}")


# 6. COMBINA TUDO EM UMA ÚNICA MATRIZ X
X = hstack([X_texto, X_categorico, X_numerico])
print(f"\n   Matriz final X: {X.shape[0]} linhas x {X.shape[1]} colunas")


# 7. DIVISÃO TREINO/TESTE INTERNA (para avaliar depois no script 03)
print("\n[4/5] Dividindo em treino (80%) e teste (20%) para avaliacao interna...")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f"   Treino: {X_treino.shape[0]} crashes")
print(f"   Teste:  {X_teste.shape[0]} crashes")


# 8. SMOTE APLICADO APENAS NO TREINO
# Regra critica (CHAWLA et al., 2002; TASKIRAN et al., 2025):
# SMOTE nunca deve ser aplicado no conjunto de teste - isso
# geraria data leakage e superestimaria o desempenho.

print("\n[5/5] Aplicando SMOTE apenas no treino (evita data leakage)...")

X_treino_denso = X_treino.toarray() if issparse(X_treino) else X_treino
X_treino_denso = np.clip(X_treino_denso, a_min=0, a_max=None)

smote = SMOTE(k_neighbors=5, random_state=42)
X_treino_bal, y_treino_bal = smote.fit_resample(X_treino_denso, y_treino)

print(f"   Antes do SMOTE:  {dict(pd.Series(y_treino).value_counts())}")
print(f"   Depois do SMOTE: {dict(pd.Series(y_treino_bal).value_counts())}")


# 9. CRIA E TREINA O ENSEMBLE SOFT VOTING
# Justificativa (NAMDAR et al., 2025; ALI et al., 2024):
# ensemble soft voting com SVM + NB + LR supera classificadores
# isolados.

print("\n" + "-" * 65)
print("Treinando ensemble soft voting (SVM + NB + LR)...")
print("-" * 65)

svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
nb = MultinomialNB(alpha=0.5)
lr = LogisticRegression(max_iter=1000, random_state=42)

ensemble = VotingClassifier(
    estimators=[("svm", svm), ("nb", nb), ("lr", lr)],
    voting="soft"
)

ensemble.fit(X_treino_bal, y_treino_bal)
print("OK - Modelo treinado com sucesso!")


# 10. SALVA O MODELO E OS PRÉ-PROCESSADORES
print("\nSalvando modelo e pre-processadores...")

joblib.dump(ensemble, "modelo.joblib")
joblib.dump(tfidf, "tfidf.joblib")
joblib.dump(onehot, "onehot.joblib")
joblib.dump(colunas_categoricas, "colunas_categoricas.joblib")
joblib.dump(colunas_numericas, "colunas_numericas.joblib")
joblib.dump(X_teste, "X_teste.joblib")
joblib.dump(y_teste, "y_teste.joblib")

print("   Arquivos salvos:")
print("     - modelo.joblib")
print("     - tfidf.joblib, onehot.joblib")
print("     - colunas_categoricas.joblib, colunas_numericas.joblib")
print("     - X_teste.joblib, y_teste.joblib")

print("\n" + "=" * 65)
print("TREINAMENTO CONCLUIDO")
print("=" * 65)
