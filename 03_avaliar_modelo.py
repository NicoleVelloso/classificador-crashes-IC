# ═══════════════════════════════════════════════════════════════════
# SCRIPT 03 — AVALIAÇÃO DO MODELO COM K-FOLD
# ═══════════════════════════════════════════════════════════════════
# Objetivo: Avaliar o modelo com validacao cruzada k=5.
#           Roda 5 experimentos com divisoes diferentes e calcula
#           a media das metricas, garantindo resultado robusto.
#
# Alinhamento com o TAP:
# - Metricas: AUC (principal), F1 critico, matriz de confusao,
#   taxa de concordancia
# - Criterio de sucesso: AUC >= 0,80 e concordancia >= 75%
#
# Alinhamento com a revisao bibliografica:
# - AUC como metrica principal (CHAWLA et al., 2002; AHMED et al., 2023)
# - F1 da classe critica (NICE, 2025)
# - Comparacao com classificacao manual (BHATTACHARYA; NEAMTIU, 2012)
# ═══════════════════════════════════════════════════════════════════

import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix


# 1. CARREGA E PROCESSA OS DADOS
ARQUIVO = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026.xlsx"
df = pd.read_excel(ARQUIVO, sheet_name="Crashes Treino")

print("=" * 65)
print("AVALIACAO DO MODELO - K-FOLD CROSS-VALIDATION (k=5)")
print("=" * 65)

# Pre-processamento (igual ao script 02, mas usando fit em todos os dados)
texto = (
    df["exceptions_type"].astype(str) + " " +
    df["exceptions_exception_message"].astype(str) + " " +
    df["blame_frame_file"].astype(str) + " " +
    df["blame_frame_symbol"].astype(str)
)
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=150,
                        lowercase=True, token_pattern=r"[a-zA-Z]+")
X_texto = tfidf.fit_transform(texto)

colunas_categoricas = ["platform", "operating_system_name", "device_manufacturer",
                       "error_type", "process_state", "blame_frame_owner"]
onehot = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_categorico = onehot.fit_transform(df[colunas_categoricas])

colunas_numericas = ["blame_frame_line", "occurrences_count", "users_affected"]
X_numerico = csr_matrix(df[colunas_numericas].values.astype(float))

X = hstack([X_texto, X_categorico, X_numerico]).toarray()
X = np.clip(X, a_min=0, a_max=None)
y = df["severity"].values

print(f"Dataset: {X.shape[0]} crashes x {X.shape[1]} features")


# 2. CONFIGURA A VALIDAÇÃO CRUZADA
# StratifiedKFold garante que a proporcao das classes seja mantida
# em cada fold.

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

print(f"\nExecutando {k} folds...\n")


# 3. LOOP DOS 5 FOLDS
resultados_folds = []
matriz_acumulada = np.zeros((3, 3), dtype=int)
ordem_classes = ["CRÍTICO", "MÉDIO", "BAIXO"]

for fold_num, (idx_treino, idx_teste) in enumerate(skf.split(X, y), start=1):
    X_treino_fold = X[idx_treino]
    X_teste_fold = X[idx_teste]
    y_treino_fold = y[idx_treino]
    y_teste_fold = y[idx_teste]

    # SMOTE so no treino do fold
    menor_classe = min(pd.Series(y_treino_fold).value_counts())
    k_smote = min(5, menor_classe - 1)

    if k_smote >= 1:
        smote = SMOTE(k_neighbors=k_smote, random_state=42)
        X_bal, y_bal = smote.fit_resample(X_treino_fold, y_treino_fold)
    else:
        X_bal, y_bal = X_treino_fold, y_treino_fold

    # Ensemble
    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    nb = MultinomialNB(alpha=0.5)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("nb", nb), ("lr", lr)],
        voting="soft"
    )
    ensemble.fit(X_bal, y_bal)

    # Predicoes
    y_pred = ensemble.predict(X_teste_fold)
    y_proba = ensemble.predict_proba(X_teste_fold)

    # Metricas
    acc = accuracy_score(y_teste_fold, y_pred)
    f1_crit = f1_score(y_teste_fold, y_pred, labels=["CRÍTICO"],
                       average="macro", zero_division=0)

    try:
        y_teste_bin = label_binarize(y_teste_fold, classes=list(ensemble.classes_))
        auc = roc_auc_score(y_teste_bin, y_proba, average="macro", multi_class="ovr")
    except ValueError:
        auc = None

    # Acumula matriz de confusao
    matriz_fold = confusion_matrix(y_teste_fold, y_pred, labels=ordem_classes)
    matriz_acumulada += matriz_fold

    resultados_folds.append({
        "fold": fold_num,
        "acuracia": float(acc),
        "auc": float(auc) if auc else None,
        "f1_critico": float(f1_crit),
        "n_treino": len(y_treino_fold),
        "n_teste": len(y_teste_fold)
    })

    print(f"Fold {fold_num}: Acuracia={acc*100:5.1f}% | AUC={auc:.3f} | F1 Critico={f1_crit:.3f}")


# 4. MÉDIAS AGREGADAS
acuracias = [r["acuracia"] for r in resultados_folds]
aucs = [r["auc"] for r in resultados_folds if r["auc"] is not None]
f1s = [r["f1_critico"] for r in resultados_folds]

print("\n" + "=" * 65)
print("RESULTADOS AGREGADOS (media +/- desvio padrao)")
print("=" * 65)
print(f"   Acuracia media:      {np.mean(acuracias)*100:5.1f}% +/- {np.std(acuracias)*100:.1f}%")
print(f"   AUC medio:           {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
print(f"   F1 Critico medio:    {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")


# 5. MATRIZ DE CONFUSÃO CONSOLIDADA
print("\n" + "-" * 65)
print("MATRIZ DE CONFUSAO CONSOLIDADA (soma dos 5 folds)")
print("-" * 65)
df_matriz = pd.DataFrame(
    matriz_acumulada,
    index=[f"Real: {c}" for c in ordem_classes],
    columns=[f"Prev: {c}" for c in ordem_classes]
)
print(df_matriz.to_string())


# 6. VERIFICAÇÃO DOS CRITÉRIOS DE SUCESSO DO TAP
print("\n" + "-" * 65)
print("CRITERIOS DE SUCESSO DO TAP")
print("-" * 65)

auc_medio = float(np.mean(aucs))
concordancia = float(np.mean(acuracias))

if auc_medio >= 0.80:
    print(f"   OK  AUC >= 0,80 (obtido: {auc_medio:.3f})")
else:
    print(f"   XX  AUC abaixo de 0,80 (obtido: {auc_medio:.3f})")

if concordancia >= 0.75:
    print(f"   OK  Taxa de concordancia >= 75% (obtida: {concordancia*100:.1f}%)")
else:
    print(f"   XX  Taxa de concordancia abaixo de 75% (obtida: {concordancia*100:.1f}%)")


# 7. SALVA OS RESULTADOS EM JSON
resultados = {
    "k": k,
    "folds": resultados_folds,
    "medias": {
        "acuracia": float(np.mean(acuracias)),
        "acuracia_std": float(np.std(acuracias)),
        "auc": auc_medio,
        "auc_std": float(np.std(aucs)),
        "f1_critico": float(np.mean(f1s)),
        "f1_critico_std": float(np.std(f1s))
    },
    "matriz_confusao": matriz_acumulada.tolist(),
    "ordem_classes": ordem_classes,
    "criterios_tap": {
        "auc_meta": 0.80,
        "auc_atingido": auc_medio >= 0.80,
        "concordancia_meta": 0.75,
        "concordancia_atingido": concordancia >= 0.75
    }
}

with open("resultados_avaliacao.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f"\nResultados salvos em: resultados_avaliacao.json")
print("=" * 65)
