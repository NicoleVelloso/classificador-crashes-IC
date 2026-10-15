import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ARQUIVO = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026_v2.xlsx"
ORDEM_CLASSES = ["CRÍTICO", "MÉDIO", "BAIXO"]
SEED = 42

COLUNAS_TEXTO = [
    "exceptions_type",
    "exceptions_exception_message",
    "blame_frame_file",
    "blame_frame_symbol",
]
COLUNAS_CATEGORICAS = [
    "platform",
    "operating_system_name",
    "device_manufacturer",
    "error_type",
    "process_state",
    "blame_frame_owner",
]
COLUNAS_NUMERICAS = ["blame_frame_line", "occurrences_count", "users_affected"]


print("=" * 70)
print("AVALIACAO DO MODELO - VALIDACAO CRUZADA REPETIDA")
print("=" * 70)

df = pd.read_excel(ARQUIVO, sheet_name="Crashes Treino")

df["texto"] = (
    df[COLUNAS_TEXTO].astype(str).apply(lambda linha: " ".join(linha), axis=1)
)

X = df[["texto"] + COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS]
y = df["severity"].values

print(f"\nDataset: {len(df)} crashes rotulados")
print("Distribuicao das classes:")
for classe in ORDEM_CLASSES:
    n = int((y == classe).sum())
    print(f"   {classe:<10} {n:>3} crashes ({n / len(y) * 100:.0f}%)")

print("\nFeatures: variam por fold (TF-IDF ajustado dentro de cada um).")



def construir_preprocessador():
    return ColumnTransformer(
        transformers=[
            (
                "texto",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=150,
                    lowercase=True,
                    token_pattern=r"[a-zA-Z]+",
                ),
                "texto",
            ),
            (
                "categorico",
                OneHotEncoder(handle_unknown="ignore"),
                COLUNAS_CATEGORICAS,
            ),
            ("numerico", MinMaxScaler(clip=True), COLUNAS_NUMERICAS),
        ]
    )


def construir_ensemble():
    return VotingClassifier(
        estimators=[
            ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
            ("nb", MultinomialNB(alpha=0.5)),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ],
        voting="soft",
    )


def montar_pipeline(classificador, com_smote=True):
    etapas = [("pre", construir_preprocessador())]
    if com_smote:
        etapas.append(("smote", SMOTE(k_neighbors=5, random_state=SEED)))
    etapas.append(("clf", classificador))
    return ImbPipeline(etapas)




f1_critico = make_scorer(
    f1_score, labels=["CRÍTICO"], average="macro", zero_division=0
)

METRICAS = {
    "acuracia": "accuracy",
    "auc": "roc_auc_ovr",
    "f1_critico": f1_critico,
}



N_SPLITS = 5
N_REPEATS = 10

cv_repetida = RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
)

modelos = {
    "Classe majoritaria (baseline)": montar_pipeline(
        DummyClassifier(strategy="most_frequent"), com_smote=False
    ),
    "SVM isolado": montar_pipeline(
        SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)
    ),
    "Naive Bayes isolado": montar_pipeline(MultinomialNB(alpha=0.5)),
    "Regressao Logistica isolada": montar_pipeline(
        LogisticRegression(max_iter=1000, random_state=SEED)
    ),
    "Ensemble sem SMOTE": montar_pipeline(construir_ensemble(), com_smote=False),
    "Ensemble soft voting": montar_pipeline(construir_ensemble()),
}

print("\n" + "=" * 70)
print(f"EXECUTANDO {N_SPLITS} FOLDS x {N_REPEATS} REPETICOES = "
      f"{N_SPLITS * N_REPEATS} AVALIACOES POR MODELO")
print("=" * 70)
print("(pode levar alguns minutos)\n")

resultados_modelos = {}

for nome, pipeline in modelos.items():
    print(f"   Avaliando: {nome} ...", flush=True)
    scores = cross_validate(
        pipeline, X, y, cv=cv_repetida, scoring=METRICAS, n_jobs=-1,
        error_score="raise",
    )
    resultados_modelos[nome] = {
        metrica: np.asarray(scores[f"test_{metrica}"]) for metrica in METRICAS
    }




def resumir(valores):
    return {
        "media": float(np.mean(valores)),
        "desvio": float(np.std(valores)),
        "p2_5": float(np.percentile(valores, 2.5)),
        "p97_5": float(np.percentile(valores, 97.5)),
    }


print("\n" + "=" * 70)
print("COMPARACAO ENTRE MODELOS")
print("=" * 70)
print(f"{'Modelo':<32} {'Acuracia':>18} {'AUC':>18}")
print("-" * 70)

tabela = {}
for nome, metricas in resultados_modelos.items():
    acc = resumir(metricas["acuracia"])
    auc = resumir(metricas["auc"])
    f1c = resumir(metricas["f1_critico"])
    tabela[nome] = {"acuracia": acc, "auc": auc, "f1_critico": f1c}
    print(
        f"{nome:<32} "
        f"{acc['media'] * 100:>6.1f}% +/- {acc['desvio'] * 100:<4.1f} "
        f"{auc['media']:>8.3f} +/- {auc['desvio']:<5.3f}"
    )

print("\nF1-score da classe CRITICA e faixa observada (percentis 2,5-97,5):")
print("-" * 70)
for nome, m in tabela.items():
    print(
        f"{nome:<32} F1 = {m['f1_critico']['media']:.3f} "
        f"[{m['f1_critico']['p2_5']:.3f} - {m['f1_critico']['p97_5']:.3f}]"
    )

print("\nFaixa observada de AUC (percentis 2,5-97,5 das 50 avaliacoes):")
print("-" * 70)
for nome, m in tabela.items():
    print(
        f"{nome:<32} AUC = {m['auc']['media']:.3f} "
        f"[{m['auc']['p2_5']:.3f} - {m['auc']['p97_5']:.3f}]"
    )

print("\n" + "=" * 70)
print("MATRIZ DE CONFUSAO DO ENSEMBLE (uma passagem de 5 folds)")
print("=" * 70)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
matriz_acumulada = np.zeros((3, 3), dtype=int)
acertos = 0
folds_detalhe = []

pipeline_final = montar_pipeline(construir_ensemble())

for numero, (idx_treino, idx_teste) in enumerate(skf.split(X, y), start=1):
    X_treino = X.iloc[idx_treino]
    X_teste = X.iloc[idx_teste]
    y_treino = y[idx_treino]
    y_teste = y[idx_teste]

    pipeline_final.fit(X_treino, y_treino)
    y_pred = pipeline_final.predict(X_teste)
    y_proba = pipeline_final.predict_proba(X_teste)

    matriz_acumulada += confusion_matrix(y_teste, y_pred, labels=ORDEM_CLASSES)
    acertos += int((y_pred == y_teste).sum())

    acc_fold = float((y_pred == y_teste).mean())
    f1_fold = float(
        f1_score(y_teste, y_pred, labels=["CRÍTICO"], average="macro", zero_division=0)
    )
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

        classes_pipeline = list(pipeline_final.named_steps["clf"].classes_)
        y_bin = label_binarize(y_teste, classes=classes_pipeline)
        auc_fold = float(
            roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        )
    except ValueError:
        auc_fold = None

    folds_detalhe.append(
        {
            "fold": numero,
            "acuracia": acc_fold,
            "auc": auc_fold,
            "f1_critico": f1_fold,
            "n_treino": int(len(y_treino)),
            "n_teste": int(len(y_teste)),
        }
    )

df_matriz = pd.DataFrame(
    matriz_acumulada,
    index=[f"Real: {c}" for c in ORDEM_CLASSES],
    columns=[f"Prev: {c}" for c in ORDEM_CLASSES],
)
print(df_matriz.to_string())

concordancia = acertos / len(y)
print(
    f"\nConcordancia com os rotulos de referencia: "
    f"{acertos}/{len(y)} = {concordancia * 100:.1f}%"
)

try:
    from scipy import stats

    teste = stats.binomtest(acertos, len(y))
    ic_baixo, ic_alto = teste.proportion_ci(confidence_level=0.95, method="wilson")
    print(
        f"   IC 95% (Wilson): [{ic_baixo * 100:.1f}% - {ic_alto * 100:.1f}%]"
    )

    META_CONCORDANCIA = 0.75
    comparacao = stats.binomtest(acertos, len(y), p=META_CONCORDANCIA,
                                 alternative="less")
    print(
        f"   Teste binomial contra a meta de "
        f"{META_CONCORDANCIA * 100:.0f}%: p = {comparacao.pvalue:.3f}"
    )
    if comparacao.pvalue > 0.05:
        print("   p > 0,05: diferenca nao distinguivel do acaso amostral.")
    else:
        print("   O desempenho observado fica abaixo da meta de forma consistente.")

    ic_concordancia = [float(ic_baixo), float(ic_alto)]
    p_valor_concordancia = float(comparacao.pvalue)
except ImportError:
    print("   (scipy nao instalado: intervalo de confianca nao calculado)")
    ic_concordancia = None
    p_valor_concordancia = None
print("Medida contra rotulos de referencia da autora, nao contra triagem independente.")


ensemble = tabela["Ensemble soft voting"]
auc_medio = ensemble["auc"]["media"]
auc_inferior = ensemble["auc"]["p2_5"]

print("\n" + "=" * 70)
print("CRITERIOS DE SUCESSO DO TAP")
print("=" * 70)

marca = "OK " if auc_medio >= 0.80 else "XX "
print(f"   {marca} AUC >= 0,80 ....... obtido: {auc_medio:.3f}")
print(f"        faixa observada: [{auc_inferior:.3f} - "
      f"{ensemble['auc']['p97_5']:.3f}]")

marca = "OK " if concordancia >= 0.75 else "XX "
print(f"   {marca} Concordancia >= 75% ... obtida: {concordancia * 100:.1f}%")
if ic_concordancia is not None:
    print(
        f"        IC 95%: [{ic_concordancia[0] * 100:.1f}% - "
        f"{ic_concordancia[1] * 100:.1f}%]"
    )
    if ic_concordancia[0] <= 0.75 <= ic_concordancia[1] and concordancia < 0.75:
        print("        meta contida no IC 95%")

if auc_inferior < 0.80 <= auc_medio:
    print(f"\n   Nota: limite inferior da faixa de AUC abaixo de 0,80 (n={len(y)}).")


saida = {
    "k": N_SPLITS,
    "folds": folds_detalhe,
    "protocolo": {
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "n_avaliacoes": N_SPLITS * N_REPEATS,
        "seed": SEED,
        "vazamento_corrigido": True,
        "observacao": (
            "TF-IDF, one-hot e SMOTE ajustados dentro de cada fold, "
            "somente sobre o conjunto de treino daquele fold."
        ),
    },
    "modelos": tabela,
    "medias": {
        "acuracia": ensemble["acuracia"]["media"],
        "acuracia_std": ensemble["acuracia"]["desvio"],
        "auc": ensemble["auc"]["media"],
        "auc_std": ensemble["auc"]["desvio"],
        "f1_critico": ensemble["f1_critico"]["media"],
        "f1_critico_std": ensemble["f1_critico"]["desvio"],
    },
    "matriz_confusao_ensemble": matriz_acumulada.tolist(),
    "matriz_confusao": matriz_acumulada.tolist(),
    "ordem_classes": ORDEM_CLASSES,
    "concordancia_rotulos_referencia": concordancia,
    "criterios_tap": {
        "auc_meta": 0.80,
        "auc_obtido": auc_medio,
        "auc_atingido": bool(auc_medio >= 0.80),
        "concordancia_meta": 0.75,
        "concordancia_obtida": concordancia,
        "concordancia_ic95": ic_concordancia,
        "concordancia_p_valor_contra_meta": p_valor_concordancia,
        "concordancia_atingida": bool(concordancia >= 0.75),
        "concordancia_atingido": bool(concordancia >= 0.75),
    },
}

with open("resultados_avaliacao.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)

print("\nResultados salvos em: resultados_avaliacao.json")
print("=" * 70)
