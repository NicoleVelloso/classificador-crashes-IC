import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


ARQUIVO = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026_v2.xlsx"
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
print("DIAGNOSTICO - DE ONDE VEM O DESEMPENHO DO MODELO")
print("=" * 70)

df = pd.read_excel(ARQUIVO, sheet_name="Crashes Treino")
df["texto"] = df[COLUNAS_TEXTO].astype(str).apply(" ".join, axis=1)
y = df["severity"].values



print("\n" + "-" * 70)
print("PARTE 1 - FAIXA DE VALORES DE CADA COLUNA NUMERICA, POR CLASSE")
print("-" * 70)

for coluna in COLUNAS_NUMERICAS:
    print(f"\n{coluna}:")
    faixas = {}
    for classe in ["CRÍTICO", "MÉDIO", "BAIXO"]:
        valores = df.loc[df["severity"] == classe, coluna]
        faixas[classe] = (valores.min(), valores.max())
        print(
            f"   {classe:<10} min={valores.min():>8.0f}  "
            f"max={valores.max():>8.0f}  mediana={valores.median():>8.0f}"
        )

    sobrepoe = False
    classes = list(faixas)
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a_min, a_max = faixas[classes[i]]
            b_min, b_max = faixas[classes[j]]
            if a_min <= b_max and b_min <= a_max:
                sobrepoe = True
    if sobrepoe:
        print("   -> as faixas se sobrepoem entre pelo menos duas classes")
    else:
        print("   -> ATENCAO: nenhuma sobreposicao. Esta coluna sozinha")
        print("      separa as tres classes por simples limiar.")



def construir_pipeline(usar_texto, usar_categorico, usar_numerico):
    transformers = []
    if usar_texto:
        transformers.append(
            (
                "texto",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=150,
                    lowercase=True,
                    token_pattern=r"[a-zA-Z]+",
                ),
                "texto",
            )
        )
    if usar_categorico:
        transformers.append(
            ("categorico", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS)
        )
    if usar_numerico:
        transformers.append(
            ("numerico", MinMaxScaler(clip=True), COLUNAS_NUMERICAS)
        )

    ensemble = VotingClassifier(
        estimators=[
            ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
            ("nb", MultinomialNB(alpha=0.5)),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ],
        voting="soft",
    )

    return ImbPipeline(
        [
            ("pre", ColumnTransformer(transformers)),
            ("smote", SMOTE(k_neighbors=5, random_state=SEED)),
            ("clf", ensemble),
        ]
    )


cenarios = {
    "Tudo (referencia)": (True, True, True),
    "So numericas": (False, False, True),
    "So users_affected + ocorrencias": "so_impacto",
    "So texto (TF-IDF)": (True, False, False),
    "So categoricas": (False, True, False),
    "Tudo MENOS numericas": (True, True, False),
}

f1_critico = make_scorer(
    f1_score, labels=["CRÍTICO"], average="macro", zero_division=0
)
METRICAS = {"acuracia": "accuracy", "auc": "roc_auc_ovr", "f1_critico": f1_critico}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)

print("\n" + "-" * 70)
print("PARTE 2 - DESEMPENHO POR GRUPO DE FEATURES")
print("-" * 70)
print("(mesmo ensemble, mesmo protocolo: 5 folds x 10 repeticoes)\n")
print(f"{'Features usadas':<34} {'Acuracia':>14} {'AUC':>16} {'F1 crit.':>10}")
print("-" * 70)

X_base = df[["texto"] + COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS]

for nome, config in cenarios.items():
    if config == "so_impacto":
        colunas_impacto = ["occurrences_count", "users_affected"]
        pipeline = ImbPipeline(
            [
                (
                    "pre",
                    ColumnTransformer(
                        [("numerico", MinMaxScaler(clip=True), colunas_impacto)]
                    ),
                ),
                ("smote", SMOTE(k_neighbors=5, random_state=SEED)),
                (
                    "clf",
                    VotingClassifier(
                        estimators=[
                            (
                                "svm",
                                SVC(
                                    kernel="rbf",
                                    C=1.0,
                                    probability=True,
                                    random_state=SEED,
                                ),
                            ),
                            ("nb", MultinomialNB(alpha=0.5)),
                            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
                        ],
                        voting="soft",
                    ),
                ),
            ]
        )
        X = df[colunas_impacto]
    else:
        pipeline = construir_pipeline(*config)
        X = X_base

    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=METRICAS, n_jobs=-1, error_score="raise"
    )
    acc = np.mean(scores["test_acuracia"])
    auc = np.mean(scores["test_auc"])
    f1c = np.mean(scores["test_f1_critico"])
    print(f"{nome:<34} {acc * 100:>11.1f}%  {auc:>13.3f}  {f1c:>10.3f}")


print("\n" + "-" * 70)
print("Impacto isolado em AUC ~0,500 = nao ha atalho para o rotulo.")
print("Texto isolado >= cenario completo = o sinal esta na descricao do crash.")
print("-" * 70)
