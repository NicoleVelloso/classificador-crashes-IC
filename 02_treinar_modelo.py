import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


ARQUIVO = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026_v2.xlsx"
ABA_TREINO = "Crashes Treino"
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


def preparar(df):
    """Monta a coluna de texto combinada. Usada tambem no script 04."""
    copia = df.copy()
    copia["texto"] = (
        copia[COLUNAS_TEXTO].astype(str).apply(lambda linha: " ".join(linha), axis=1)
    )
    return copia[["texto"] + COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS]


print("=" * 70)
print("TREINAMENTO DO MODELO FINAL")
print("=" * 70)

df = pd.read_excel(ARQUIVO, sheet_name=ABA_TREINO)
X = preparar(df)
y = df["severity"].values

print(f"\nDataset: {len(df)} crashes rotulados")
for classe in ["CRÍTICO", "MÉDIO", "BAIXO"]:
    n = int((y == classe).sum())
    print(f"   {classe:<10} {n:>4} crashes ({n / len(y) * 100:.0f}%)")



print("\n" + "-" * 70)
print("MONTANDO O PIPELINE")
print("-" * 70)

preprocessador = ColumnTransformer(
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
        ("categorico", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS),
        ("numerico", MinMaxScaler(clip=True), COLUNAS_NUMERICAS),
    ]
)

ensemble = VotingClassifier(
    estimators=[
        ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
        ("nb", MultinomialNB(alpha=0.5)),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ],
    voting="soft",
)

pipeline = ImbPipeline(
    [
        ("pre", preprocessador),
        ("smote", SMOTE(k_neighbors=5, random_state=SEED)),
        ("clf", ensemble),
    ]
)

print("   [1] TF-IDF com unigramas e bigramas sobre os campos textuais")
print("   [2] One-hot encoding sobre os campos categoricos")
print("   [3] MinMaxScaler sobre as features numericas")
print("   [4] SMOTE para balancear as classes")
print("   [5] Ensemble soft voting (SVM + Naive Bayes + Reg. Logistica)")


print("\n" + "-" * 70)
print("TREINANDO")
print("-" * 70)

pipeline.fit(X, y)
print("   OK - modelo treinado")

distribuicao_pos_smote = pd.Series(
    pipeline.named_steps["smote"].fit_resample(
        pipeline.named_steps["pre"].transform(X), y
    )[1]
).value_counts()
print("\n   Distribuicao apos o SMOTE (dentro do pipeline):")
for classe, n in distribuicao_pos_smote.items():
    print(f"      {classe:<10} {n:>4}")

n_features = pipeline.named_steps["pre"].transform(X).shape[1]
print(f"\n   Matriz de features: {len(df)} linhas x {n_features} colunas")


joblib.dump(pipeline, "modelo.joblib")

print("\n" + "-" * 70)
print("SALVANDO")
print("-" * 70)
print("   modelo.joblib  (pipeline completo: pre-processamento + modelo)")

print("\n" + "=" * 70)
print("TREINAMENTO CONCLUIDO")
print("=" * 70)
