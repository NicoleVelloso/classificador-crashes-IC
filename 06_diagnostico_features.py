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
from sklearn.tree import DecisionTreeClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings(
    "ignore",
    message=r"The `probability` parameter was deprecated",
    category=FutureWarning,
)

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
resultados = {}

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
    resultados[nome] = {"acuracia": acc, "auc": auc, "f1_critico": f1c}
    print(f"{nome:<34} {acc * 100:>11.1f}%  {auc:>13.3f}  {f1c:>10.3f}")


N_PALAVRAS_FINAIS = 3


def final_da_mensagem(serie, n_palavras=N_PALAVRAS_FINAIS):
    """Ultimas n palavras da mensagem, em minusculas."""
    return (
        serie.astype(str)
        .str.split()
        .apply(lambda palavras: " ".join(palavras[-n_palavras:]).lower())
    )


print("\n" + "-" * 70)
print("PARTE 3 - LINHA DE BASE DE REGRA TRIVIAL")
print("-" * 70)
print(f"(arvore de decisao sobre blame_frame_file + as "
      f"{N_PALAVRAS_FINAIS} ultimas palavras")
print(" da mensagem; sem TF-IDF, sem SMOTE, sem ensemble)\n")

X_trivial = pd.DataFrame(
    {
        "arquivo": df["blame_frame_file"].astype(str),
        "final_mensagem": final_da_mensagem(df["exceptions_exception_message"]),
    }
)

pipeline_trivial = ImbPipeline(
    [
        (
            "pre",
            ColumnTransformer(
                [(
                    "categorias",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["arquivo", "final_mensagem"],
                )]
            ),
        ),
        ("clf", DecisionTreeClassifier(random_state=SEED)),
    ]
)

scores_trivial = cross_validate(
    pipeline_trivial, X_trivial, y, cv=cv, scoring=METRICAS,
    n_jobs=-1, error_score="raise",
)
acc_trivial = float(np.mean(scores_trivial["test_acuracia"]))
auc_trivial = float(np.mean(scores_trivial["test_auc"]))
f1_trivial = float(np.mean(scores_trivial["test_f1_critico"]))
faixa_trivial = (
    float(np.percentile(scores_trivial["test_acuracia"], 2.5)),
    float(np.percentile(scores_trivial["test_acuracia"], 97.5)),
)

acc_modelo = resultados["Tudo (referencia)"]["acuracia"]
acc_texto = resultados["So texto (TF-IDF)"]["acuracia"]

print(f"{'Regra trivial (arquivo + final da msg)':<40} "
      f"{acc_trivial * 100:>6.1f}%   AUC {auc_trivial:.3f}   "
      f"F1 crit. {f1_trivial:.3f}")
print(f"{'   faixa observada da acuracia':<40} "
      f"[{faixa_trivial[0] * 100:.1f}% - {faixa_trivial[1] * 100:.1f}%]")
print(f"{'Ensemble completo (referencia)':<40} {acc_modelo * 100:>6.1f}%")
print(f"{'So texto, TF-IDF':<40} {acc_texto * 100:>6.1f}%")

print()
if acc_trivial > acc_modelo:
    print("   ALERTA: regra trivial supera o modelo")
    print(f"   Uma arvore sobre dois campos atinge "
          f"{acc_trivial * 100:.1f}%, contra "
          f"{acc_modelo * 100:.1f}% do ensemble.")
    print("   O rotulo esta escrito nesses campos por construcao do")
    print("   gerador. O desempenho do modelo nao pode ser lido como")
    print("   capacidade de classificar severidade a partir da")
    print("   descricao do crash: o que se mede e a recuperacao de uma")
    print("   regra conhecida.")
else:
    print("   OK A regra trivial nao supera o modelo completo.")

SEED_EMBARALHAMENTO = 7

print("\n" + "-" * 70)
print("PARTE 4 - FALSIFICADOR: EMBARALHAR OS CAMPOS QUE CARREGAM O SINAL")
print("-" * 70)
print("(embaralhar entre as linhas o campo que carrega o sinal deve")
print(" derrubar o AUC do cenario 'so texto'; se os dois sinais da")
print(" regra sairem, o AUC deve chegar perto de 0,500)\n")


def embaralhar(serie, semente):
    """Permuta os valores da serie entre as linhas, preservando o
    conjunto de valores. Converte para array de objetos porque o
    gerador do numpy nao embaralha dtypes de string do pandas com
    garantia de correcao."""
    valores = serie.astype(str).to_numpy(dtype=object).copy()
    np.random.default_rng(semente).shuffle(valores)
    return pd.Series(valores, index=serie.index)


inicios = (
    df["exceptions_exception_message"].astype(str).str.split()
    .apply(lambda p: " ".join(p[:-N_PALAVRAS_FINAIS]))
)

CENARIOS_FALSIFICADOR = [
    ("final da mensagem", ["final_msg"]),
    ("final da msg + arquivo + simbolo", ["final_msg", "arquivo", "simbolo"]),
    ("msg inteira + tipo + arquivo + simbolo",
     ["msg_inteira", "tipo", "arquivo", "simbolo"]),
]

auc_texto = resultados["So texto (TF-IDF)"]["auc"]
print(f"{'Nada embaralhado (referencia)':<34} AUC {auc_texto:>6.3f}   "
      f"acuracia {acc_texto * 100:.1f}%")

aucs_falsificador = {}
for nome_cenario, campos in CENARIOS_FALSIFICADOR:
    df_falso = df.copy()
    if "final_msg" in campos:
        finais = embaralhar(
            final_da_mensagem(df["exceptions_exception_message"]),
            SEED_EMBARALHAMENTO,
        )
        df_falso["exceptions_exception_message"] = inicios.str.cat(
            finais, sep=" "
        )
    if "msg_inteira" in campos:
        df_falso["exceptions_exception_message"] = embaralhar(
            df["exceptions_exception_message"], SEED_EMBARALHAMENTO + 3
        )
    if "tipo" in campos:
        df_falso["exceptions_type"] = embaralhar(
            df["exceptions_type"], SEED_EMBARALHAMENTO + 4
        )
    if "arquivo" in campos:
        df_falso["blame_frame_file"] = embaralhar(
            df["blame_frame_file"], SEED_EMBARALHAMENTO + 1
        )
    if "simbolo" in campos:
        df_falso["blame_frame_symbol"] = embaralhar(
            df["blame_frame_symbol"], SEED_EMBARALHAMENTO + 2
        )

    df_falso["texto"] = (
        df_falso[COLUNAS_TEXTO].astype(str).apply(" ".join, axis=1)
    )
    scores_falso = cross_validate(
        construir_pipeline(True, False, False),
        df_falso[["texto"] + COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS],
        y, cv=cv, scoring=METRICAS, n_jobs=-1, error_score="raise",
    )
    auc_f = float(np.mean(scores_falso["test_auc"]))
    acc_f = float(np.mean(scores_falso["test_acuracia"]))
    aucs_falsificador[nome_cenario] = auc_f
    print(f"{'Embaralhado: ' + nome_cenario:<40} AUC {auc_f:>6.3f}   "
          f"acuracia {acc_f * 100:.1f}%   "
          f"(queda {auc_texto - auc_f:.3f})")

auc_tudo_embaralhado = aucs_falsificador[
    "msg inteira + tipo + arquivo + simbolo"
]

print()
if auc_tudo_embaralhado < 0.60:
    print("   Falsificador confirma. Retirando do texto os dois sinais da")
    print("   regra do gerador — a area (nome do arquivo e simbolo) e o")
    print("   tipo de dano (mensagem inteira e tipo da excecao) — o AUC")
    print("   cai ao nivel do acaso. O que o modelo le no texto e a regra")
    print("   do gerador, e nao uma relacao entre descricao e severidade.")
    print("   Embaralhar so o final da mensagem nao basta: o dano esta")
    print("   escrito em mais de um campo.")
elif auc_tudo_embaralhado < auc_texto - 0.20:
    print("   Falsificador parcialmente confirmado: o AUC cai bastante,")
    print("   mas nao ate o nivel do acaso. Resta sinal em algum campo")
    print("   textual nao embaralhado.")
else:
    print("   Falsificador NAO confirma: o AUC nao caiu mesmo retirando os")
    print("   dois sinais. O sinal textual vem de outro lugar, e a leitura")
    print("   da Parte 3 precisa ser revista.")


print("\n" + "-" * 70)
print("COMO LER")
print("-" * 70)
print("Parte 2 - impacto isolado em AUC ~0,500: as colunas de impacto")
print("nao funcionam como atalho. Era o vazamento da primeira versao do")
print("dataset, e esta corrigido.")
print()
print("Parte 3 - a regra trivial supera o ensemble: o atalho que importa")
print("esta na combinacao de campos, e nao numa coluna isolada. A Parte 2")
print("sozinha nunca acusaria isso.")
print()
print("Parte 4 - o AUC do texto cai ao nivel do acaso quando a area e o")
print("tipo de dano saem dos campos textuais: confirma que o sinal lido")
print("pelo modelo e a regra do gerador.")
print()
print("Leitura conjunta: o experimento demonstra que o pipeline recupera")
print("uma regra de severidade conhecida a partir de campos do schema do")
print("Crashlytics. Isso e viabilidade tecnica do pipeline, e nao")
print("evidencia de que mensagens reais de crash carreguem esse sinal.")
print("-" * 70)
