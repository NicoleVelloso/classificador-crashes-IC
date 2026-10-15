
import json
import warnings

import joblib
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



ARQUIVO = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026_v2.xlsx"
ABA_TESTE = "Crashes Teste (Novos)"

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
    """Identica a do script 02, para garantir o mesmo formato."""
    copia = df.copy()
    copia["texto"] = (
        copia[COLUNAS_TEXTO].astype(str).apply(lambda linha: " ".join(linha), axis=1)
    )
    return copia[["texto"] + COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS]


print("=" * 70)
print("CLASSIFICACAO AUTOMATICA DE CRASHES NOVOS")
print("=" * 70)

pipeline = joblib.load("modelo.joblib")
print("\nOK - pipeline carregado (pre-processamento + modelo)")

df_novos = pd.read_excel(ARQUIVO, sheet_name=ABA_TESTE)
print(f"OK - {len(df_novos)} crashes novos carregados")



print("\n[1/2] Aplicando o modelo...")

X_novos = preparar(df_novos)
predicoes = pipeline.predict(X_novos)
scores = pipeline.predict_proba(X_novos)
classes_modelo = list(pipeline.named_steps["clf"].classes_)

print(f"   OK - {len(predicoes)} crashes classificados")


print("\n[2/2] Gerando relatorio...")

relatorio = []
for posicao, (_, crash) in enumerate(df_novos.iterrows()):
    classe = predicoes[posicao]
    linha_scores = scores[posicao]
    score_classe = float(linha_scores[classes_modelo.index(classe)])

    relatorio.append(
        {
            "event_id": str(crash["event_id"]),
            "issue_id": str(crash["issue_id"]),
            "exceptions_type": str(crash["exceptions_type"]),
            "exceptions_exception_message": str(crash["exceptions_exception_message"]),
            "blame_frame_file": str(crash["blame_frame_file"]),
            "blame_frame_line": int(crash["blame_frame_line"]),
            "blame_frame_symbol": str(crash["blame_frame_symbol"]),
            "platform": str(crash["platform"]),
            "device_model": str(crash["device_model"]),
            "error_type": str(crash["error_type"]),
            "occurrences_count": int(crash["occurrences_count"]),
            "users_affected": int(crash["users_affected"]),
            "severidade_predita": str(classe),
            "score_modelo": score_classe,
            "scores": {
                classe_nome: float(valor)
                for classe_nome, valor in zip(classes_modelo, linha_scores)
            },
        }
    )



print("\n" + "=" * 70)
print("RESULTADO DA CLASSIFICACAO")
print("=" * 70)

ordem = {"CRÍTICO": 0, "MÉDIO": 1, "BAIXO": 2}
ordenado = sorted(
    relatorio, key=lambda item: (ordem[item["severidade_predita"]], -item["score_modelo"])
)

for posicao, item in enumerate(ordenado, 1):
    print(
        f"\n{posicao:2d}. {item['severidade_predita']:8s} "
        f"(score do modelo: {item['score_modelo'] * 100:.1f}%)"
    )
    print(f"    Excecao: {item['exceptions_type']}")
    print(f"    Local:   {item['blame_frame_file']}:{item['blame_frame_line']}")
    print(
        f"    Impacto: {item['users_affected']} usuarios | "
        f"{item['occurrences_count']} ocorrencias"
    )

print("\n" + "-" * 70)
print("RESUMO")
print("-" * 70)

contagem = pd.Series([r["severidade_predita"] for r in relatorio]).value_counts()
for classe in ["CRÍTICO", "MÉDIO", "BAIXO"]:
    print(f"   {classe:8s}: {int(contagem.get(classe, 0)):3d} crashes")

print("\n   Score = saida do soft voting, nao probabilidade de acerto (calibracao nao avaliada).")
print("   Sem rotulo de referencia: nao mede acuracia. Medicao no script 03.")

with open("classificacoes_crashes_novos.json", "w", encoding="utf-8") as f:
    json.dump(relatorio, f, indent=2, ensure_ascii=False)

print("\nRelatorio salvo em: classificacoes_crashes_novos.json")
print("=" * 70)
