# ═══════════════════════════════════════════════════════════════════
# SCRIPT 04 — CLASSIFICAÇÃO DE CRASHES NOVOS
# ═══════════════════════════════════════════════════════════════════
# Objetivo: Usar o modelo treinado para classificar automaticamente
#           os 15 crashes novos (aba "Crashes Teste") - simulando o
#           uso real do prototipo em producao.
#
# Este script demonstra a proposta central da IC:
#   "Recebe crashes do Firebase -> classifica automaticamente"
# ═══════════════════════════════════════════════════════════════════

import json
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix


# 1. CARREGA MODELO E PRÉ-PROCESSADORES SALVOS
print("=" * 65)
print("CLASSIFICACAO AUTOMATICA DE CRASHES NOVOS")
print("=" * 65)

modelo = joblib.load("modelo.joblib")
tfidf = joblib.load("tfidf.joblib")
onehot = joblib.load("onehot.joblib")
colunas_categoricas = joblib.load("colunas_categoricas.joblib")
colunas_numericas = joblib.load("colunas_numericas.joblib")

print("OK - Modelo e pre-processadores carregados")


# 2. CARREGA OS CRASHES NOVOS
ARQUIVO = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026.xlsx"
df_novos = pd.read_excel(ARQUIVO, sheet_name="Crashes Teste (Novos)")

print(f"OK - {len(df_novos)} crashes novos carregados para classificacao")


# 3. APLICA O MESMO PRÉ-PROCESSAMENTO DO TREINO
# IMPORTANTE: usamos os pre-processadores TREINADOS (tfidf, onehot)
# e chamamos .transform() (nao .fit_transform()!). Isso garante que
# a transformacao eh identica a do treinamento.

print("\n[1/3] Aplicando pre-processamento aos crashes novos...")

# Texto: TF-IDF com bigramas
texto_novos = (
    df_novos["exceptions_type"].astype(str) + " " +
    df_novos["exceptions_exception_message"].astype(str) + " " +
    df_novos["blame_frame_file"].astype(str) + " " +
    df_novos["blame_frame_symbol"].astype(str)
)
X_texto_novos = tfidf.transform(texto_novos)

# Categoricas: one-hot
X_cat_novos = onehot.transform(df_novos[colunas_categoricas])

# Numericas
X_num_novos = csr_matrix(df_novos[colunas_numericas].values.astype(float))

# Junta tudo
X_novos = hstack([X_texto_novos, X_cat_novos, X_num_novos]).toarray()
X_novos = np.clip(X_novos, a_min=0, a_max=None)

print(f"   Features geradas: {X_novos.shape[1]}")


# 4. CLASSIFICA CADA CRASH NOVO
print("\n[2/3] Aplicando o modelo para classificar cada crash...")

predicoes = modelo.predict(X_novos)
probabilidades = modelo.predict_proba(X_novos)
classes_modelo = modelo.classes_

print(f"   OK - {len(predicoes)} crashes classificados")


# 5. MONTA O RELATÓRIO COMPLETO
print("\n[3/3] Gerando relatorio...")

relatorio = []
for i, (idx, crash) in enumerate(df_novos.iterrows()):
    classe = predicoes[i]

    # Probabilidade da classe atribuida (confianca do modelo)
    probs = probabilidades[i]
    prob_classe = float(probs[list(classes_modelo).index(classe)])

    # Todas as probabilidades para mostrar no relatorio
    todas_probs = {cls: float(p) for cls, p in zip(classes_modelo, probs)}

    relatorio.append({
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
        "confianca": prob_classe,
        "probabilidades": todas_probs
    })


# 6. EXIBE O RESULTADO NO TERMINAL
print("\n" + "=" * 65)
print("RESULTADO DA CLASSIFICACAO")
print("=" * 65)

# Ordenar por severidade e depois por confianca
ordem_severidade = {"CRÍTICO": 0, "MÉDIO": 1, "BAIXO": 2}
relatorio_ordenado = sorted(
    relatorio,
    key=lambda x: (ordem_severidade[x["severidade_predita"]], -x["confianca"])
)

for i, item in enumerate(relatorio_ordenado, 1):
    print(f"\n{i:2d}. {item['severidade_predita']:8s} (confianca: {item['confianca']*100:.1f}%)")
    print(f"    Excecao: {item['exceptions_type']}")
    print(f"    Local:   {item['blame_frame_file']}:{item['blame_frame_line']}")
    print(f"    Impacto: {item['users_affected']} usuarios | {item['occurrences_count']} ocorrencias")


# 7. RESUMO POR CLASSE
print("\n" + "-" * 65)
print("RESUMO")
print("-" * 65)

contagem = pd.Series([r["severidade_predita"] for r in relatorio]).value_counts()
for classe in ["CRÍTICO", "MÉDIO", "BAIXO"]:
    qtd = contagem.get(classe, 0)
    print(f"   {classe:8s}: {qtd:2d} crashes")


# 8. SALVA O RELATÓRIO EM JSON
with open("classificacoes_crashes_novos.json", "w", encoding="utf-8") as f:
    json.dump(relatorio, f, indent=2, ensure_ascii=False)

print(f"\nRelatorio salvo em: classificacoes_crashes_novos.json")
print("=" * 65)
