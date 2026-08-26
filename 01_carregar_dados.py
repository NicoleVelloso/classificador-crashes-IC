# ═══════════════════════════════════════════════════════════════════
# SCRIPT 01 — CARREGAMENTO DO DATASET FIREBASE
# ═══════════════════════════════════════════════════════════════════
# Objetivo: Ler o Excel com os crashes no schema real do Firebase
#           Crashlytics e verificar se os dados foram carregados
#           corretamente. Duas abas: treino (rotulado) e teste (novo).
#
# Alinhamento:
# - TAP: dataset baseado em Firebase Crashlytics
# - Plano de Projeto: dados sintéticos para preservar LGPD
# - Revisão bibliográfica: features validadas para apps mobile
#   (JORAYEVA et al., 2022; GHADESI et al., 2023)
# ═══════════════════════════════════════════════════════════════════

import pandas as pd


# 1. CAMINHO DO ARQUIVO
ARQUIVO = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026.xlsx"


# 2. LEITURA DAS DUAS ABAS
df_treino = pd.read_excel(ARQUIVO, sheet_name="Crashes Treino")
df_teste = pd.read_excel(ARQUIVO, sheet_name="Crashes Teste (Novos)")


# 3. VERIFICAÇÃO — TREINO
print("=" * 65)
print("VERIFICAÇÃO DO DATASET — SCHEMA FIREBASE CRASHLYTICS")
print("=" * 65)

print(f"\nABA 1 - Crashes Treino (rotulados)")
print(f"   Linhas: {df_treino.shape[0]} crashes")
print(f"   Colunas: {df_treino.shape[1]}")

print(f"\n   Distribuicao das classes:")
for classe, qtd in df_treino["severity"].value_counts().items():
    pct = qtd / len(df_treino) * 100
    print(f"     - {classe:8s}: {qtd:2d} crashes ({pct:.0f}%)")


# 4. VERIFICAÇÃO — TESTE
print(f"\nABA 2 - Crashes Teste (novos, sem rotulo)")
print(f"   Linhas: {df_teste.shape[0]} crashes")
print(f"   Colunas: {df_teste.shape[1]}")


# 5. AMOSTRA DE UM CRASH REAL
print("\n" + "-" * 65)
print("EXEMPLO DE UM CRASH DE TREINO (linha 1):")
print("-" * 65)

primeiro_crash = df_treino.iloc[0]
for coluna in df_treino.columns:
    valor = str(primeiro_crash[coluna])
    if len(valor) > 60:
        valor = valor[:60] + "..."
    print(f"   {coluna:38s} -> {valor}")


# 6. VERIFICA VALORES AUSENTES
print("\n" + "-" * 65)
print("VALORES AUSENTES POR COLUNA (esperado: 0):")
print("-" * 65)

ausentes_treino = df_treino.isnull().sum()
ausentes_teste = df_teste.isnull().sum()

problemas = False
for coluna in df_treino.columns:
    if ausentes_treino[coluna] > 0:
        print(f"   AVISO Treino - {coluna}: {ausentes_treino[coluna]} ausentes")
        problemas = True

for coluna in df_teste.columns:
    if ausentes_teste[coluna] > 0:
        print(f"   AVISO Teste - {coluna}: {ausentes_teste[coluna]} ausentes")
        problemas = True

if not problemas:
    print("   OK Nenhum valor ausente. Dataset integro.")


# 7. RESUMO FINAL
print("\n" + "=" * 65)
print("Carregamento concluido")
print("=" * 65)
print(f"   Treino: {df_treino.shape[0]} crashes rotulados")
print(f"   Teste:  {df_teste.shape[0]} crashes novos para classificar")
print("=" * 65)
