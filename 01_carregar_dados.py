

import pandas as pd

ARQUIVO = "Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026_v2.xlsx"
ABA_TREINO = "Crashes Treino"
ABA_TESTE = "Crashes Teste (Novos)"
ORDEM_CLASSES = ["BAIXO", "MÉDIO", "CRÍTICO"]

print("=" * 65)
print("VERIFICAÇÃO DO DATASET — SCHEMA FIREBASE CRASHLYTICS")
print("=" * 65)


df_treino = pd.read_excel(ARQUIVO, sheet_name=ABA_TREINO)

print(f"\nABA 1 - {ABA_TREINO} (rotulados)")
print(f"   Linhas: {len(df_treino)} crashes")
print(f"   Colunas: {df_treino.shape[1]}")

print("\n   Distribuicao das classes:")
for classe in ORDEM_CLASSES:
    n = int((df_treino["severity"] == classe).sum())
    print(f"     - {classe:<8}: {n:>3} crashes ({n / len(df_treino) * 100:.0f}%)")



df_teste = pd.read_excel(ARQUIVO, sheet_name=ABA_TESTE)

print(f"\nABA 2 - {ABA_TESTE} (sem rotulo)")
print(f"   Linhas: {len(df_teste)} crashes")
print(f"   Colunas: {df_teste.shape[1]}")



print("\n" + "-" * 65)
print("EXEMPLO DE UM CRASH DE TREINO (linha 1):")
print("-" * 65)

primeiro = df_treino.iloc[0]
for coluna in df_treino.columns:
    valor = str(primeiro[coluna])
    if len(valor) > 58:
        valor = valor[:55] + "..."
    print(f"   {coluna:<38} -> {valor}")


print("\n" + "-" * 65)
print("VALORES AUSENTES POR COLUNA (esperado: 0):")
print("-" * 65)

ausentes_treino = df_treino.isna().sum()
ausentes_teste = df_teste.isna().sum()
problemas = {
    **{f"[treino] {c}": int(n) for c, n in ausentes_treino.items() if n > 0},
    **{f"[teste] {c}": int(n) for c, n in ausentes_teste.items() if n > 0},
}

if problemas:
    for coluna, n in problemas.items():
        print(f"   ATENCAO {coluna}: {n} valores ausentes")
else:
    print("   OK Nenhum valor ausente. Dataset integro.")



print("\n" + "-" * 65)
print("COLUNAS DE IMPACTO - CONFERENCIA DE SOBREPOSICAO:")
print("-" * 65)

for coluna in ["occurrences_count", "users_affected"]:
    faixas = {}
    for classe in ORDEM_CLASSES:
        v = df_treino.loc[df_treino["severity"] == classe, coluna]
        faixas[classe] = (v.min(), v.max())

    pares = 0
    for i in range(len(ORDEM_CLASSES)):
        for j in range(i + 1, len(ORDEM_CLASSES)):
            a, b = faixas[ORDEM_CLASSES[i]], faixas[ORDEM_CLASSES[j]]
            if a[0] <= b[1] and b[0] <= a[1]:
                pares += 1

    if pares == 3:
        print(f"   OK {coluna}: as tres classes se sobrepoem")
    else:
        print(f"   ATENCAO {coluna}: separa classes sem sobreposicao")
        print("           esta coluna pode estar funcionando como proxy do rotulo")


print("\n" + "=" * 65)
print("Carregamento concluido")
print("=" * 65)
print(f"   Treino: {len(df_treino)} crashes rotulados")
print(f"   Teste:  {len(df_teste)} crashes novos para classificar")
print("=" * 65)
