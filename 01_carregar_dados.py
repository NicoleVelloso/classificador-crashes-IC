import re
import sys

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


COLUNAS_ESTRUTURAIS = {
    "event_id", "installation_uuid", "crash_id", "issue_id", "session_id",
}

PADROES_IDENTIFICADOR = {
    "e-mail": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    "UUID": re.compile(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    ),
    "ID em hexadecimal longo": re.compile(r"\b[0-9a-fA-F]{32,}\b"),
}


def conferir_identificadores(df, nome_aba):
    """Procura padroes de identificador pessoal nas colunas de texto.

    Devolve uma lista de achados; lista vazia significa que nenhuma
    coluna de texto contem padrao de identificador.
    """
    achados = []
    colunas_texto = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and c not in COLUNAS_ESTRUTURAIS
    ]
    for coluna in colunas_texto:
        valores = df[coluna].astype(str)
        for nome_padrao, padrao in PADROES_IDENTIFICADOR.items():
            atingidas = valores.str.contains(padrao, regex=True, na=False)
            n = int(atingidas.sum())
            if n:
                exemplo = str(valores[atingidas].iloc[0])[:60]
                achados.append((nome_aba, coluna, nome_padrao, n, exemplo))
    return achados


print("\n" + "-" * 65)
print("PADROES DE IDENTIFICADOR NAS COLUNAS DE TEXTO:")
print("-" * 65)

achados_id = (
    conferir_identificadores(df_treino, ABA_TREINO)
    + conferir_identificadores(df_teste, ABA_TESTE)
)

if achados_id:
    print("   EXECUCAO INTERROMPIDA")
    print("   Foram encontrados padroes de identificador pessoal nas")
    print("   colunas de texto. Esta PoC so aceita dados sinteticos.")
    print()
    for aba, coluna, padrao, n, exemplo in achados_id:
        print(f"     [{aba}] {coluna}: {n} ocorrencia(s) de {padrao}")
        print(f"        exemplo: {exemplo}")
    print()
    print("   Remova ou anonimize esses valores antes de prosseguir.")
    sys.exit(1)

print("   OK Nenhum padrao de identificador nas colunas de texto.")
print(f"      Colunas estruturais de ID nao verificadas: "
      f"{', '.join(sorted(COLUNAS_ESTRUTURAIS & set(df_treino.columns))) or 'nenhuma'}")



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
