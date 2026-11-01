## O que é

Protótipo em Python que classifica automaticamente a severidade de crashes de
aplicativos mobile (crítico, médio ou baixo), a partir de dados no schema do
Firebase Crashlytics.

O modelo é um ensemble soft voting combinando SVM, Naive Bayes e Regressão
Logística, com TF-IDF (unigramas e bigramas) sobre os campos textuais, one-hot encoding
sobre os categóricos e SMOTE para balancear as classes.

Este repositório acompanha o artigo da pesquisa e existe para que os
resultados possam ser reproduzidos. Toda a avaliação roda com semente
aleatória fixa.

## Como rodar — passo a passo

### 1. Preparar o ambiente (só na primeira vez)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No Windows, o terceiro comando é `venv\Scripts\activate`.

### 2. Rodar os scripts na ordem

```bash
python3 01_carregar_dados.py
python3 02_treinar_modelo.py
python3 03_avaliar_modelo.py
python3 04_classificar_crashes_novos.py
python3 05_gerar_dashboard.py
```

### 3. Abrir o dashboard

```bash
open dashboard.html
```

No Linux é `xdg-open dashboard.html`; no Windows, `start dashboard.html`.

---

## Atalho: rodar tudo de uma vez

```bash
source venv/bin/activate
bash rodar_tudo.sh
```

Funciona no macOS e no Linux. No Windows, use o Git Bash (que acompanha o
Git for Windows) ou o WSL — no `cmd` e no PowerShell o comando `bash` não
existe. Rodar os cinco scripts um a um, como na seção anterior, funciona em
qualquer sistema e não depende de nada além do Python.

---

## Se o Terminal foi fechado

O ambiente virtual precisa ser reativado a cada sessão nova:

```bash
source venv/bin/activate
```

Sem isso o `python3` não encontra as bibliotecas.

---

## O que cada script faz

| Script | Função | Arquivos que gera |
|---|---|---|
| `00_gerar_dataset_v2.py` | Gera a planilha de crashes sintéticos | `..._v2.xlsx` |
| `01_carregar_dados.py` | Carrega e valida a integridade do dataset | — |
| `02_treinar_modelo.py` | Treina o modelo final e salva em disco | `modelo.joblib` |
| `03_avaliar_modelo.py` | Validação cruzada e comparação com baselines | `resultados_avaliacao.json` |
| `04_classificar_crashes_novos.py` | Classifica os crashes da aba de teste | `classificacoes_crashes_novos.json` |
| `05_gerar_dashboard.py` | Monta a visualização em HTML | `dashboard.html` |
| `06_diagnostico_features.py` | Mostra de onde vem o desempenho do modelo | — |

O script `00` não precisa ser executado: a planilha já vem gerada no pacote.
Rodar de novo produz um arquivo idêntico, porque a semente aleatória é fixa.

O script `06` é opcional. Ele roda o modelo variando quais grupos de features
entram, e serve para verificar que nenhuma coluna isolada está funcionando
como atalho para o rótulo. O atalho que importa não está em coluna isolada, e
sim na combinação de `blame_frame_file` com o final da mensagem: as Partes 3 e
4 do script medem esse caso.

---

## Dataset

A planilha tem duas abas:

**Crashes Treino** — 150 crashes com 23 colunas, incluindo a severidade
atribuída pelo gerador e a justificativa da classificação.

**Crashes Teste (Novos)** — 40 crashes com 21 colunas. Faltam justamente as
duas colunas de severidade. Prever a classe de cada um desses crashes é a
tarefa do modelo; o resultado sai no arquivo
`classificacoes_crashes_novos.json` e no dashboard. A planilha não é
reescrita pelos scripts: ela é apenas a entrada.

Distribuição no treino: 36 críticos (24%), 54 médios (36%), 60 baixos (40%).

---

## Observações metodológicas

**Sobre a validação.** O pré-processamento inteiro (TF-IDF, one-hot, SMOTE)
roda dentro de um `Pipeline` do imbalanced-learn, ajustado separadamente em
cada fold da validação cruzada. O conjunto de validação é apenas transformado,
nunca usado no ajuste. Uma versão anterior deste protótipo ajustava os
vetorizadores sobre o dataset inteiro antes da divisão, o que inflava as
métricas.

**Sobre o score do modelo.** O valor exibido no dashboard é a saída do soft
voting, não uma probabilidade de acerto. A calibração do modelo não foi
avaliada.

**Sobre os crashes de teste.** Os 40 crashes da segunda aba não têm rótulo de
referência. A classificação deles ilustra o funcionamento do protótipo, mas não
mede acurácia — essa medição está no script 03.

**Sobre o dataset.** Os dados são inteiramente sintéticos. Cada campo é
sorteado pelo script `00_gerar_dataset_v2.py` segundo regras explícitas no
código. Nenhum crash de produção foi exportado, consultado ou transformado
para compor o dataset: o que orientou a escrita das regras foi o conhecimento
sobre a *forma* dos dados — quais campos o schema tem, que exceções são comuns
em Android e iOS, que ordens de grandeza as contagens assumem. Como
consequência, não houve tratamento de dados pessoais e não houve anonimização,
porque não existiam identificadores reais a anonimizar. Os nomes de pacote e
de arquivo são fictícios e seguem o padrão `com.example`.

As colunas de impacto (`occurrences_count` e `users_affected`) são sorteadas
independentemente da severidade, para que não funcionem como substituto do
rótulo. O script `06` verifica isso.

**Sobre o que falta no dataset.** Só há exceções das camadas nativas: Java e
Kotlin no Android, Objective-C e Swift no iOS. Em aplicativos React Native,
parte relevante dos erros ocorre na camada JavaScript e chega ao Crashlytics
com formato diferente. Essa classe de erro não está representada.

---

## Avisos das bibliotecas

Rodando os scripts 02, 03, 04 e 06 com `python3 -W always`, sem filtro por
categoria, aparece um único aviso distinto:

| Aviso | Decisão |
|---|---|
| `FutureWarning` em `sklearn/svm/_base.py`: "The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`" | Aceito com justificativa. O soft voting precisa de `predict_proba` nos três classificadores, e trocar o SVC por `CalibratedClassifierCV` alteraria a saída do modelo e todos os números já relatados.

Os quatro scripts silenciam esse aviso por mensagem, não por categoria:
qualquer aviso diferente dele volta a aparecer na saída.
