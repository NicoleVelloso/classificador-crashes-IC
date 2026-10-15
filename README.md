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
existe.
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
entram, e serve para verificar que nenhuma coluna isolada está funcionando como
atalho para o rótulo.
