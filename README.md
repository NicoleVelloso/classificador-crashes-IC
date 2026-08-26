
## O que é este protótipo

Protótipo em Python que classifica automaticamente a severidade de crashes de aplicativos mobile (Firebase Crashlytics) em três níveis: **CRÍTICO**, **MÉDIO** ou **BAIXO**.

## Pré-requisitos

- **Python 3.10 ou superior** (recomendado 3.11 ou 3.12)
- Terminal (Mac/Linux) ou PowerShell (Windows)
- Editor de código (recomendado: VS Code)

Para verificar se o Python está instalado, abra o Terminal e digite:

```bash
python3 --version
```

---

## Instalação

### 1. Criar a pasta do projeto e entrar nela

```bash
mkdir prototipo-ic
cd prototipo-ic
```

### 2. Copiar todos os arquivos deste protótipo para a pasta

Você precisa dos seguintes arquivos na pasta `prototipo-ic`:

- `01_carregar_dados.py`
- `02_treinar_modelo.py`
- `03_avaliar_modelo.py`
- `04_classificar_crashes_novos.py`
- `05_gerar_dashboard.py`
- `requirements.txt`
- `Crashes_Firebase_Schema_Real_IC_Nicole_PICTA2026.xlsx`

### 3. Criar o ambiente virtual (venv)

```bash
python3 -m venv venv
```

### 4. Ativar o ambiente virtual

**No Mac/Linux:**
```bash
source venv/bin/activate
```

**No Windows:**
```bash
venv\Scripts\activate
```

Após ativar, você verá `(venv)` no início da linha do terminal.

### 5. Instalar as bibliotecas necessárias

```bash
pip install -r requirements.txt
```

---

## Como executar

**Importante:** o ambiente virtual precisa estar ativo antes de rodar qualquer script.

### Ordem de execução (primeira vez ou após qualquer mudança no dataset):

```bash
python3 01_carregar_dados.py
python3 02_treinar_modelo.py
python3 03_avaliar_modelo.py
python3 04_classificar_crashes_novos.py
python3 05_gerar_dashboard.py
```

### Abrir o dashboard no navegador:

**Mac:**
```bash
open dashboard.html
```

**Linux:**
```bash
xdg-open dashboard.html
```

**Windows:**
```bash
start dashboard.html
```

---

## O que cada script faz

| Script | Função |
|---|---|
| `01_carregar_dados.py` | Lê o Excel do dataset e verifica integridade dos dados |
| `02_treinar_modelo.py` | Pré-processa, aplica SMOTE, treina o ensemble e salva o modelo |
| `03_avaliar_modelo.py` | Executa validação cruzada k=5 e calcula métricas (AUC, F1, matriz de confusão) |
| `04_classificar_crashes_novos.py` | Usa o modelo treinado para classificar novos crashes |
| `05_gerar_dashboard.py` | Gera um dashboard HTML visual com os resultados |

---

## Como usar com dados novos

Se quiser classificar crashes diferentes:

1. Abra o arquivo Excel
2. Vá para a aba **"Crashes Teste (Novos)"**
3. Adicione ou substitua as linhas com os novos crashes (mantendo o mesmo formato do schema Firebase)
4. Salve o arquivo
5. Rode novamente:

```bash
python3 04_classificar_crashes_novos.py
python3 05_gerar_dashboard.py
```

**Não precisa retreinar o modelo** — ele já está salvo em `modelo.joblib` após a primeira execução do script `02`.

---

## Resolução de problemas

### Erro: `command not found: python`
No Mac, o comando é `python3` (com o 3 no final).

### Erro: `ModuleNotFoundError: No module named 'pandas'`
O ambiente virtual não está ativo. Rode:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Aviso: `FutureWarning`
É apenas um aviso do scikit-learn sobre mudanças futuras de sintaxe. Não afeta o funcionamento — pode ignorar.

### Modelo não encontrado ao rodar script 04
Você não executou o script 02 antes. Rode primeiro:
```bash
python3 02_treinar_modelo.py
```
