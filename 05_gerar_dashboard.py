# ═══════════════════════════════════════════════════════════════════
# SCRIPT 05 — DASHBOARD HTML DE CLASSIFICAÇÃO
# ═══════════════════════════════════════════════════════════════════
# Objetivo: Gerar um dashboard HTML visual mostrando os crashes
#           novos classificados pelo prototipo, com filtros, cards
#           e resumo executivo.
#
# Eh a demonstracao pratica do valor do prototipo:
# recebe crashes -> classifica -> prioriza automaticamente.
# ═══════════════════════════════════════════════════════════════════

import json


# 1. CARREGA OS DADOS
with open("classificacoes_crashes_novos.json", "r", encoding="utf-8") as f:
    classificacoes = json.load(f)

with open("resultados_avaliacao.json", "r", encoding="utf-8") as f:
    avaliacao = json.load(f)


# 2. ORDENA POR SEVERIDADE E CONFIANÇA
ordem_severidade = {"CRÍTICO": 0, "MÉDIO": 1, "BAIXO": 2}
classificacoes.sort(
    key=lambda x: (ordem_severidade[x["severidade_predita"]], -x["confianca"])
)


# 3. CONTAGEM POR CLASSE
qtd_critico = sum(1 for c in classificacoes if c["severidade_predita"] == "CRÍTICO")
qtd_medio = sum(1 for c in classificacoes if c["severidade_predita"] == "MÉDIO")
qtd_baixo = sum(1 for c in classificacoes if c["severidade_predita"] == "BAIXO")
total = len(classificacoes)


# 4. MONTA OS CARDS DE CRASHES
def cor_severidade(sev):
    return {
        "CRÍTICO": {"bg": "#FFEBEE", "border": "#C62828", "text": "#B71C1C", "emoji": "🔴"},
        "MÉDIO": {"bg": "#FFF3E0", "border": "#EF6C00", "text": "#E65100", "emoji": "🟡"},
        "BAIXO": {"bg": "#E8F5E9", "border": "#2E7D32", "text": "#1B5E20", "emoji": "🟢"}
    }[sev]


cards_html = ""
for i, c in enumerate(classificacoes, 1):
    cor = cor_severidade(c["severidade_predita"])
    conf_pct = c["confianca"] * 100

    # Barra de probabilidades
    barras_probs = ""
    for cls in ["CRÍTICO", "MÉDIO", "BAIXO"]:
        p = c["probabilidades"].get(cls, 0) * 100
        cor_barra = cor_severidade(cls)["border"]
        barras_probs += f"""
        <div class="prob-linha">
          <span class="prob-label">{cls}</span>
          <div class="prob-bar-bg">
            <div class="prob-bar" style="width: {p:.0f}%; background: {cor_barra};"></div>
          </div>
          <span class="prob-valor">{p:.0f}%</span>
        </div>
        """

    cards_html += f"""
    <div class="crash-card" data-severidade="{c['severidade_predita']}"
         style="background: {cor['bg']}; border-left: 5px solid {cor['border']};">
      <div class="crash-header">
        <div class="crash-numero">#{i:02d}</div>
        <div class="crash-badge" style="background: {cor['border']};">
          {cor['emoji']} {c['severidade_predita']}
        </div>
        <div class="crash-confianca">
          Confiança: <strong>{conf_pct:.1f}%</strong>
        </div>
      </div>
      <div class="crash-body">
        <div class="crash-info-row">
          <span class="crash-label">Exceção:</span>
          <span class="crash-value crash-code">{c['exceptions_type']}</span>
        </div>
        <div class="crash-info-row">
          <span class="crash-label">Mensagem:</span>
          <span class="crash-value">{c['exceptions_exception_message']}</span>
        </div>
        <div class="crash-info-row">
          <span class="crash-label">Local:</span>
          <span class="crash-value crash-code">{c['blame_frame_file']}:{c['blame_frame_line']}</span>
        </div>
        <div class="crash-info-row">
          <span class="crash-label">Método:</span>
          <span class="crash-value crash-code">{c['blame_frame_symbol']}</span>
        </div>
        <div class="crash-grid">
          <div class="crash-metric">
            <div class="crash-metric-label">Plataforma</div>
            <div class="crash-metric-value">{c['platform']}</div>
          </div>
          <div class="crash-metric">
            <div class="crash-metric-label">Dispositivo</div>
            <div class="crash-metric-value">{c['device_model']}</div>
          </div>
          <div class="crash-metric">
            <div class="crash-metric-label">Tipo</div>
            <div class="crash-metric-value">{c['error_type']}</div>
          </div>
          <div class="crash-metric">
            <div class="crash-metric-label">Ocorrências</div>
            <div class="crash-metric-value">{c['occurrences_count']:,}</div>
          </div>
          <div class="crash-metric">
            <div class="crash-metric-label">Usuários</div>
            <div class="crash-metric-value">{c['users_affected']:,}</div>
          </div>
          <div class="crash-metric">
            <div class="crash-metric-label">Event ID</div>
            <div class="crash-metric-value crash-code">{c['event_id']}</div>
          </div>
        </div>
        <div class="probs-section">
          <div class="probs-title">Distribuição das probabilidades:</div>
          {barras_probs}
        </div>
      </div>
    </div>
    """


# 5. RESULTADOS DA AVALIAÇÃO
medias = avaliacao["medias"]
auc = medias["auc"]
acuracia = medias["acuracia"]
f1_critico = medias["f1_critico"]


# 6. HTML COMPLETO
html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Protótipo — Classificação Automática de Crashes | IC PICTA 2026</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    min-height: 100vh; padding: 30px 20px; color: #333;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; }}

  header {{
    background: white; border-radius: 16px; padding: 30px 40px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 25px;
  }}
  header h1 {{ color: #1F3864; font-size: 26px; margin-bottom: 8px; }}
  .subtitulo {{ color: #555; font-size: 14px; margin-bottom: 4px; }}
  .autora {{ color: #2a5298; font-size: 13px; font-weight: 600; margin-top: 8px; }}

  .badges-header {{ display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; }}
  .badge {{
    background: #F0F4F8; padding: 6px 14px; border-radius: 20px;
    font-size: 12px; color: #1F3864; font-weight: 600;
  }}
  .badge.success {{ background: #E8F5E9; color: #1B5E20; }}

  .stats-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;
    margin-bottom: 25px;
  }}
  .stat-card {{
    background: white; border-radius: 12px; padding: 20px;
    text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  }}
  .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
  .stat-value {{ font-size: 36px; font-weight: 700; margin: 8px 0; }}
  .stat-total {{ color: #1F3864; }}
  .stat-critico {{ color: #C62828; }}
  .stat-medio {{ color: #EF6C00; }}
  .stat-baixo {{ color: #2E7D32; }}
  .stat-emoji {{ font-size: 24px; }}

  .model-info {{
    background: white; border-radius: 12px; padding: 25px 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 25px;
  }}
  .model-info h3 {{ color: #1F3864; margin-bottom: 15px; font-size: 18px; }}
  .metrics-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
  .metric-item {{ text-align: center; padding: 15px; background: #F8F9FC; border-radius: 8px; }}
  .metric-item .label {{ font-size: 12px; color: #666; margin-bottom: 5px; }}
  .metric-item .value {{ font-size: 24px; font-weight: 700; color: #1F3864; }}

  .filters {{
    display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap;
  }}
  .filter-btn {{
    background: white; border: none; padding: 10px 20px; border-radius: 8px;
    cursor: pointer; font-size: 14px; font-weight: 600;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: all 0.2s;
  }}
  .filter-btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
  .filter-btn.active {{ background: #1F3864; color: white; }}

  .crash-card {{
    background: white; border-radius: 12px; padding: 20px 25px;
    margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    transition: all 0.2s;
  }}
  .crash-card:hover {{ transform: translateX(4px); box-shadow: 0 6px 20px rgba(0,0,0,0.12); }}
  .crash-card.hidden {{ display: none; }}

  .crash-header {{
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 15px; padding-bottom: 12px; border-bottom: 1px solid #eee;
  }}
  .crash-numero {{ font-size: 20px; font-weight: 700; color: #1F3864; }}
  .crash-badge {{ color: white; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; }}
  .crash-confianca {{ font-size: 13px; color: #555; }}

  .crash-body {{ font-size: 14px; }}
  .crash-info-row {{ display: flex; margin-bottom: 8px; }}
  .crash-label {{ min-width: 100px; font-weight: 600; color: #555; font-size: 13px; }}
  .crash-value {{ flex: 1; color: #222; word-break: break-word; }}
  .crash-code {{ font-family: 'Menlo', monospace; font-size: 12px; background: #F5F5F5; padding: 2px 8px; border-radius: 4px; }}

  .crash-grid {{
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;
    margin-top: 15px; padding: 12px; background: rgba(255,255,255,0.5); border-radius: 8px;
  }}
  .crash-metric {{ text-align: center; }}
  .crash-metric-label {{ font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
  .crash-metric-value {{ font-size: 14px; font-weight: 600; color: #222; margin-top: 3px; }}

  .probs-section {{ margin-top: 15px; padding-top: 15px; border-top: 1px dashed #ccc; }}
  .probs-title {{ font-size: 12px; color: #666; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .prob-linha {{ display: flex; align-items: center; margin-bottom: 4px; }}
  .prob-label {{ min-width: 70px; font-size: 12px; font-weight: 600; color: #555; }}
  .prob-bar-bg {{ flex: 1; height: 14px; background: #eee; border-radius: 7px; overflow: hidden; margin: 0 10px; }}
  .prob-bar {{ height: 100%; border-radius: 7px; transition: width 0.3s; }}
  .prob-valor {{ min-width: 40px; text-align: right; font-size: 12px; font-weight: 600; color: #222; }}

  footer {{ text-align: center; color: white; margin-top: 30px; font-size: 13px; opacity: 0.9; }}

  @media (max-width: 900px) {{
    .stats-grid, .metrics-row {{ grid-template-columns: 1fr 1fr; }}
    .crash-grid {{ grid-template-columns: 1fr 1fr 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>Classificação Automática de Crashes</h1>
  </header>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Total de Crashes</div>
      <div class="stat-value stat-total">{total}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Críticos</div>
      <div class="stat-value stat-critico">{qtd_critico}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Médios</div>
      <div class="stat-value stat-medio">{qtd_medio}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Baixos</div>
      <div class="stat-value stat-baixo">{qtd_baixo}</div>
    </div>
  </div>

  <div class="model-info">
    <h3>Desempenho do Modelo</h3>
    <div class="metrics-row">
      <div class="metric-item">
        <div class="label">AUC Médio</div>
        <div class="value">{auc:.3f}</div>
      </div>
      <div class="metric-item">
        <div class="label">Taxa de Concordância</div>
        <div class="value">{acuracia*100:.1f}%</div>
      </div>
      <div class="metric-item">
        <div class="label">F1-Score Classe Crítico</div>
        <div class="value">{f1_critico:.3f}</div>
      </div>
    </div>
  </div>

  <div class="filters">
    <button class="filter-btn active" onclick="filtrar('TODOS', this)">Todos ({total})</button>
    <button class="filter-btn" onclick="filtrar('CRÍTICO', this)">🔴 Críticos ({qtd_critico})</button>
    <button class="filter-btn" onclick="filtrar('MÉDIO', this)">🟡 Médios ({qtd_medio})</button>
    <button class="filter-btn" onclick="filtrar('BAIXO', this)">🟢 Baixos ({qtd_baixo})</button>
  </div>

  <div id="crashes-container">
    {cards_html}
  </div>

</div>

<script>
  function filtrar(sev, btn) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    document.querySelectorAll('.crash-card').forEach(card => {{
      if (sev === 'TODOS' || card.dataset.severidade === sev) {{
        card.classList.remove('hidden');
      }} else {{
        card.classList.add('hidden');
      }}
    }});
  }}
</script>
</body>
</html>
"""


# 7. SALVA O ARQUIVO HTML
with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print("=" * 65)
print("DASHBOARD GERADO COM SUCESSO")
print("=" * 65)
print("\nArquivo: dashboard.html")
print("\nPara abrir no navegador:")
print("   Mac:     open dashboard.html")
print("   Linux:   xdg-open dashboard.html")
print("   Windows: start dashboard.html")
print("=" * 65)
