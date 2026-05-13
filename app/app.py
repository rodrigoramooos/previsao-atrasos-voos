"""
=============================================================================
  FlightSense — Previsão de Cancelamentos de Voos EUA 2024
  Grupo 1 — Rodrigo Ramos (a2023137922) | Bruno Almeida (a2023143583)
  Coimbra Business School | ISCAC — Ciência de Dados para a Gestão 2025/2026
=============================================================================
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FlightSense · Previsão de Cancelamentos",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#090E1A;--bg-card:#0F1729;--bg-el:#162035;
  --border:rgba(21,101,255,.18);--border-glow:rgba(0,212,255,.35);
  --blue:#1565FF;--cyan:#00D4FF;--amber:#F59E0B;
  --red:#EF4444;--green:#10B981;
  --tx:#E8EDF8;--muted:#6B7FA3;--dim:#3D506B;
  --mono:'JetBrains Mono',monospace;--sans:'DM Sans',sans-serif
}
.stApp{background:var(--bg)!important;font-family:var(--sans)!important}
.stApp::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(21,101,255,.03) 1px,transparent 1px),
  linear-gradient(90deg,rgba(21,101,255,.03) 1px,transparent 1px);
  background-size:40px 40px}
[data-testid="stSidebar"]{background:#080C18!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{font-family:var(--sans)!important;color:var(--tx)!important}
.block-container{padding:1.5rem 2.5rem 3rem!important;max-width:1400px}
h1,h2,h3{font-family:var(--mono)!important}
[data-testid="stMetric"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;
  border-radius:10px!important;padding:1rem 1.2rem!important;position:relative;overflow:hidden;transition:border-color .2s}
[data-testid="stMetric"]:hover{border-color:var(--border-glow)!important}
[data-testid="stMetric"]::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--blue),var(--cyan))}
[data-testid="stMetricLabel"] p{font-size:.72rem!important;font-weight:500!important;
  color:var(--muted)!important;text-transform:uppercase;letter-spacing:.06em;font-family:var(--mono)!important}
[data-testid="stMetricValue"]{font-size:1.7rem!important;font-weight:700!important;
  color:var(--tx)!important;font-family:var(--mono)!important;line-height:1.2!important}
[data-testid="stMetricDelta"]{font-size:.78rem!important;font-family:var(--mono)!important}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:1.5rem 0!important}
.js-plotly-plot .plotly .main-svg{background:transparent!important}
[data-testid="stPlotlyChart"]>div{background:transparent!important}
[data-testid="stForm"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;
  border-radius:14px!important;padding:1.6rem 1.8rem!important}
[data-testid="stFormSubmitButton"]>button{
  background:linear-gradient(135deg,var(--blue) 0%,#0D4FCC 100%)!important;
  border:1px solid rgba(21,101,255,.5)!important;color:white!important;
  font-family:var(--mono)!important;font-weight:600!important;font-size:.85rem!important;
  letter-spacing:.04em!important;border-radius:8px!important;padding:.65rem 1.4rem!important;
  transition:all .2s!important;box-shadow:0 0 20px rgba(21,101,255,.25)!important}
[data-testid="stFormSubmitButton"]>button:hover{
  box-shadow:0 0 32px rgba(0,212,255,.4)!important;border-color:var(--cyan)!important;
  transform:translateY(-1px)!important}
[data-testid="stTabs"] [role="tab"]{font-family:var(--mono)!important;font-size:.8rem!important;
  font-weight:500!important;color:var(--muted)!important;letter-spacing:.04em}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--cyan)!important;border-bottom-color:var(--cyan)!important}
[data-testid="stExpander"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px!important}
[data-testid="stAlert"][kind="error"]{background:rgba(239,68,68,.08)!important;border:1px solid rgba(239,68,68,.3)!important;border-radius:10px!important}
[data-testid="stAlert"][kind="success"]{background:rgba(16,185,129,.08)!important;border:1px solid rgba(16,185,129,.3)!important;border-radius:10px!important}
[data-testid="stMultiSelect"] span[data-baseweb="tag"]{
  background:rgba(21,101,255,.2)!important;border:1px solid rgba(21,101,255,.35)!important;
  color:#93C5FD!important;font-family:var(--mono)!important;font-size:.72rem!important;border-radius:4px!important}
[data-testid="stRadio"] [aria-checked="true"]+label,[data-testid="stRadio"] [aria-checked="true"]~div label{
  color:var(--cyan)!important;font-weight:600!important}
#MainMenu,footer,header{visibility:hidden!important}
[data-testid="stDecoration"]{display:none!important}
.ph{border-left:3px solid var(--cyan);padding:.1rem 0 .1rem 1rem;margin-bottom:.3rem}
.ph h1{font-family:var(--mono)!important;font-size:1.55rem!important;font-weight:700!important;
  color:var(--tx)!important;letter-spacing:-.02em;margin:0!important}
.ps{font-size:.82rem;color:var(--muted);margin:.4rem 0 1.6rem 1.3rem;font-family:var(--sans)!important;letter-spacing:.01em}
.sl{font-family:var(--mono);font-size:.68rem;font-weight:700;color:var(--dim);letter-spacing:.12em;
  text-transform:uppercase;margin-bottom:.8rem;padding-bottom:.5rem;border-bottom:1px solid var(--border)}
.chip{display:inline-block;font-family:var(--mono);font-size:.7rem;font-weight:600;
  padding:3px 10px;border-radius:99px;letter-spacing:.05em;margin-right:6px;margin-bottom:4px}
.cb{background:rgba(21,101,255,.15);color:#93C5FD;border:1px solid rgba(21,101,255,.3)}
.cc{background:rgba(0,212,255,.10);color:#67E8F9;border:1px solid rgba(0,212,255,.25)}
.cg{background:rgba(16,185,129,.12);color:#6EE7B7;border:1px solid rgba(16,185,129,.28)}
.cr{background:rgba(239,68,68,.12);color:#FCA5A5;border:1px solid rgba(239,68,68,.28)}
.ca{background:rgba(245,158,11,.12);color:#FCD34D;border:1px solid rgba(245,158,11,.28)}
.mb{background:var(--bg-el);border:1px solid var(--border);border-radius:8px;
  padding:.6rem 1rem;font-family:var(--mono);font-size:.75rem;color:var(--muted);
  display:flex;align-items:center;gap:12px;margin-bottom:1.2rem}
.mb strong{color:var(--cyan)}
.prow{display:flex;justify-content:space-between;padding:7px 0;
  border-bottom:1px solid rgba(21,101,255,.08);font-size:.83rem}
.prow:last-child{border-bottom:none}
.pk{color:var(--muted);font-family:var(--mono);font-size:.75rem}
.pv{color:var(--tx);font-weight:500}
.sf{font-family:var(--mono);font-size:.68rem;color:var(--dim);
  line-height:1.7;border-top:1px solid var(--border);padding-top:.8rem;margin-top:.4rem}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
NOMES_MES = {1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
             7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"}
NOMES_DIA = {1:"Segunda-feira",2:"Terça-feira",3:"Quarta-feira",
             4:"Quinta-feira",5:"Sexta-feira",6:"Sábado",7:"Domingo"}
C_RED="#EF4444"; C_GREEN="#10B981"; C_BLUE="#1565FF"; C_CYAN="#00D4FF"; C_AMBER="#F59E0B"
BG_CARD="#0F1729"
PL = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="DM Sans",color="#6B7FA3",size=11),
          title_font=dict(family="JetBrains Mono",color="#E8EDF8",size=13),
          legend=dict(bgcolor="rgba(15,23,41,.8)",bordercolor="rgba(21,101,255,.2)",borderwidth=1),
          margin=dict(t=50,b=40,l=20,r=20))

# Métricas reais — cancelamentos (HistGradient Boosting otimizado)
TAXA_BASE    = 1.53   # taxa de cancelamento pós-limpeza
TAXA_JAN     = 3.73   # taxa de cancelamento em Janeiro
AUC_ROC      = 0.854
F1_SCORE     = 0.151
RECALL       = 0.266
THRESHOLD    = 0.806  # TunedThresholdClassifierCV otimizado por F1

# Métricas reais — atrasos (XGBoost otimizado)
TAXA_BASE_DEL = 8.52
AUC_ROC_DEL   = 0.718
F1_SCORE_DEL  = 0.289
RECALL_DEL    = 0.396
THRESHOLD_DEL = 0.613

# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO
# ─────────────────────────────────────────────────────────────────────────────
def _load_model_safe(path):
    try:
        m = joblib.load(path)
        if isinstance(m, str) or (not hasattr(m,"predict_proba") and not hasattr(m,"predict")):
            st.error(f"Ficheiro de modelo inválido: {path}"); st.stop()
        return m
    except FileNotFoundError:
        st.error(f"{path} não encontrado."); st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar {path}: {e}"); st.stop()

@st.cache_resource(show_spinner="A inicializar modelos...")
def carregar_modelos():
    m_canc = _load_model_safe("modelo_cancelamentos_voos.pkl")
    m_del  = _load_model_safe("modelo_atrasos_voos.pkl")
    try:
        with open("feature_names.json") as f:
            features = [x for x in json.load(f) if x != "is_delayed"]
    except FileNotFoundError:
        st.error("feature_names.json não encontrado."); st.stop()
    return m_canc, m_del, features

@st.cache_data(show_spinner="A carregar dados...")
def carregar_dados():
    try:
        df_temp = pd.read_csv("data/flight_data_processed.csv")
        if "origin" not in df_temp.columns:
            cols_orig = [c for c in df_temp.columns if c.startswith("origin_")]
            if cols_orig:
                df_temp["origin"] = df_temp[cols_orig].idxmax(axis=1).str.replace("origin_","",regex=False)
        return df_temp
    except FileNotFoundError:
        st.error("data/flight_data_processed.csv não encontrado."); st.stop()

modelo, modelo_del, feature_names = carregar_modelos()
df = carregar_dados()

# ─────────────────────────────────────────────────────────────────────────────
# AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────
def construir_entrada(mes, dia_mes, dia_semana, aeroporto, distancia):
    """Constrói o vetor de features para o modelo de cancelamentos."""
    e = pd.DataFrame(0, index=[0], columns=feature_names)
    # Features derivadas automaticamente dos inputs
    vals = {
        "month":          mes,
        "day_of_month":   dia_mes,
        "day_of_week":    dia_semana,
        "distance":       distancia,
        "is_long_flight": int(distancia > 1500),
        "is_short_flight":int(distancia < 300),
        "is_weekend":     int(dia_semana in {6, 7}),
    }
    for col, val in vals.items():
        if col in e.columns:
            e[col] = val
    ap = f"origin_{aeroporto}"
    if ap in e.columns:
        e[ap] = 1
    return e.astype("float32")

def cor_risco(p):
    return C_GREEN if p < .3 else (C_AMBER if p < .6 else C_RED)

def gauge_chart(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob*100, 1),
        number={"suffix":"%","font":{"size":40,"family":"JetBrains Mono","color":"#E8EDF8"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#3D506B","tickwidth":1,
                       "tickfont":{"family":"JetBrains Mono","size":9,"color":"#6B7FA3"}},
               "bar":{"color":cor_risco(prob),"thickness":0.22},
               "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
               "steps":[{"range":[0,30],"color":"rgba(16,185,129,.08)"},
                        {"range":[30,60],"color":"rgba(245,158,11,.08)"},
                        {"range":[60,100],"color":"rgba(239,68,68,.08)"}],
               "threshold":{"line":{"color":"#E8EDF8","width":2},"thickness":.7,"value":prob*100}},
    ))
    fig.update_layout(height=230,margin=dict(t=20,b=10,l=30,r=30),
                      **{k:v for k,v in PL.items() if k != "margin"})
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:\'JetBrains Mono\';font-size:1.1rem;font-weight:700;'
                'color:#E8EDF8;margin-bottom:2px">✈ Flight<span style="color:#00D4FF">Sense</span></div>'
                '<div style="font-size:.73rem;color:#3D506B;font-family:\'DM Sans\';'
                'letter-spacing:.02em;margin-bottom:1rem">ML · Previsão de Cancelamentos EUA 2024</div>',
                unsafe_allow_html=True)
    st.divider()
    pagina = st.radio("nav", options=["  Dashboard","  Previsão","  Sobre"],
                      label_visibility="collapsed")
    st.divider()
    if "Dashboard" in pagina:
        st.markdown('<div class="sl">Filtros</div>', unsafe_allow_html=True)
        meses_disp = sorted(df["month"].dropna().unique().astype(int))
        meses_sel  = st.multiselect("Meses", options=meses_disp, default=meses_disp,
                                    format_func=lambda m: NOMES_MES.get(m, str(m))[:3])
    else:
        meses_sel = None
    st.divider()
    st.markdown(f'<div class="sf">Grupo 1 · ISCAC 2025/2026<br>Rodrigo Ramos · Bruno Almeida<br>'
                f'<span style="color:#1565FF">HistGradient Boosting</span> · AUC {AUC_ROC}</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if "Dashboard" in pagina:
    st.markdown('<div class="ph"><h1>Dashboard de Cancelamentos</h1></div>'
                '<div class="ps">Análise exploratória · Dataset BTS 2024 · 1 041 151 voos comerciais EUA (Jan–Fev)</div>',
                unsafe_allow_html=True)

    df_f = df[df["month"].isin(meses_sel)] if meses_sel else df
    total = len(df_f)
    canc  = int(df_f["cancelled"].sum())
    taxa  = canc / total * 100 if total > 0 else 0
    ap_v  = df_f.groupby("origin")["cancelled"].agg(["sum","count"]).query("count>=50")
    top_ap   = (ap_v["sum"] / ap_v["count"]).idxmax() if not ap_v.empty else "N/A"
    top_taxa = (ap_v["sum"] / ap_v["count"]).max() * 100 if not ap_v.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Voos na amostra", f"{total:,}")
    k2.metric("Cancelados", f"{canc:,}", delta=f"{taxa:.2f}% da amostra")
    k3.metric("Taxa global", f"{taxa:.2f}%", delta=f"{taxa - TAXA_BASE:.2f}% vs. média")
    k4.metric("Aeroporto crítico", top_ap, delta=f"{top_taxa:.1f}% taxa")
    st.divider()

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown('<div class="sl">Taxa de cancelamento por mês</div>', unsafe_allow_html=True)
        tm = (df_f.groupby("month")["cancelled"].agg(["sum","count"])
              .assign(taxa=lambda x: x["sum"] / x["count"] * 100).reset_index())
        tm["mes"] = tm["month"].map(lambda m: NOMES_MES.get(int(m), str(m))[:3])
        fig_m = px.bar(tm, x="mes", y="taxa", color="taxa",
                       color_continuous_scale=[[0,"#162035"],[.4,"#1565FF"],[1,"#00D4FF"]],
                       text_auto=".2f")
        fig_m.update_traces(textposition="outside",
                            textfont=dict(family="JetBrains Mono", size=10, color="#E8EDF8"),
                            marker_line_width=0)
        fig_m.update_layout(**PL, coloraxis_showscale=False,
                            title="Taxa de cancelamento por mês",
                            xaxis=dict(title="", tickfont=dict(family="JetBrains Mono", size=10)),
                            yaxis=dict(title="Taxa (%)", gridcolor="rgba(21,101,255,.06)",
                                       tickfont=dict(family="JetBrains Mono", size=9)))
        st.plotly_chart(fig_m, use_container_width=True)

    with col_b:
        st.markdown('<div class="sl">Top 12 aeroportos — taxa de cancelamento</div>', unsafe_allow_html=True)
        t12 = (df_f.groupby("origin")["cancelled"].agg(["sum","count"]).query("count>=50")
               .assign(taxa=lambda x: x["sum"] / x["count"] * 100)
               .nlargest(12, "taxa").reset_index())
        fig_a = px.bar(t12, x="taxa", y="origin", orientation="h", color="taxa",
                       color_continuous_scale=[[0,"#162035"],[.5,C_AMBER],[1,C_RED]],
                       text_auto=".2f")
        fig_a.update_traces(textposition="outside",
                            textfont=dict(family="JetBrains Mono", size=9, color="#E8EDF8"),
                            marker_line_width=0)
        fig_a.update_layout(**PL, coloraxis_showscale=False,
                            title="Top 12 aeroportos — taxa de cancelamento",
                            yaxis=dict(categoryorder="total ascending",
                                       tickfont=dict(family="JetBrains Mono", size=10, color="#E8EDF8")),
                            xaxis=dict(title="Taxa (%)", gridcolor="rgba(21,101,255,.06)",
                                       tickfont=dict(family="JetBrains Mono", size=9)))
        st.plotly_chart(fig_a, use_container_width=True)

    col_c, col_d = st.columns(2, gap="large")
    with col_c:
        st.markdown('<div class="sl">Distribuição da variável alvo</div>', unsafe_allow_html=True)
        fig_p = go.Figure(go.Pie(
            values=[total - canc, canc], labels=["Operacional","Cancelado"], hole=.62,
            marker=dict(colors=[C_GREEN, C_RED], line=dict(color=BG_CARD, width=3)),
            textfont=dict(family="JetBrains Mono", size=11),
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
        fig_p.add_annotation(
            text=f"<b>{taxa:.1f}%</b><br><span style='font-size:11px'>cancelados</span>",
            x=.5, y=.5, showarrow=False,
            font=dict(family="JetBrains Mono", size=18, color="#E8EDF8"))
        fig_p.update_layout(**{**PL, "title":"Distribuição da variável alvo",
                               "showlegend":True,
                               "legend":dict(orientation="h", y=-.1, x=.2)})
        st.plotly_chart(fig_p, use_container_width=True)

    with col_d:
        st.markdown('<div class="sl">Padrão semanal de cancelamentos</div>', unsafe_allow_html=True)
        td = (df_f.groupby("day_of_week")["cancelled"].agg(["sum","count"])
              .assign(taxa=lambda x: x["sum"] / x["count"] * 100).reset_index())
        td["dia"] = td["day_of_week"].map(lambda d: NOMES_DIA.get(int(d), str(d))[:3])
        fig_d = go.Figure(go.Scatter(
            x=td["dia"], y=td["taxa"], mode="lines+markers",
            line=dict(color=C_CYAN, width=2.5, shape="spline"),
            marker=dict(size=8, color=C_CYAN, line=dict(color=BG_CARD, width=2)),
            fill="tozeroy", fillcolor="rgba(0,212,255,.06)",
            hovertemplate="<b>%{x}</b><br>Taxa: %{y:.2f}%<extra></extra>"))
        fig_d.update_layout(**PL, title="Padrão semanal de cancelamentos",
                            xaxis=dict(title="", tickfont=dict(family="JetBrains Mono", size=10)),
                            yaxis=dict(title="Taxa (%)", gridcolor="rgba(21,101,255,.06)",
                                       tickfont=dict(family="JetBrains Mono", size=9)))
        st.plotly_chart(fig_d, use_container_width=True)

    # Gráfico de dia do mês — feature mais importante segundo SHAP
    if "day_of_month" in df_f.columns:
        st.markdown('<div class="sl">Taxa de cancelamento por dia do mês · feature mais importante (SHAP)</div>',
                    unsafe_allow_html=True)
        tdom = (df_f.groupby("day_of_month")["cancelled"].agg(["sum","count"])
                .assign(taxa=lambda x: x["sum"] / x["count"] * 100).reset_index())
        media_dom = tdom["taxa"].mean()
        fig_dom = go.Figure()
        fig_dom.add_hline(y=media_dom, line_dash="dash", line_color="#3D506B",
                          annotation_text=f"Média {media_dom:.2f}%",
                          annotation_font=dict(family="JetBrains Mono", size=9, color="#6B7FA3"))
        fig_dom.add_trace(go.Bar(
            x=tdom["day_of_month"], y=tdom["taxa"],
            marker=dict(
                color=tdom["taxa"],
                colorscale=[[0,"#162035"],[0.4,"#1565FF"],[1,"#EF4444"]],
                line=dict(width=0)),
            hovertemplate="Dia %{x}: %{y:.2f}%<extra></extra>"))
        fig_dom.update_layout(**PL, title="Taxa de cancelamento por dia do mês",
                              xaxis=dict(title="Dia do mês", dtick=1,
                                         tickfont=dict(family="JetBrains Mono", size=9)),
                              yaxis=dict(title="Taxa (%)", gridcolor="rgba(21,101,255,.06)",
                                         tickfont=dict(family="JetBrains Mono", size=9)))
        st.plotly_chart(fig_dom, use_container_width=True)

    if "distance" in df_f.columns:
        st.markdown('<div class="sl">Distribuição de distância · cancelados vs operacionais</div>',
                    unsafe_allow_html=True)
        fig_dist = px.histogram(df_f, x="distance", color="cancelled", barmode="overlay",
                                nbins=60, opacity=.75,
                                color_discrete_map={0:C_GREEN, 1:C_RED},
                                labels={"distance":"Distância (milhas)","cancelled":""})
        fig_dist.update_traces(marker_line_width=0)
        fig_dist.update_layout(**PL, title="Distribuição de distância · cancelados vs operacionais",
                               xaxis=dict(gridcolor="rgba(21,101,255,.06)"),
                               yaxis=dict(gridcolor="rgba(21,101,255,.06)"))
        fig_dist.for_each_trace(lambda t: t.update(name="Cancelado" if t.name=="1" else "Operacional"))
        st.plotly_chart(fig_dist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREVISÃO
# ─────────────────────────────────────────────────────────────────────────────
elif "Previsão" in pagina:
    st.markdown('<div class="ph"><h1>Previsão de Perturbações</h1></div>'
                '<div class="ps">Insere os dados do voo · os modelos respondem em tempo real</div>',
                unsafe_allow_html=True)

    aba_canc, aba_del = st.tabs(["  Cancelamento", "  Atraso"])

    # ── Formulário partilhado ──────────────────────────────────────────────
    def formulario(form_key):
        with st.form(form_key, clear_on_submit=False):
            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                st.markdown('<div class="sl">Período temporal</div>', unsafe_allow_html=True)
                mes        = st.selectbox("Mês do voo", options=[1, 2],
                                          format_func=lambda m: NOMES_MES[m], key=f"mes_{form_key}")
                dia_mes    = st.slider("Dia do mês", 1, 31, 15, key=f"dm_{form_key}")
                dia_semana = st.selectbox("Dia da semana", options=list(NOMES_DIA.keys()),
                                          format_func=lambda d: NOMES_DIA[d], key=f"ds_{form_key}")
            with c2:
                st.markdown('<div class="sl">Dados do voo</div>', unsafe_allow_html=True)
                aeroportos = sorted(df["origin"].dropna().unique())
                aeroporto  = st.selectbox("Aeroporto de origem (IATA)", options=aeroportos,
                                          key=f"ap_{form_key}")
                distancia  = st.number_input("Distância (milhas)", min_value=50, max_value=5000,
                                             value=800, step=50, key=f"dist_{form_key}")
            with c3:
                st.markdown('<div class="sl">Classificação automática</div>', unsafe_allow_html=True)
                curto  = distancia < 300
                longo  = distancia > 1500
                medio  = not curto and not longo
                fds    = dia_semana in {6, 7}
                def dot(on): return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{"#00D4FF" if on else "#3D506B"};margin-right:6px"></span>'
                st.markdown(
                    f'<div style="background:#0F1729;border:1px solid rgba(21,101,255,.15);'
                    f'border-radius:8px;padding:.8rem 1rem;font-family:\'JetBrains Mono\';'
                    f'font-size:.75rem;color:#6B7FA3">'
                    f'<div style="margin-bottom:6px">{dot(curto)}'
                    f'<span style="color:{"#00D4FF" if curto else "#3D506B"}">Curta distância</span>'
                    f'<span style="float:right">&lt; 300 mi</span></div>'
                    f'<div style="margin-bottom:6px">{dot(medio)}'
                    f'<span style="color:{"#93C5FD" if medio else "#3D506B"}">Médio curso</span>'
                    f'<span style="float:right">300–1500 mi</span></div>'
                    f'<div style="margin-bottom:8px">{dot(longo)}'
                    f'<span style="color:{"#FCD34D" if longo else "#3D506B"}">Longa distância</span>'
                    f'<span style="float:right">&gt; 1500 mi</span></div>'
                    f'<div style="padding-top:8px;border-top:1px solid rgba(21,101,255,.1)">'
                    f'{dot(fds)}<span style="color:{"#E8EDF8" if fds else "#3D506B"}">'
                    f'{"Fim de semana" if fds else "Dia útil"}</span></div></div>',
                    unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("  Calcular probabilidade",
                                                  use_container_width=True, type="primary")
        return submitted, mes, dia_mes, dia_semana, aeroporto, distancia

    def mostrar_resultado(prob, pred, mes, dia_mes, dia_semana, aeroporto, distancia,
                          taxa_base, label_pos, label_neg, threshold, auc, f1, recall):
        st.divider()
        r1, r2 = st.columns([1, 1], gap="large")
        with r1:
            lbl  = label_pos if pred == 1 else label_neg
            clr  = C_RED if pred == 1 else C_GREEN
            desc = ("O modelo identificou padrões históricos associados a este evento. "
                    "Probabilidade acima do limiar de decisão."
                    if pred == 1 else
                    "Sem padrões de risco elevado identificados. "
                    "Existe sempre risco residual não capturado pelo modelo.")
            brd = "rgba(239,68,68,.4)" if pred == 1 else "rgba(16,185,129,.4)"
            shd = "rgba(239,68,68,.08)" if pred == 1 else "rgba(16,185,129,.08)"
            st.markdown(
                f'<div style="background:#0F1729;border:1px solid {brd};box-shadow:0 0 24px {shd};'
                f'border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1rem">'
                f'<div style="font-family:\'JetBrains Mono\';font-size:.7rem;color:{clr};'
                f'letter-spacing:.1em;margin-bottom:6px">{lbl}</div>'
                f'<div style="font-size:.85rem;color:#6B7FA3;line-height:1.55">{desc}</div></div>',
                unsafe_allow_html=True)

            st.markdown('<div class="sl">Parâmetros da previsão</div>', unsafe_allow_html=True)
            params = [
                ("Mês",           NOMES_MES[mes]),
                ("Dia do mês",    str(dia_mes)),
                ("Dia da semana", NOMES_DIA[dia_semana]),
                ("Fim de semana", "Sim" if dia_semana in {6,7} else "Não"),
                ("Aeroporto",     aeroporto),
                ("Distância",     f"{distancia:,} mi"),
                ("Voo curto",     "Sim" if distancia < 300 else "Não"),
                ("Voo longo",     "Sim" if distancia > 1500 else "Não"),
            ]
            rows = "".join(
                f'<div class="prow"><span class="pk">{k}</span><span class="pv">{v}</span></div>'
                for k, v in params)
            st.markdown(
                f'<div style="background:#0F1729;border:1px solid rgba(21,101,255,.15);'
                f'border-radius:10px;padding:.8rem 1.1rem">{rows}</div>',
                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Média global", f"{taxa_base}%")
            rc2.metric("Este voo",     f"{prob*100:.1f}%",
                       delta=f"{prob*100 - taxa_base:+.1f}% vs. global")
            rc3.metric("Threshold",    str(threshold))

        with r2:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)
            nivel = "BAIXO" if prob < .3 else ("MÉDIO" if prob < .6 else "ALTO")
            cn    = C_GREEN if prob < .3 else (C_AMBER if prob < .6 else C_RED)
            pct   = prob * 100
            bars  = [
                ("rgba(16,185,129,.7)" if pct < 30   else "rgba(16,185,129,.15)"),
                ("rgba(245,158,11,.7)" if 30<=pct<60 else "rgba(245,158,11,.15)"),
                ("rgba(239,68,68,.7)"  if pct >= 60  else "rgba(239,68,68,.15)"),
            ]
            st.markdown(
                f'<div style="background:#0F1729;border:1px solid rgba(21,101,255,.15);'
                f'border-radius:10px;padding:1rem 1.2rem;margin-top:-.5rem">'
                f'<div style="font-family:\'JetBrains Mono\';font-size:.68rem;color:#3D506B;'
                f'letter-spacing:.1em;margin-bottom:10px">NÍVEL DE RISCO</div>'
                f'<div style="display:flex;gap:6px;margin-bottom:10px">'
                + "".join(f'<div style="flex:1;height:6px;border-radius:99px;background:{b}"></div>'
                          for b in bars)
                + f'</div><span style="font-family:\'JetBrains Mono\';font-size:1.1rem;'
                f'font-weight:700;color:{cn}">{nivel}</span>'
                f'<span style="font-size:.8rem;color:#6B7FA3;margin-left:8px">'
                f'{pct:.1f}% probabilidade</span></div>',
                unsafe_allow_html=True)

        with st.expander("  Como interpretar este resultado?"):
            st.markdown(f"""
**O que significa {prob*100:.1f}%?**
Em voos com características semelhantes — {NOMES_MES[mes]}, {NOMES_DIA[dia_semana]},
partindo de {aeroporto} — o modelo estimou uma probabilidade de **{prob*100:.1f}%**.

**O threshold é {threshold}**, otimizado por validação cruzada (5 folds) para maximizar o F1-score.
Acima deste valor, o voo é classificado em risco. Esta escolha minimiza os falsos negativos —
casos em que um evento real passa despercebido.

**Desempenho do modelo:** AUC-ROC {auc} · F1 {f1} · Recall {recall}

**Limitações:** o modelo não tem acesso a dados meteorológicos em tempo real, estado da aeronave
ou decisões operacionais das companhias aéreas — os principais determinantes reais de cancelamentos e atrasos.
            """)

    # ── Aba Cancelamento ──────────────────────────────────────────────────
    with aba_canc:
        st.markdown(
            f'<div class="mb"><span>Modelo</span><strong>HistGradient Boosting</strong>'
            f'<span style="margin-left:auto">AUC-ROC <strong style="color:#00D4FF">{AUC_ROC}</strong></span>'
            f'<span>F1 <strong style="color:#F59E0B">{F1_SCORE}</strong></span>'
            f'<span>Recall <strong style="color:#E8EDF8">{RECALL}</strong></span>'
            f'<span>Threshold <strong style="color:#E8EDF8">{THRESHOLD}</strong></span></div>',
            unsafe_allow_html=True)
        sub_c, mes_c, dm_c, ds_c, ap_c, dist_c = formulario("canc")
        if sub_c:
            entrada = construir_entrada(mes_c, dm_c, ds_c, ap_c, dist_c)
            prob    = float(modelo.predict_proba(entrada)[0][1])
            pred    = int(modelo.predict(entrada)[0])
            mostrar_resultado(prob, pred, mes_c, dm_c, ds_c, ap_c, dist_c,
                              TAXA_BASE,
                              "RISCO DE CANCELAMENTO ELEVADO",
                              "VOO PROVAVELMENTE OPERACIONAL",
                              THRESHOLD, AUC_ROC, F1_SCORE, RECALL)

    # ── Aba Atraso ────────────────────────────────────────────────────────
    with aba_del:
        st.markdown(
            f'<div class="mb"><span>Modelo</span><strong>XGBoost</strong>'
            f'<span style="margin-left:auto">AUC-ROC <strong style="color:#00D4FF">{AUC_ROC_DEL}</strong></span>'
            f'<span>F1 <strong style="color:#F59E0B">{F1_SCORE_DEL}</strong></span>'
            f'<span>Recall <strong style="color:#E8EDF8">{RECALL_DEL}</strong></span>'
            f'<span>Threshold <strong style="color:#E8EDF8">{THRESHOLD_DEL}</strong></span></div>',
            unsafe_allow_html=True)
        st.info("Atraso definido como: atraso meteorológico ou por aeronave anterior >= 15 minutos, "
                "excluindo voos cancelados.", icon=None)
        sub_d, mes_d, dm_d, ds_d, ap_d, dist_d = formulario("del")
        if sub_d:
            entrada = construir_entrada(mes_d, dm_d, ds_d, ap_d, dist_d)
            prob    = float(modelo_del.predict_proba(entrada)[0][1])
            pred    = int(modelo_del.predict(entrada)[0])
            mostrar_resultado(prob, pred, mes_d, dm_d, ds_d, ap_d, dist_d,
                              TAXA_BASE_DEL,
                              "RISCO DE ATRASO ELEVADO",
                              "VOO PROVAVELMENTE PONTUAL",
                              THRESHOLD_DEL, AUC_ROC_DEL, F1_SCORE_DEL, RECALL_DEL)

# ─────────────────────────────────────────────────────────────────────────────
# SOBRE
# ─────────────────────────────────────────────────────────────────────────────
elif "Sobre" in pagina:
    st.markdown('<div class="ph"><h1>Sobre o Projeto</h1></div>'
                '<div class="ps">FlightSense · Análise e Previsão de Cancelamentos EUA 2024</div>',
                unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom:1.5rem">'
                '<span class="chip cb">ISCAC · CBS</span>'
                '<span class="chip cc">Ciência de Dados para a Gestão</span>'
                '<span class="chip ca">2025/2026</span>'
                '<span class="chip cg">Projeto em CD</span></div>', unsafe_allow_html=True)

    cr, cb = st.columns(2, gap="large")
    def member_card(nome, num, role):
        return (f'<div style="background:#0F1729;border:1px solid rgba(21,101,255,.18);'
                f'border-radius:12px;padding:1.2rem 1.4rem">'
                f'<div style="font-family:\'JetBrains Mono\';font-size:.7rem;color:#3D506B;'
                f'letter-spacing:.1em;margin-bottom:6px">INVESTIGADOR</div>'
                f'<div style="font-size:1rem;font-weight:600;color:#E8EDF8;margin-bottom:2px">{nome}</div>'
                f'<div style="font-family:\'JetBrains Mono\';font-size:.75rem;color:#1565FF">{num}</div>'
                f'<div style="font-size:.8rem;color:#6B7FA3;margin-top:6px">{role}</div></div>')
    with cr:
        st.markdown(member_card("Rodrigo Ramos","a2023137922",
                                "Engenharia de Dados · Visualização · Notebooks Kaggle"),
                    unsafe_allow_html=True)
    with cb:
        st.markdown(member_card("Bruno Almeida","a2023143583",
                                "Documentação técnica · Modelação estatística · SHAP"),
                    unsafe_allow_html=True)

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registos (pós-limpeza)", "1 041 151")
    k2.metric("Features no modelo",     "340")
    k3.metric("Modelo final",           "HistGB Otimizado")
    k4.metric("AUC-ROC (teste)",        str(AUC_ROC))
    st.divider()

    t1, t2, t3, t4 = st.tabs(["📂  Dataset","⚙️  Metodologia","📈  Resultados","📑  Referências"])

    with t1:
        st.markdown("""
**Fonte:** [Kaggle · Flight Delay and Cancellation Data 2024](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024)
**Origem:** Bureau of Transportation Statistics (BTS) — U.S. DOT
**Período:** Janeiro e Fevereiro de 2024 · 1 041 151 registos após limpeza

| Variável | Tipo | Descrição |
|---|---|---|
| `month` | Numérico discreto | Mês do voo (1–2) |
| `day_of_month` | Numérico discreto | Dia do mês (1–31) |
| `day_of_week` | Numérico discreto | Dia da semana (1=Segunda … 7=Domingo) |
| `origin` | Categórico textual | Aeroporto de origem (código IATA) |
| `distance` | Numérico contínuo | Distância do voo em milhas |
| `is_long_flight` | Binário numérico | 1 se distância > 1 500 mi |
| `is_short_flight` | Binário numérico | 1 se distância < 300 mi |
| `is_weekend` | Binário numérico | 1 se sábado ou domingo |
| `cancelled` | Binário numérico | **Alvo** — 0 operacional / 1 cancelado |

**Desequilíbrio:** 98.47% operacionais · 1.53% cancelados (após limpeza)
        """)

    with t2:
        st.markdown("""
**M1 — Iniciação** · Objetivos SMART, perguntas de investigação, análise de viabilidade.

**M2 — Exploração e Preparação**
- *Missings* mantidos (associados a cancelamentos — informação estrutural, não erro)
- Remoção de 7 401 duplicados e 23 *outliers* físicos (velocidades supersónicas)
- Exclusão de variáveis com *data leakage*: `weather_delay`, `taxi_out`, `air_time`, etc.
- *Feature engineering*: `is_long_flight`, `is_short_flight`, `is_weekend`
- *One-Hot Encoding* em `origin` (~338 aeroportos · 340 features totais)
- Sem escalonamento — HistGradient Boosting e XGBoost são invariantes a transformações monotónicas

**M3 — Modelação**
- *Baseline*: Regressão Logística com `class_weight='balanced'`
- 3 candidatos comparados com *threshold* otimizado via curva Precisão-Recall
- `GridSearchCV` com `scoring='average_precision'` + `TunedThresholdClassifierCV`
- Divisão 80/20 estratificada · `StratifiedKFold` k=5 na validação cruzada
- Critério de seleção: **Average Precision** (PR-AUC) — mais honesta que ROC-AUC em classes desequilibradas
        """)

    with t3:
        res = pd.DataFrame({
            "Modelo": [
                "Regressão Logística (baseline)",
                "XGBoost",
                "HistGradient Boosting",
                "HistGB Otimizado ✓ (modelo final)",
            ],
            "F1-Score":      [0.087, 0.166, 0.163, 0.151],
            "Recall":        [0.233, 0.365, 0.214, 0.266],
            "Precision":     [0.053, 0.108, 0.132, 0.105],
            "AUC-ROC":       [0.761, 0.869, 0.867, 0.854],
            "Avg Precision": [0.042, 0.096, 0.096, 0.087],
            "Threshold":     [0.734, 0.785, 0.846, 0.806],
        })
        st.dataframe(res, hide_index=True, use_container_width=True)
        st.markdown(f"""
- **HistGradient Boosting** selecionado com base na *Average Precision* (critério mais honesto para classes raras)
- *Threshold* {THRESHOLD} otimizado por validação cruzada (5 folds) · maximiza F1
- De 3 181 cancelamentos no teste, o modelo detetou **845** (Recall = {RECALL*100:.1f}%)
- Por cada cancelamento detetado, geram-se ~8.5 alertas preventivos (falsos positivos)
- Teto de desempenho limitado pela ausência de dados meteorológicos em tempo real
        """)

    with t4:
        st.markdown("""
**Dataset e fontes de dados:**
- **Nadeem, A. (2024).** *Flight Delay & Cancellation Data (1M+ 2024)*. Kaggle.
- **BTS (2024).** *Marketing Carrier On-Time Performance*. U.S. DOT. https://transtats.bts.gov

**Bibliotecas e metodologia:**
- **Pedregosa et al. (2011).** Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- **Chen & Guestrin (2016).** XGBoost: A scalable tree boosting system. *KDD 2016*.
- **Lundberg & Lee (2017).** A unified approach to interpreting model predictions. *NeurIPS 2017*.

---
**Docente:** Dora Melo — dmelo@iscac.pt
**Notebook Kaggle:** [Modelação — Previsão de Cancelamentos](https://www.kaggle.com/code/rodrigoramooos/modelacao-previsao-de-cancelamentos-em-voos)
        """)
