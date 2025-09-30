import json
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, Input, Output, dcc, html, no_update, ctx

# ==================== Caminhos e dados ====================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "dataset.json"
if not DATASET_PATH.exists():
    raise FileNotFoundError("Gere o dataset com: python scripts/build_dataset.py")

dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

# ==================== DataFrames base ====================
df_values = pd.DataFrame(dataset.get("values", []))
df_over   = pd.DataFrame(dataset.get("overview", []))

# ==================== Anos (referência) e divulgação (+2) ====================
if len(df_values):
    YEARS_REF_ALL = sorted({int(y) for y in df_values["year"].unique()})
else:
    YEARS_REF_ALL = []

# Mostrar apenas 2019+ (ou o primeiro ano disponível, se for >2019) para evitar espaço morto
START_YEAR = max(2019, YEARS_REF_ALL[0]) if YEARS_REF_ALL else 2019
YEARS_REF  = [y for y in YEARS_REF_ALL if y >= START_YEAR]

def year_pub(y_ref: int) -> int:
    return int(y_ref) + 2

YEARS_PUB   = [year_pub(y) for y in YEARS_REF]
PUB_TO_REF  = {year_pub(y): y for y in YEARS_REF}
YEAR_LAST_REF = YEARS_REF[-1] if YEARS_REF else None
YEAR_LAST_PUB = year_pub(YEAR_LAST_REF) if YEAR_LAST_REF is not None else None

# ==================== Geografias ====================
GEOS      = {g["id"]: g for g in dataset["geos"]}
STATE_ID  = "Pará"
MACROS    = [g["id"] for g in dataset["geos"] if g["type"] == "macro" and g.get("parent") in (STATE_ID, None, "PA")]

# ==================== ODS Labels ====================
ODS_LABELS = {
    "ODS01": "ODS 01 — Erradicação da Pobreza",
    "ODS02": "ODS 02 — Fome Zero e Agricultura Sustentável",
    "ODS03": "ODS 03 — Saúde e Bem-Estar",
    "ODS04": "ODS 04 — Educação de Qualidade",
    "ODS05": "ODS 05 — Igualdade de Gênero",
    "ODS06": "ODS 06 — Água Potável e Saneamento",
    "ODS07": "ODS 07 — Energia Acessível e Limpa",
    "ODS08": "ODS 08 — Trabalho Decente e Crescimento Econômico",
    "ODS09": "ODS 09 — Indústria, Inovação e Infraestrutura",
    "ODS10": "ODS 10 — Redução das Desigualdades",
    "ODS11": "ODS 11 — Cidades e Comunidades Sustentáveis",
    "ODS12": "ODS 12 — Consumo e Produção Responsáveis",
    "ODS13": "ODS 13 — Ação Contra a Mudança do Clima",
    "ODS14": "ODS 14 — Vida na Água",
    "ODS15": "ODS 15 — Vida Terrestre",
    "ODS16": "ODS 16 — Paz, Justiça e Instituições Eficazes",
    "ODS17": "ODS 17 — Parcerias e Meios de Implementação",
}

# Metadados de componentes (se existirem)
COMP_META = {}
for ods_id, od in dataset.get("ods", {}).items():
    for c in od.get("components", []):
        COMP_META[(ods_id, c["id"])] = c

# ==================== Helpers ====================
def fmt_num(v):
    if v is None: return "—"
    try: v=float(v)
    except: return "—"
    if abs(v)>=1000 and abs(v)<1_000_000: return f"{v:,.0f}".replace(",", ".")
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

def ods_num(ods_id:str) -> str:
    return ods_id[-2:]  # "ODS03" -> "03"

# Ícones (suporta "1.png" .. "17.png" OU "01.png" .. "17.png")
ASSETS_ODS_DIR = BASE_DIR / "assets" / "ods"
def icon_url_for(ods_id: str) -> str:
    n2 = int(ods_num(ods_id))
    p1 = ASSETS_ODS_DIR / f"{n2}.png"
    p2 = ASSETS_ODS_DIR / f"{n2:02d}.png"
    if p1.exists(): return f"/assets/ods/{n2}.png"
    if p2.exists(): return f"/assets/ods/{n2:02d}.png"
    return "/assets/ods/fallback.png"

def get_series(ods_id, comp_id, geo_id):
    """Série completa, porém apenas anos visíveis (>= START_YEAR)."""
    if df_values.empty:
        return [{"year_ref": y, "value": None} for y in YEARS_REF]
    sub = df_values[(df_values["ods"]==ods_id)&(df_values["component_id"]==comp_id)&(df_values["geo_id"]==geo_id)]
    m = {int(r["year"]): r["value"] for _, r in sub.iterrows()}
    return [{"year_ref": y, "value": m.get(y)} for y in YEARS_REF]

def get_point(ods_id, comp_id, geo_id, year_ref):
    if df_values.empty:
        return None
    sub = df_values[
        (df_values["ods"]==ods_id) &
        (df_values["component_id"]==comp_id) &
        (df_values["geo_id"]==geo_id) &
        (df_values["year"]==year_ref)
    ]
    return float(sub["value"].iloc[0]) if len(sub) else None

def compute_ods_index(ods_id: str, geo_id: str, year_ref: int):
    if len(df_values):
        sub = df_values[(df_values["ods"]==ods_id) & (df_values["geo_id"]==geo_id) & (df_values["year"]==year_ref)]
        if len(sub):
            val = float(sub["value"].mean())
            is_pct = 0 <= val <= 100
            return val, is_pct
    if len(df_over):
        sub2 = df_over[(df_over["ods"]==ods_id)&(df_over["geo_id"]==geo_id)]
        if len(sub2):
            val = float(sub2["value"].mean())
            is_pct = 0 <= val <= 100
            return val, is_pct
    return None, False

# ---------- Município -> Macro (robusto) ----------
MACRO_OF_MUNI = dataset.get("macro_of_muni", {})
if not MACRO_OF_MUNI:
    for g in dataset["geos"]:
        if g.get("type") == "municipality":
            muni = g["id"]
            macro = (
                g.get("parent") or g.get("macro") or g.get("macro_id")
                or g.get("macroRegion") or g.get("macro_regiao") or g.get("macro-regiao")
            )
            if macro:
                MACRO_OF_MUNI[muni] = macro

def parent_macro_of(muni_id: str) -> str | None:
    if muni_id in MACRO_OF_MUNI:
        return MACRO_OF_MUNI[muni_id]
    g = GEOS.get(muni_id)
    if not g: return None
    return (
        g.get("parent") or g.get("macro") or g.get("macro_id")
        or g.get("macroRegion") or g.get("macro_regiao") or g.get("macro-regiao")
    )

# ---------- Séries contexto cumulativo ----------
def build_context_series(ods_id, comp_id, level, macro, muni):
    """
    Estado -> [Pará]
    Macro  -> [Macro, Pará]
    Município -> [Município, Macro do município, Pará]
    """
    series = []
    if level == "estado":
        series.append(("Pará", get_series(ods_id, comp_id, STATE_ID)))
    elif level == "macro" and macro:
        series.append(("Regional", get_series(ods_id, comp_id, macro)))
        series.append(("Pará", get_series(ods_id, comp_id, STATE_ID)))
    elif level == "municipio" and muni:
        macro_id = parent_macro_of(muni) or macro
        series.append(("Município", get_series(ods_id, comp_id, muni)))
        if macro_id:
            series.append(("Regional", get_series(ods_id, comp_id, macro_id)))
        series.append(("Pará", get_series(ods_id, comp_id, STATE_ID)))
    else:
        series.append(("Pará", get_series(ods_id, comp_id, STATE_ID)))
    return series

# ==================== App ====================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Painel ODS – Pará (Claro)"

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Expor o WSGI para produção (Gunicorn vai importar "app:server")
server = app.server
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

app.layout = dbc.Container([
    # State
    dcc.Store(id="st-ods",        data=None),
    dcc.Store(id="st-year-ref",   data=YEAR_LAST_REF),
    dcc.Store(id="st-level",      data="estado"),

    # Header
    dbc.Row([
        dbc.Col(html.Div([
            html.Div("ODS – Pará", className="brand-title"),
            html.Div("Exploração multi-nível: Estado, Macro-Região e Municípios", className="brand-sub"),
        ]), md=7),

        # Slider "Ano de divulgação"
        dbc.Col(html.Div([
            html.Div("Ano de divulgação", className="slider-title"),
            dcc.Slider(
                id="sl-year-pub",
                min=min(YEARS_PUB) if YEARS_PUB else 0,
                max=max(YEARS_PUB) if YEARS_PUB else 1,
                step=None,
                value=YEAR_LAST_PUB,
                marks={p: {"label": str(p)} for p in YEARS_PUB},
                tooltip={"always_visible": True, "placement": "bottom"},
                className="year-slider"
            ),
            html.Div(id="lbl-ref", className="slider-sub")
        ]), md=3, className="mb-2"),

        # Removido: Tabs (não há mais aba Educação)
        dbc.Col(html.Div(), md=2),
    ], className="mt-2 align-items-end"),

    # Filtros
    dbc.Row([
        dbc.Col(dcc.RadioItems(
            id="rd-level",
            options=[{"label": "Estado", "value": "estado"},
                     {"label": "Macro-Região", "value": "macro"},
                     {"label": "Município", "value": "municipio"}],
            value="estado", inline=True,
            inputStyle={"marginRight":"6px","marginLeft":"12px"}
        ), md=6),

        dbc.Col(dcc.Dropdown(id="dd-macro", className="themed-dd",
                             options=[{"label": m, "value": m} for m in MACROS],
                             placeholder="Selecione a Macro-Região",
                             style={"display":"none"}), md=3),
        dbc.Col(dcc.Dropdown(id="dd-muni", className="themed-dd",
                             options=[], placeholder="Selecione o Município",
                             style={"display":"none"}), md=3),
    ], className="my-2"),

    html.Hr(),

    html.Div(id="view")
], fluid=True, className="page-light")

# ==================== callbacks ====================
@app.callback(
    Output("st-year-ref","data"),
    Output("lbl-ref","children"),
    Input("sl-year-pub","value")
)
def sync_year_ref(pub_year):
    if pub_year is None and YEARS_PUB:
        pub_year = YEARS_PUB[-1]
    ref = PUB_TO_REF.get(pub_year, None)
    label = f"(ref {ref})" if ref is not None else ""
    return ref, label

@app.callback(
    Output("dd-macro","style"), Output("dd-muni","style"),
    Output("st-level","data"), Input("rd-level","value")
)
def _toggle_level(level):
    if level=="macro":     return {"display":"block"}, {"display":"none"}, level
    if level=="municipio": return {"display":"block"}, {"display":"block"}, level
    return {"display":"none"}, {"display":"none"}, level

def list_municipios():
    return sorted([g["id"] for g in dataset["geos"] if g["type"]=="municipality"])

@app.callback(Output("dd-muni","options"), Input("dd-macro","value"), Input("st-level","data"))
def _load_munis(_, level):
    return [{"label": m, "value": m} for m in list_municipios()] if level=="municipio" else []

@app.callback(Output("view","children"),
              Input("st-year-ref","data"),
              Input("st-level","data"),
              Input("dd-macro","value"),
              Input("dd-muni","value"),
              Input("st-ods","data"))
def render_view(year_ref, level, macro, muni, ods_selected):
    if year_ref is None and YEARS_REF:
        year_ref = YEARS_REF[-1]
    geo = "Pará"
    if level=="macro" and macro: geo = macro
    if level=="municipio" and muni: geo = muni
    return render_ods(geo, year_ref, level, macro, muni, ods_selected)

# ---------- ODS ----------
def make_ods_card(ods_id: str, title: str, geo: str, year_ref: int):
    n = int(ods_num(ods_id))
    val, is_pct = compute_ods_index(ods_id, geo, year_ref)
    val_txt = fmt_num(val) + ("%" if (val is not None and is_pct) else "")
    return html.Div(
        [
            html.Div(className="ods-pill", style={"backgroundImage": f"url('{icon_url_for(ods_id)}')" }),
            html.Div([
                html.Div(f"ODS {n}", className="ods-small"),
                html.Div(title, className="ods-big"),
                html.Div([
                    "Índice ", html.Span(val_txt, className="ods-idx"),
                    html.Span(f" • divulgação {year_pub(year_ref)}", className="ods-small muted")
                ], className="ods-small"),
            ], className="ods-text")
        ],
        id={"type":"card-ods", "ods": ods_id},
        n_clicks=0, className="ods-row", title=title
    )

def render_ods(geo, year_ref, level, macro, muni, ods_selected):
    expected = [f"ODS{i:02d}" for i in range(1, 18)]
    ods_list = []
    for key in expected:
        label = dataset.get("ods", {}).get(key, {}).get("label", ODS_LABELS[key])
        ods_list.append({"id": key, "label": label})

    header = dbc.Row([dbc.Col(html.Div([
        html.Span(f"{geo}", className="crumb-item"),
        html.Span(f" • divulgação {year_pub(year_ref)}", className="crumb-sep"),
    ]))])

    # 3 colunas: 6 + 6 + 5
    counts = [6, 6, 5]
    if len(ods_list) != sum(counts):
        k = len(ods_list); base = k // 3; counts = [base, base, k - 2*base]

    cols, start = [], 0
    for c in counts:
        chunk = ods_list[start:start+c]; start += c
        stack = []
        for o in chunk:
            title = o["label"].split("—", 1)[-1].strip() if "—" in o["label"] else o["label"]
            stack.append(make_ods_card(o["id"], title, geo, year_ref))
        cols.append(dbc.Col(stack, md=4, xs=12))
    grid = dbc.Row(cols, className="g-3")

    if not ods_selected:
        return html.Div([header, html.H2("Índices / ODS", className="h2-title"), grid])

    comps = dataset.get("ods", {}).get(ods_selected, {}).get("components", [])
    if not comps:
        return html.Div([
            header, html.H2("Índices / ODS", className="h2-title"), grid,
            html.Hr(), html.Div("Sem componentes cadastrados para este ODS.", className="tile-sub")
        ])

    # === Gráficos de componentes ===
    comp_cards = []
    for c in comps:
        code = c.get("code") or ""
        nice_title = (f"[{code}] " if code else "") + (c.get("label") or "Indicador")
        short_desc = c.get("desc") or "Descrição não encontrada no descritivo."
        unit_txt   = f" • Unidade: {c['unit']}" if c.get("unit") else ""

        context = build_context_series(ods_selected, c["id"], level, macro, muni)
        colors  = {"Município": "#C8A530", "Regional": "#D12D4A", "Pará": "#1976D2"}

        fig = go.Figure()
        max_y = 0.0
        for name, serie in context:
            xs_pub = [year_pub(p["year_ref"]) for p in serie]
            ys     = [p["value"] for p in serie]
            max_y  = max(max_y, max([v for v in ys if v is not None] or [0]))
            fig.add_trace(go.Scatter(
                x=xs_pub, y=ys, mode="lines+markers", name=name,
                line=dict(width=2, color=colors.get(name, "#1976D2")),
                marker=dict(size=6, color=colors.get(name, "#1976D2"))
            ))
        fig.update_layout(
            margin=dict(l=16,r=12,t=18,b=30), height=260, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12)),
            xaxis=dict(
                tickmode="array",
                tickvals=YEARS_PUB,
                ticktext=[str(p) for p in YEARS_PUB],
                title="Ano (divulgação)"
            ),
            yaxis=dict(range=[0, (max_y*1.10 if max_y>0 else 1)], title="")
        )

        comp_cards.append(
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.Div(nice_title, className="tile-title"),
                    html.Div(html.Span(fmt_num(get_point(ods_selected, c["id"], geo, year_ref)), className="tile-number")),
                    html.Div(short_desc + unit_txt, className="tile-sub"),
                    dcc.Graph(figure=fig, config={"displayModeBar": False})
                ]), className="card-tile"),
                md=6, lg=4, xs=12
            )
        )

    # === Barras comparativas ===
    bars = []
    if level=="estado":
        for c in comps:
            rows=[]
            for m in MACROS:
                v = get_point(ods_selected, c["id"], m, year_ref)
                if v is not None: rows.append({"geo": m, "value": v})
            if rows:
                figb = go.Figure()
                figb.add_trace(go.Bar(x=[r["geo"] for r in rows], y=[r["value"] for r in rows]))
                max_bar = max([r["value"] for r in rows] or [0])
                figb.update_layout(margin=dict(l=10,r=10,t=30,b=80), height=320, xaxis=dict(tickangle=-30),
                                   template="plotly_white",
                                   title=f"Ano de divulgação {year_pub(year_ref)}")
                figb.update_yaxes(range=[0, max_bar*1.10 if max_bar>0 else 1])
                bars.append(dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div(f"Comparação por Macro-Região • {c.get('label','')}", className="tile-sub"),
                    dcc.Graph(figure=figb, config={"displayModeBar": False})
                ]), className="card-tile"), md=12, lg=6))
    elif level=="macro" and macro:
        muni_ids = [g["id"] for g in dataset["geos"] if g["type"]=="municipality" and (parent_macro_of(g["id"])==macro)]
        for c in comps:
            rows=[]
            for gid in muni_ids:
                v = get_point(ods_selected, c["id"], gid, year_ref)
                if v is not None: rows.append({"geo":gid, "value":v})
            if rows:
                rows = sorted(rows, key=lambda r: r["value"], reverse=True)[:20]
                figb = go.Figure()
                figb.add_trace(go.Bar(x=[r["geo"] for r in rows], y=[r["value"] for r in rows]))
                max_bar = max([r["value"] for r in rows] or [0])
                figb.update_layout(margin=dict(l=10,r=10,t=30,b=120), height=380, xaxis=dict(tickangle=-60),
                                   template="plotly_white",
                                   title=f"Ano de divulgação {year_pub(year_ref)}")
                figb.update_yaxes(range=[0, max_bar*1.10 if max_bar>0 else 1])
                bars.append(dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div(f"Top municípios (ano) • {c.get('label','')}", className="tile-sub"),
                    dcc.Graph(figure=figb, config={"displayModeBar": False})
                ]), className="card-tile"), md=12))

    ods_title = (dataset.get("ods", {}).get(ods_selected, {}) or {}).get("label", ODS_LABELS[ods_selected])
    return html.Div([
        header,
        html.H2("Índices / ODS", className="h2-title"), grid,
        html.Hr(),
        html.H2(f"Componentes – {ods_title} ({geo}, divulgação {year_pub(year_ref)})", className="h2-title"),
        dbc.Row(comp_cards, className="g-3"),
        html.Div(style={"height":"12px"}),
        dbc.Row(bars, className="g-3")
    ])

# Clique no ODS
@app.callback(Output("st-ods","data", allow_duplicate=True),
              Input({"type":"card-ods","ods": dash.ALL}, "n_clicks"),
              prevent_initial_call=True)
def choose_ods(_):
    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get("type")=="card-ods":
        return trig.get("ods")
    return no_update

# run (dev local)
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
