#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, re
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html, no_update, ctx, MATCH

# ==================== Caminhos e dados ====================
BASE_DIR     = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "dataset.json"
DESC_XLSX    = BASE_DIR / "data" / "Descritivos de Indicadores ODS 2023-2025.xlsx"

if not DATASET_PATH.exists():
    raise FileNotFoundError("Gere o dataset com: python scripts/build_dataset.py")

dataset   = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
df_values = pd.DataFrame(dataset.get("values", []))
df_over   = pd.DataFrame(dataset.get("overview", []))

# ==================== Anos (referência) e divulgação (+2) ====================
def year_pub(y_ref: int) -> int: 
    return int(y_ref) + 2

if len(df_values):
    YEARS_REF_ALL = sorted({int(y) for y in df_values["year"].unique()})
else:
    YEARS_REF_ALL = []

TARGET_PUB    = {2023, 2024, 2025}
YEARS_REF     = [y for y in YEARS_REF_ALL if year_pub(y) in TARGET_PUB]
YEARS_PUB     = [year_pub(y) for y in YEARS_REF]
PUB_TO_REF    = {year_pub(y): y for y in YEARS_REF}
YEAR_LAST_REF = YEARS_REF[-1] if YEARS_REF else None
YEAR_LAST_PUB = year_pub(YEAR_LAST_REF) if YEAR_LAST_REF is not None else None

# ==================== Geografias ====================
GEOS      = {g["id"]: g for g in dataset.get("geos", [])}
STATE_ID  = "Pará"
MACROS    = [g["id"] for g in dataset.get("geos", []) if g.get("type") == "macro" and g.get("parent") in (STATE_ID, None, "PA")]

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

# Metadados existentes no dataset.json
COMP_META = {}
for ods_id, od in dataset.get("ods", {}).items():
    for c in od.get("components", []):
        COMP_META[(ods_id, c.get("id"))] = c

# ==================== Helpers ====================
def fmt_num(v):
    if v is None: return "—"
    try: v=float(v)
    except: return "—"
    if abs(v)>=1000 and abs(v)<1_000_000: return f"{v:,.0f}".replace(",", ".")
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

def ods_num(ods_id:str) -> str: return ods_id[-2:]

ASSETS_ODS_DIR = BASE_DIR / "assets" / "ods"
def icon_url_for(ods_id: str) -> str:
    n2 = int(ods_num(ods_id))
    p1 = ASSETS_ODS_DIR / f"{n2}.png"
    p2 = ASSETS_ODS_DIR / f"{n2:02d}.png"
    if p1.exists(): return f"/assets/ods/{n2}.png"
    if p2.exists(): return f"/assets/ods/{n2:02d}.png"
    return "/assets/ods/fallback.png"

def get_series(ods_id, comp_id, geo_id):
    if df_values.empty:
        return [{"year_ref": y, "value": None} for y in YEARS_REF]
    sub = df_values[(df_values["ods"]==ods_id)&(df_values["component_id"]==comp_id)&(df_values["geo_id"]==geo_id)]
    m = {int(r["year"]): r["value"] for _, r in sub.iterrows()}
    return [{"year_ref": y, "value": m.get(y)} for y in YEARS_REF]

def get_point(ods_id, comp_id, geo_id, year_ref):
    if df_values.empty: return None
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
            val = float(sub["value"].mean()); is_pct = 0 <= val <= 100
            return val, is_pct
    if len(df_over):
        sub2 = df_over[(df_over["ods"]==ods_id)&(df_over["geo_id"]==geo_id)]
        if len(sub2):
            val = float(sub2["value"].mean()); is_pct = 0 <= val <= 100
            return val, is_pct
    return None, False

# ---------- Município -> Macro ----------
MACRO_OF_MUNI = dataset.get("macro_of_muni", {}) or {}
def parent_macro_of(muni_id: str) -> str | None:
    if muni_id in MACRO_OF_MUNI:
        return MACRO_OF_MUNI[muni_id]
    g = GEOS.get(muni_id)
    if not g: return None
    return (
        g.get("parent") or g.get("macro") or g.get("macro_id")
        or g.get("macroRegion") or g.get("macro_regiao") or g.get("macro-regiao")
    )

# ==================== Descritivos (Nome/Descrição/Fonte) ====================
def _norm_txt(s: str) -> str:
    if s is None: return ""
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    s = re.sub(r"[^\w\s\-–—/]", "", s)
    return s

# 1) por índice: (ano_pub, ODSxx, idx) -> meta
# 2) por título normalizado (fallback): (ano_pub, ODSxx, tnorm) -> meta
DESC_BY_INDEX  = {}
DESC_BY_TITLE  = {}
DESC_BUCKETS   = {}  # (ano, ODS) -> [(tnorm, meta)]

def load_descriptions_excel(path: Path):
    if not path.exists():
        print(f"[WARN] Descritivos não encontrado: {path}")
        return
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
    except Exception as ex:
        print(f"[WARN] Falha abrindo {path.name}: {ex}")
        return

    total = 0
    for sheet in xls.sheet_names:
        yr = None
        m = re.search(r"20(23|24|25)", sheet)
        if m: yr = int("20"+m.group(1))
        else:
            try:
                v = int(sheet)
                if v in {2023, 2024, 2025}: yr = v
            except: pass
        if yr is None: continue

        try:
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        except Exception as ex:
            print(f"[WARN] Não consegui ler aba '{sheet}': {ex}"); continue

        cols = {c: str(c).strip().lower() for c in df.columns}
        def col_like(*tokens):
            for c in df.columns:
                s = cols.get(c, "")
                if all(t in s for t in tokens): return c
            return None

        col_ods  = col_like("ods") or col_like("objetivo")
        col_name = col_like("indicador") or col_like("nome") or col_like("título") or col_like("titulo")
        col_text = col_like("cálcul") or col_like("calculo") or col_like("descr") or col_like("defini")
        col_src  = col_like("fonte")

        counters = {}
        for _, r in df.iterrows():
            ods_key = None
            if col_ods:
                m2 = re.search(r"(\d{1,2})", str(r.get(col_ods, "")))
                if m2: ods_key = f"ODS{int(m2.group(1)):02d}"

            title = str(r.get(col_name) or "").strip()
            text  = str(r.get(col_text) or "").strip()
            src   = str(r.get(col_src)  or "").strip()
            if not (ods_key and title): 
                continue

            counters.setdefault(ods_key, 0)
            counters[ods_key] += 1
            idx = counters[ods_key]

            meta = {"title": title, "desc": text, "source": src}
            DESC_BY_INDEX[(yr, ods_key, idx)] = meta
            tnorm = _norm_txt(title)
            DESC_BY_TITLE[(yr, ods_key, tnorm)] = meta
            DESC_BUCKETS.setdefault((yr, ods_key), []).append((tnorm, meta))
            total += 1

    print(f"[INFO] Descritivos carregados: {total} linhas, "
          f"{len(DESC_BY_INDEX)} por índice, {len(DESC_BY_TITLE)} por título.")

load_descriptions_excel(DESC_XLSX)

def _tokens(s: str):
    return {t for t in re.findall(r"\w+", _norm_txt(s)) if len(t) >= 4}

def _component_index_from(c: dict, fallback_position: int) -> int | None:
    lbl = (c.get("label") or "").strip()
    m = re.match(r"\s*\[(\d+)\]", lbl)
    if m:
        try: return int(m.group(1))
        except: pass
    code = (c.get("code") or "").strip()
    if code:
        m2 = re.match(r"(\d+)", code)
        if m2:
            try: return int(m2.group(1))
            except: pass
    return fallback_position if fallback_position is not None else None

def guess_desc_for_component(ods_id: str, comp_meta: dict, pub_year: int, ordinal: int):
    # 1) índice (preferencial)
    idx = _component_index_from(comp_meta, ordinal)
    if idx is not None:
        meta = DESC_BY_INDEX.get((pub_year, ods_id, int(idx)))
        if meta: return meta
    # 2) match exato por título
    title = (comp_meta.get("label") or "").strip()
    if title:
        t2 = re.sub(r"^\s*\[[^\]]+\]\s*", "", title).strip()
        for t in (title, t2):
            k = (pub_year, ods_id, _norm_txt(t))
            if k in DESC_BY_TITLE: return DESC_BY_TITLE[k]
    # 3) fuzzy mesmo ODS
    bucket = DESC_BUCKETS.get((pub_year, ods_id), [])
    if bucket and title:
        want = _tokens(title)
        best, best_score = None, 0
        for tnorm, meta in bucket:
            have = set(re.findall(r"\w+", tnorm))
            score = len(want.intersection(have))
            if score > best_score:
                best, best_score = meta, score
        if best_score > 0: return best
    # 4) fuzzy ano todo
    if title:
        want = _tokens(title)
        best, best_score = None, 0
        for (yr, _ods), items in DESC_BUCKETS.items():
            if yr != pub_year: continue
            for tnorm, meta in items:
                have = set(re.findall(r"\w+", tnorm))
                score = len(want.intersection(have))
                if score > best_score:
                    best, best_score = meta, score
        if best_score > 0: return best
    return None

# ==== título particularizado dos gráficos ====
def display_title_for_chart(desc_pkg: dict | None, comp_meta: dict, pos: int) -> str:
    excel_title = (desc_pkg or {}).get("title", "") if desc_pkg else ""
    if excel_title:
        return excel_title.strip()
    base = (comp_meta.get("label") or "Indicador").strip()
    base = re.sub(r"^\s*\[[^\]]+\]\s*", "", base)
    code = (comp_meta.get("code") or "").strip()
    if code and not re.match(r"^\s*\[", base):
        return f"[{code}] {base}"
    return base

# ==================== App ====================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Painel ODS – Pará (Claro)"
server = app.server

app.layout = dbc.Container([
    dcc.Store(id="st-ods",        data=None),
    dcc.Store(id="st-year-ref",   data=YEAR_LAST_REF),
    dcc.Store(id="st-level",      data="estado"),
    dcc.Store(id="st-viewmode",   data="grid"),

    dbc.Row([
        dbc.Col(html.Div([
            html.Div("ODS – Pará", className="brand-title"),
            html.Div("Exploração multi-nível: Estado, Macro-Região e Municípios", className="brand-sub"),
        ]), md=7),

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
        ]), md=5, className="mb-2"),
    ], className="mt-2 align-items-end"),

    dbc.Row([
        dbc.Col(dcc.RadioItems(
            id="rd-level",
            options=[{"label":"Estado","value":"estado"},
                     {"label":"Região de Integração","value":"macro"},
                     {"label":"Município","value":"municipio"}],
            value="estado", inline=True,
            inputStyle={"marginRight":"6px","marginLeft":"12px"}
        ), md=6),
        dbc.Col(dcc.Dropdown(id="dd-macro", className="themed-dd",
                             options=[{"label": m, "value": m} for m in MACROS],
                             placeholder="Selecione a Região de Integração",
                             style={"display":"none"}), md=3),
        dbc.Col(dcc.Dropdown(id="dd-muni", className="themed-dd",
                             options=[], placeholder="Selecione o Município",
                             style={"display":"none"}), md=3),
    ], className="my-2"),

    html.Div(id="focus-toolbar", className="focus-toolbar", children=[
        dbc.Button("← Voltar aos ODS", id="btn-back", color="light", className="btn-back", n_clicks=0)
    ]),

    html.Hr(),
    html.Div(id="view")
], fluid=True, className="page-light")

# ==================== Callbacks ====================
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
    return sorted([g["id"] for g in dataset.get("geos", []) if g.get("type")=="municipality"])

@app.callback(Output("dd-muni","options"), Input("dd-macro","value"), Input("st-level","data"))
def _load_munis(_, level):
    return [{"label": m, "value": m} for m in list_municipios()] if level=="municipio" else []

@app.callback(Output("focus-toolbar","style"), Input("st-viewmode","data"))
def _tool_visibility(mode): return {"display": "flex"} if mode == "focus" else {"display": "none"}

@app.callback(
    Output("view","children"),
    Input("st-year-ref","data"),
    Input("st-level","data"),
    Input("dd-macro","value"),
    Input("dd-muni","value"),
    Input("st-ods","data"),
    Input("st-viewmode","data")
)
def render_view(year_ref, level, macro, muni, ods_selected, viewmode):
    try:
        if year_ref is None and YEARS_REF:
            year_ref = YEARS_REF[-1]
        geo = "Pará"
        if level=="macro" and macro: geo = macro
        if level=="municipio" and muni: geo = muni

        if viewmode != "focus":
            return render_ods_grid(geo, year_ref)
        if not ods_selected:
            return render_ods_grid(geo, year_ref)
        return render_ods_focus(geo, year_ref, level, macro, muni, ods_selected)
    except Exception as ex:
        return dbc.Alert([html.B("Falha ao montar a visão."), html.Br(), str(ex)], color="danger")

# ---------- LISTA ODS (GRID) ----------
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
        n_clicks=0, className="ods-row fade-in", title=title
    )

def render_ods_grid(geo, year_ref):
    expected = [f"ODS{i:02d}" for i in range(1, 18)]
    ods_list = []
    for key in expected:
        label = dataset.get("ods", {}).get(key, {}).get("label", ODS_LABELS[key])
        ods_list.append({"id": key, "label": label})

    header = dbc.Row([dbc.Col(html.Div([
        html.Span(f"{geo}", className="crumb-item"),
        html.Span(f" • {year_ref}", className="crumb-sep"),  # ano de referência na lista também
        html.Span(f" • divulgação {year_pub(year_ref)}", className="crumb-sep"),
    ]))])

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

    return html.Div([header, html.H2("Índices / ODS", className="h2-title"), grid])

# ---------- FOCUS DE UM ODS ----------
def render_ods_focus(geo, year_ref, level, macro, muni, ods_selected):
    comps = dataset.get("ods", {}).get(ods_selected, {}).get("components", [])
    ods_title = (dataset.get("ods", {}).get(ods_selected, {}) or {}).get("label", ODS_LABELS[ods_selected])

    header = dbc.Row([dbc.Col(html.Div([
        html.Span(f"{geo}", className="crumb-item"),
        html.Span(" • ", className="crumb-sep"),
        html.Span(ods_title, className="crumb-item"),
        html.Span(f" • {year_ref}", className="crumb-sep"),  # >>> ANO DE REFERÊNCIA
        html.Span(f" • divulgação {year_pub(year_ref)}", className="crumb-sep"),
    ]))])

    if not comps:
        return html.Div([header, html.Div("Sem componentes cadastrados para este ODS.", className="tile-sub")])

    comp_cards = []
    pub_year = year_pub(year_ref)

    for pos, c in enumerate(comps, start=1):
        # ---------------- Descritivo (planilha) ----------------
        desc_pkg   = guess_desc_for_component(ods_selected, c, pub_year, ordinal=pos)
        long_title = display_title_for_chart(desc_pkg, c, pos)  # título particularizado
        short_desc = ((desc_pkg or {}).get("desc")   or (c.get("desc")   or "")).strip()
        source_txt = ((desc_pkg or {}).get("source") or (c.get("source") or "")).strip()
        unit_txt   = f" • Unidade: {c['unit']}" if c.get("unit") else ""

        nice_title = long_title  # usamos o mesmo título no topo do card

        # ---------------- Séries (linhas) em ANO DE REFERÊNCIA ----------------
        context = build_context_series(ods_selected, c["id"], level, macro, muni)
        colors  = {"Município": "#C8A530", "Regional": "#D12D4A", "Pará": "#1976D2"}

        fig = go.Figure(); max_y = 0.0
        for name, serie in context:
            xs_ref = [p["year_ref"] for p in serie]  # <<< REFERÊNCIA NO EIXO
            ys     = [p["value"] for p in serie]
            max_y  = max(max_y, max([v for v in ys if v is not None] or [0]))
            fig.add_trace(go.Scatter(
                x=xs_ref, y=ys, mode="lines+markers", name=name,
                line=dict(width=2, color=colors.get(name, "#1976D2")),
                marker=dict(size=6, color=colors.get(name, "#1976D2"))
            ))

        fig.update_layout(
            title=long_title, title_x=0.02, title_font=dict(size=13, family="Inter,Arial"),
            margin=dict(l=16,r=12,t=36,b=30), height=260, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12)),
            xaxis=dict(tickmode="array", tickvals=YEARS_REF, ticktext=[str(r) for r in YEARS_REF], title="Ano"),
            yaxis=dict(range=[0, (max_y*1.10 if max_y>0 else 1)], title="")
        )

        cid = c["id"]
        is_open_default = bool(short_desc or source_txt)

        comp_cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div([
                            html.Span(nice_title, className="tile-title"),
                            html.Span("▼", className="chev")
                        ], className="comp-title-row", id={"type":"comp-title", "cid": cid}, n_clicks=0),

                        # valor + badge do ano de referência
                        html.Div([
                            html.Span(fmt_num(get_point(ods_selected, c["id"], geo, year_ref)), className="tile-number"),
                            html.Span(f" • {year_ref}", className="tile-sub", style={"marginLeft": "6px"})
                        ]),

                        dbc.Collapse(
                            html.Div([
                                (html.Div(short_desc, className="comp-desc") if short_desc else ""),
                                (html.Div(["Fonte: ", html.Span(source_txt, className="comp-source")], className="comp-source-row") if source_txt else ""),
                                (html.Div(unit_txt, className="tile-sub") if unit_txt else "")
                            ], className="comp-collapse-body"),
                            id={"type":"comp-collapse", "cid": cid},
                            is_open=is_open_default
                        ),

                        dcc.Graph(figure=fig, config={"displayModeBar": False})
                    ]),
                    className="card-tile fade-in"
                ),
                md=6, lg=6, xl=4, xs=12
            )
        )

    # ---- Barras comparativas ----
    bars = []
    if level=="estado":
        for pos, c in enumerate(comps, start=1):
            desc_pkg   = guess_desc_for_component(ods_selected, c, pub_year, ordinal=pos)
            long_title = display_title_for_chart(desc_pkg, c, pos)

            rows=[]
            for m in MACROS:
                v = get_point(ods_selected, c["id"], m, year_ref)
                if v is not None: rows.append({"geo": m, "value": v})
            if rows:
                figb = go.Figure()
                figb.add_trace(go.Bar(x=[r["geo"] for r in rows], y=[r["value"] for r in rows]))
                max_bar = max([r["value"] for r in rows] or [0])
                figb.update_layout(
                    title=f"Comparação por Macro-Região — {long_title}",
                    title_x=0.02, title_font=dict(size=13, family="Inter,Arial"),
                    margin=dict(l=10,r=10,t=42,b=80), height=320, xaxis=dict(tickangle=-30),
                    template="plotly_white"
                )
                figb.update_yaxes(range=[0, max_bar*1.10 if max_bar>0 else 1], title="")
                bars.append(dbc.Col(dbc.Card(dbc.CardBody([
                    dcc.Graph(figure=figb, config={"displayModeBar": False})
                ]), className="card-tile"), md=12, lg=6))
    elif level=="macro" and macro:
        muni_ids = [g["id"] for g in dataset.get("geos", []) if g.get("type")=="municipality" and (parent_macro_of(g["id"])==macro)]
        for pos, c in enumerate(comps, start=1):
            desc_pkg   = guess_desc_for_component(ods_selected, c, pub_year, ordinal=pos)
            long_title = display_title_for_chart(desc_pkg, c, pos)

            rows=[]
            for gid in muni_ids:
                v = get_point(ods_selected, c["id"], gid, year_ref)
                if v is not None: rows.append({"geo":gid, "value":v})
            if rows:
                rows = sorted(rows, key=lambda r: r["value"], reverse=True)[:20]
                figb = go.Figure()
                figb.add_trace(go.Bar(x=[r["geo"] for r in rows], y=[r["value"] for r in rows]))
                max_bar = max([r["value"] for r in rows] or [0])
                figb.update_layout(
                    title=f"Top municípios — {long_title}",
                    title_x=0.02, title_font=dict(size=13, family="Inter,Arial"),
                    margin=dict(l=10,r=10,t=42,b=120), height=380, xaxis=dict(tickangle=-60),
                    template="plotly_white"
                )
                figb.update_yaxes(range=[0, max_bar*1.10 if max_bar>0 else 1], title="")
                bars.append(dbc.Col(dbc.Card(dbc.CardBody([
                    dcc.Graph(figure=figb, config={"displayModeBar": False})
                ]), className="card-tile"), md=12))

    return html.Div([
        header,
        html.H2("Componentes", className="h2-title"),
        dbc.Row(comp_cards, className="g-3"),
        html.Div(style={"height":"12px"}),
        dbc.Row(bars, className="g-3")
    ])

# Clique em ODS (vai para FOCUS)
@app.callback(
    Output("st-ods","data", allow_duplicate=True),
    Output("st-viewmode","data", allow_duplicate=True),
    Input({"type":"card-ods","ods": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def choose_ods(_):
    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get("type")=="card-ods":
        return trig.get("ods"), "focus"
    return no_update, no_update

# Voltar para GRID
@app.callback(
    Output("st-viewmode","data", allow_duplicate=True),
    Output("st-ods","data", allow_duplicate=True),
    Input("btn-back","n_clicks"),
    prevent_initial_call=True
)
def back_to_grid(n):
    if n: return "grid", None
    return no_update, no_update

# Toggle colapso por componente (clicando no título)
@app.callback(
    Output({"type":"comp-collapse","cid": MATCH}, "is_open"),
    Input({"type":"comp-title","cid": MATCH}, "n_clicks"),
    State({"type":"comp-collapse","cid": MATCH}, "is_open"),
    prevent_initial_call=True
)
def _toggle_comp(n, is_open):
    return (not is_open) if n else is_open

# ---------- Séries contexto cumulativo ----------
def build_context_series(ods_id, comp_id, level, macro, muni):
    series = []
    if level == "estado":
        series.append(("Pará", get_series(ods_id, comp_id, STATE_ID)))
    elif level == "macro" and macro:
        series.append(("Regional", get_series(ods_id, comp_id, macro)))
        series.append(("Pará", get_series(ods_id, comp_id, STATE_ID)))
    elif level == "municipio" and muni:
        macro_id = parent_macro_of(muni) or macro
        series.append(("Município", get_series(ods_id, comp_id, muni)))
        if macro_id: series.append(("Regional", get_series(ods_id, comp_id, macro_id)))
        series.append(("Pará", get_series(ods_id, comp_id, STATE_ID)))
    else:
        series.append(("Pará", get_series(ods_id, comp_id, STATE_ID)))
    return series

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
                                                