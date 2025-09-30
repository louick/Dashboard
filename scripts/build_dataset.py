import json, re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "dataset.json"

IDS_FILE = DATA_DIR / "IDS - Municipal (Geral).xlsx"
DESC_FILE = DATA_DIR / "Descritivo de Indicadores - ODS 2025.xlsx"
IDS_ODS4_FILE = DATA_DIR / "IDS-ODS 04.xlsx"  # opcional (reforço p/ ODS04)

# Educação (opcionais)
EM_FILE = DATA_DIR / "divulgacao_ensino_medio_municipios_2023.xlsx"
DOC_ESTADOS_FILE = DATA_DIR / "Adequação Docente_Estados_2023.xlsx"
MICRO_CURSOS_FILE = DATA_DIR / "MICRODADOS_CADASTRO_CURSOS_2023.xlsx"

# Aceita "Construção de Indicadores 03.xlsx" ou "Construção Indicadores 03.xlsx"
ODS_FILE_RE = re.compile(r"Construção(?: de)? Indicadores?\s*(\d{2})\.xlsx", re.I)

# Nomes oficiais dos ODS (PT-BR)
ODS_LABELS = {
    "ODS01": "Erradicação da Pobreza",
    "ODS02": "Fome Zero e Agricultura Sustentável",
    "ODS03": "Saúde e Bem-Estar",
    "ODS04": "Educação de Qualidade",
    "ODS05": "Igualdade de Gênero",
    "ODS06": "Água Potável e Saneamento",
    "ODS07": "Energia Acessível e Limpa",
    "ODS08": "Trabalho Decente e Crescimento Econômico",
    "ODS09": "Indústria, Inovação e Infraestrutura",
    "ODS10": "Redução das Desigualdades",
    "ODS11": "Cidades e Comunidades Sustentáveis",
    "ODS12": "Consumo e Produção Responsáveis",
    "ODS13": "Ação Contra a Mudança do Clima",
    "ODS14": "Vida na Água",
    "ODS15": "Vida Terrestre",
    "ODS16": "Paz, Justiça e Instituições Eficazes",
    "ODS17": "Parcerias e Meios de Implementação",
}

# ---------------- utils ----------------
def normalize_year_cols(cols):
    fixed = []
    for c in cols:
        if isinstance(c, int):
            fixed.append(c); continue
        try:
            f = float(c)
            if f.is_integer() and 1900 <= int(f) <= 2100:
                fixed.append(int(f)); continue
        except Exception:
            pass
        fixed.append(c)
    return fixed

def find_header_row(df):
    for i in range(min(10, len(df))):
        vals = df.iloc[i].astype(str).tolist()
        if any(("Nome" in v) or ("Município" in v) or ("Território" in v) for v in vals) and any(re.search(r"20(1|2)\d", v) for v in vals):
            return i
    return 1

def classify_geo(name: str):
    s = str(name).strip()
    if s.lower() == "pará":
        return "state"
    if s.startswith("RI "):
        return "macro"
    return "municipality"

def safe_str(x):
    return None if (pd.isna(x) or str(x).strip() == "") else str(x).strip()

def coerce_number(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == ""):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if s.lower() in {"-", "—", "–", "na", "n/a", "nan", "none"}:
        return None
    s = s.replace("Meta =", "").replace("Meta", "").replace("=", " ").strip()
    s = s.replace("%", "").strip()
    if re.search(r"\d+,\d+$", s):
        s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^0-9eE\.\-+]", "", s)
    if s in {"", ".", "-"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def extract_code_from_text(txt: str) -> Optional[str]:
    """Extrai códigos tipo '1.1', '2.3.1' ou '01' do nome da aba."""
    if not txt: return None
    s = str(txt)
    m = re.search(r"(\d+(?:\.\d+){1,2})", s)  # 1.1 / 2.3 / 3.1.1
    if m: return m.group(1)
    m2 = re.search(r"\b(\d{1,2})\b", s)       # 01 / 1
    if m2: return m2.group(1).lstrip("0") or "0"
    return None

# ------------- dicionário de indicadores -------------
def load_indicator_dictionary() -> Dict[str, Dict[str, dict]]:
    """
    Lê o(s) arquivo(s) descritivos e retorna:
      dict[ods_key][code] = {"title":..., "desc":..., "unit":..., "source":...}
    Aceita variações de colunas; varre todas as abas, tenta achar algo como:
      - 'ODS' / 'Objetivo'  (para pegar o número do ODS)
      - 'Indicador' / 'Código' / 'Cod' (para o código)
      - 'Nome' / 'Título' / 'Título do Indicador'
      - 'Descrição' / 'Definição'
      - 'Unidade' / 'Unid'
      - 'Fonte'
    """
    out: Dict[str, Dict[str, dict]] = {}

    def harvest_df(df):
        # normaliza nomes
        cols = {c: str(c).strip().lower() for c in df.columns}
        def col_like(*tokens):
            for c in df.columns:
                s = cols[c]
                if all(t in s for t in tokens):
                    return c
            return None

        col_ods  = col_like("ods") or col_like("objetivo")
        col_code = col_like("indicador") or col_like("cód") or col_like("codigo") or col_like("código")
        col_name = col_like("título") or col_like("titulo") or col_like("nome")
        col_desc = col_like("descr") or col_like("defini")
        col_unit = col_like("unid")
        col_src  = col_like("fonte")

        for _, r in df.iterrows():
            ods_val = safe_str(r.get(col_ods)) if col_ods else None
            code_val = safe_str(r.get(col_code)) if col_code else None
            name_val = safe_str(r.get(col_name)) if col_name else None
            desc_val = safe_str(r.get(col_desc)) if col_desc else None
            unit_val = safe_str(r.get(col_unit)) if col_unit else None
            src_val  = safe_str(r.get(col_src))  if col_src  else None

            # tentar extrair número do ODS a partir de "Objetivo 04" ou "ODS 4"
            ods_key = None
            if ods_val:
                m = re.search(r"(\d{1,2})", ods_val)
                if m:
                    ods_key = f"ODS{int(m.group(1)):02d}"

            # aceita linhas sem ODS explícito (vamos popular depois se houver 'code' e 'name')
            if not code_val and name_val:
                # procurar código dentro do nome
                code_val = extract_code_from_text(name_val)

            if not (code_val and (ods_key or name_val or desc_val)):
                continue

            # se não há ods_key, tenta inferir pelo prefixo do código (ex.: "4.1.1" -> ODS04)
            if not ods_key and code_val and re.match(r"(\d+)(?:\..*)?$", code_val):
                n = int(re.match(r"(\d+)", code_val).group(1))
                ods_key = f"ODS{n:02d}"

            if not ods_key:
                continue

            out.setdefault(ods_key, {})
            out[ods_key][code_val] = {
                "title": name_val or f"Indicador {code_val}",
                "desc": desc_val or "",
                "unit": unit_val or "",
                "source": src_val or ""
            }

    # 1) Descritivo geral
    if DESC_FILE.exists():
        try:
            xls = pd.ExcelFile(DESC_FILE)
            for sh in xls.sheet_names:
                df = pd.read_excel(DESC_FILE, sheet_name=sh)
                harvest_df(df)
            print(f"[INFO] Dicionário: carregado de '{DESC_FILE.name}'")
        except Exception as ex:
            print(f"[WARN] Não consegui ler {DESC_FILE.name}: {ex}")

    # 2) Planilha específica ODS 04 (se existir)
    if IDS_ODS4_FILE.exists():
        try:
            xls = pd.ExcelFile(IDS_ODS4_FILE)
            for sh in xls.sheet_names:
                df = pd.read_excel(IDS_ODS4_FILE, sheet_name=sh)
                harvest_df(df)
            print(f"[INFO] Dicionário: reforço de '{IDS_ODS4_FILE.name}'")
        except Exception as ex:
            print(f"[WARN] Não consegui ler {IDS_ODS4_FILE.name}: {ex}")

    return out

# ------------- IDS overview (geos + índices agregados) -------------
def load_ids_overview() -> Dict[str, Any]:
    xls = pd.ExcelFile(IDS_FILE)
    plan = "Plan1" if "Plan1" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(IDS_FILE, sheet_name=plan)
    df.columns = [str(c).strip() for c in df.columns]

    name_col = next((c for c in df.columns if "Nome" in c or "Território" in c), None)
    if not name_col:
        raise RuntimeError("Não encontrei coluna de nome/território no IDS - Municipal (Geral).xlsx")
    macro_col = next((c for c in df.columns if "R. Integ" in c), None)

    ods_cols = [c for c in df.columns if re.fullmatch(r"ODS\s*\d{1,2}", str(c))]
    ids_col  = next((c for c in df.columns if str(c).strip().upper() == "IDS"), None)

    geos, seen = [], set()
    geo_macro_map = {}
    for _, r in df.iterrows():
        nm = safe_str(r.get(name_col))
        if not nm: continue
        kind = classify_geo(nm)
        gid = nm
        if gid not in seen:
            geos.append({"id": gid, "name": gid, "type": kind, "parent": "Pará" if kind == "macro" else None})
            seen.add(gid)
        if macro_col and kind == "municipality":
            par = safe_str(r.get(macro_col))
            if par:
                geo_macro_map[gid] = par

    overview_records = []
    for _, r in df.iterrows():
        nm = safe_str(r.get(name_col))
        if not nm: continue
        for oc in ods_cols:
            v = coerce_number(r.get(oc, None))
            if v is not None:
                n = int(re.findall(r"\d+", oc)[0])
                overview_records.append({"geo_id": nm, "ods": f"ODS{n:02d}", "value": float(v)})
        if ids_col:
            v = coerce_number(r.get(ids_col, None))
            if v is not None:
                overview_records.append({"geo_id": nm, "ods": "IDS", "value": float(v)})

    return {
        "geos": geos,
        "geo_macro_map": geo_macro_map,
        "overview": overview_records,
    }

# ------------- Construção de Indicadores (series + componentes) -------------
def parse_construcao_files(geo_macro_map: Dict[str, str], dict_ind: Dict[str, Dict[str, dict]]):
    values = []
    components: Dict[str, List[Dict[str, str]]] = {}
    years_all = set()
    files = sorted(p for p in DATA_DIR.glob("*.xlsx") if ODS_FILE_RE.match(p.name))
    print(f"[INFO] Construção arquivos: {[p.name for p in files]}")
    for p in files:
        nn = int(ODS_FILE_RE.match(p.name).group(1))
        ods_key = f"ODS{nn:02d}"
        xls = pd.ExcelFile(p)
        comps_for_this = []
        for sheet in xls.sheet_names:
            raw = pd.read_excel(p, sheet_name=sheet, header=None)
            hdr = find_header_row(raw)
            header = list(raw.iloc[hdr])
            tmp = raw.iloc[hdr+1:].copy()
            tmp.columns = header
            tmp.columns = normalize_year_cols(tmp.columns)

            ren = {}
            for c in tmp.columns:
                s = str(c).strip()
                if s.lower().startswith("nome"): ren[c] = "Nome_Município"
                elif "código ibge 01" in s.lower(): ren[c] = "Código IBGE 01"
                elif "código ibge 02" in s.lower(): ren[c] = "Código IBGE 02"
                elif s.lower().startswith("r. integ"): ren[c] = "R. Integ."
            tmp = tmp.rename(columns=ren)

            if "Nome_Município" in tmp.columns and "R. Integ." in tmp.columns:
                for _, r in tmp[["Nome_Município", "R. Integ."]].dropna().iterrows():
                    geo_macro_map[str(r["Nome_Município"]).strip()] = str(r["R. Integ."]).strip()

            year_cols = [c for c in tmp.columns if isinstance(c, int) and 1900 <= c <= 2100]
            years_all.update(year_cols)

            # ------ metadados do componente ------
            comp_id = re.sub(r"\W+", "", sheet).lower()
            code_guess = extract_code_from_text(sheet)  # tenta "1.1", "2.3.1" etc
            meta = dict_ind.get(ods_key, {}).get(code_guess or "", {}) if dict_ind else {}
            comp_meta = {
                "id": comp_id,
                "code": code_guess or "",
                "label": meta.get("title") or sheet,   # título legível
                "unit": meta.get("unit", ""),
                "desc": meta.get("desc", ""),
                "source": meta.get("source", "")
            }
            comps_for_this.append(comp_meta)

            base = tmp.copy()
            if "Nome_Município" in base.columns:
                base = base[base["Nome_Município"].notna()]

            for _, row in base.iterrows():
                nm = safe_str(row.get("Nome_Município"))
                if not nm: continue
                kind = classify_geo(nm)
                if kind not in ("state", "macro", "municipality"): continue
                gid = nm
                for y in year_cols:
                    v = coerce_number(row.get(y, None))
                    if v is None:
                        continue
                    values.append({
                        "ods": ods_key,
                        "component_id": comp_id,
                        "component_label": comp_meta["label"],
                        "unit": comp_meta["unit"],
                        "geo_id": gid,
                        "year": int(y),
                        "value": float(v)
                    })
        components[ods_key] = comps_for_this

    years_sorted = sorted([int(y) for y in years_all])
    if len(years_sorted) >= 3:
        years_sorted = years_sorted[-3:]
    return components, values, years_sorted

# ------------- Educação (opcional) -------------
def try_load_educacao(geo_macro_map: Dict[str, str]):
    ed = {"metrics": {}, "years": []}
    if EM_FILE.exists():
        try:
            em = pd.read_excel(EM_FILE)
            name_col = next((c for c in em.columns if "Munic" in str(c) or "Município" in str(c)), None)
            code_col = next((c for c in em.columns if "IBGE" in str(c)), None)
            year = 2023
            if name_col:
                for _, r in em.iterrows():
                    nm = safe_str(r.get(name_col))
                    if not nm: continue
                    for c in em.columns:
                        if c == name_col or c == code_col: continue
                        v = coerce_number(r.get(c, None))
                        if v is None: continue
                        key = f"EM::{c}"
                        ed["metrics"].setdefault(key, {})
                        ed["metrics"][key][nm] = float(v)
            ed["years"] = sorted(set(ed["years"] + [year]))
            print("[INFO] Educação: Ensino Médio carregado.")
        except Exception as ex:
            print(f"[WARN] Ensino Médio não pôde ser lido: {ex}")

    if DOC_ESTADOS_FILE.exists():
        try:
            ad = pd.read_excel(DOC_ESTADOS_FILE)
            name_col = next((c for c in ad.columns if "UF" in str(c) or "Estado" in str(c)), None)
            if name_col:
                for _, r in ad.iterrows():
                    uf = safe_str(r.get(name_col))
                    if not uf: continue
                    if uf.upper() in ("PA", "PARÁ", "PARA", "Pará", "Para"):
                        for c in ad.columns:
                            if c == name_col: continue
                            v = coerce_number(r.get(c, None))
                            if v is None: continue
                            key = f"ADEQ::{c}"
                            ed["metrics"].setdefault(key, {})
                            ed["metrics"][key]["Pará"] = float(v)
                print("[INFO] Educação: Adequação Docente carregado.")
        except Exception as ex:
            print(f"[WARN] Adequação Docente não pôde ser lido: {ex}")

    if MICRO_CURSOS_FILE.exists():
        try:
            mc = pd.read_excel(MICRO_CURSOS_FILE, nrows=200000)
            mun_col = next((c for c in mc.columns if "munic" in str(c).lower()), None)
            if mun_col:
                counts = mc.groupby(mun_col).size().reset_index(name="qtde_cursos")
                for _, r in counts.iterrows():
                    nm = safe_str(r[mun_col])
                    if not nm: continue
                    key = "SUP::Qtde cursos cadastrados"
                    ed["metrics"].setdefault(key, {})
                    ed["metrics"][key][nm] = float(r["qtde_cursos"])
                print("[INFO] Educação: Microdados cursos agregado (contagem por município).")
        except Exception as ex:
            print(f"[WARN] Microdados cursos não pôde ser lido: {ex}")

    return ed

# ------------- main -------------
def main():
    print("[INFO] Montando dicionário de indicadores…")
    dict_ind = load_indicator_dictionary()

    print("[INFO] Lendo IDS - Municipal (Geral)…")
    ids_over = load_ids_overview()
    geos = ids_over["geos"]
    geo_macro_map = ids_over["geo_macro_map"]
    overview = ids_over["overview"]

    print("[INFO] Varendo Construção de Indicadores…")
    components, values, years_sorted = parse_construcao_files(geo_macro_map, dict_ind)

    ods_all = sorted(list(components.keys()))
    print("[INFO] Tentando agregar Educação…")
    educ = try_load_educacao(geo_macro_map)

    ods_block = {}
    for k in ods_all:
        pretty = ODS_LABELS.get(k, f"ODS {k[-2:]}")
        ods_block[k] = {"label": f"ODS {k[-2:]} — {pretty}", "components": components.get(k, [])}

    dataset = {
        "years": years_sorted,
        "geos": geos,
        "ods": ods_block,
        "values": values,
        "overview": overview,
        "education": educ
    }

    OUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Gerado {OUT_PATH} | geos={len(geos)} | ODS={len(ods_all)} | séries={len(values)} | overview={len(overview)}")
    if educ and educ.get("metrics"):
        print(f"[OK] Educação: {len(educ['metrics'])} métricas agregadas.")

if __name__ == "__main__":
    main()
