#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# ==== Paths ====
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "dataset.json"

# Pastas por ano (ex.: data/ODS 2023, ODS 2024, ODS 2025)
YEAR_DIR_RE = re.compile(r"^ODS\s*20\d{2}$", re.I)
ODS_DIR_RE  = re.compile(r"^ODS\s*(\d{1,2})$", re.I)

# Aceita nomes "Construção de Indicadores 03.xlsx" etc
FLAT_CONSTR_RE = re.compile(r"Construção(?:\s+de)?\s+Indicadores?\s*(\d{1,2})\.xlsx", re.I)

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

# ================= utils =================
def safe_str(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    return s or None

def coerce_number(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().lower()
    if s in {"-", "—", "–", "na", "n/a", "nan", "none"}:
        return None
    s = s.replace("meta =", "").replace("meta", "").replace("=", " ").strip()
    s = s.replace("%", "").strip()
    # 1.234,56
    if re.search(r"\d+,\d+$", s, re.I):
        s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^0-9eE\.\-+]", "", s)
    if s in {"", ".", "-"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _dedup_names(names: List[Any]) -> List[Any]:
    """
    Deduplica mantendo a ordem. Se repetir, anexa sufixo _2, _3, ...
    Preserva tipo int quando for ano único; quando duplicado vira string com sufixo.
    """
    out = []
    counts: Dict[str, int] = {}
    for n in names:
        key = str(n)
        if key not in counts:
            counts[key] = 1
            out.append(n)
        else:
            counts[key] += 1
            out.append(f"{key}_{counts[key]}")
    return out

def normalize_year_cols(cols):
    """Converte cabeçalhos ano '2019', '2020.0', 2019.0 -> 2019 (int) quando possível, e deduplica nomes."""
    fixed: List[Any] = []
    for c in cols:
        # já é inteiro plausível de ano
        if isinstance(c, (int, np.integer)) and 1900 <= int(c) <= 2100:
            fixed.append(int(c))
            continue
        # tenta converter float ou string de ano
        try:
            if c is not None and not (isinstance(c, float) and pd.isna(c)):
                s = str(c).strip()
                if re.fullmatch(r"20\d{2}", s):
                    fixed.append(int(s))
                    continue
                f = float(s.replace(",", ".")) if s else None
                if f is not None and np.isfinite(f) and f.is_integer() and 1900 <= int(f) <= 2100:
                    fixed.append(int(f))
                    continue
        except Exception:
            pass
        fixed.append(c)
    # dedup final (evita crash do pandas em colunas repetidas)
    return _dedup_names(fixed)

def classify_geo(name: str):
    s = str(name).strip()
    if s.lower() == "pará":
        return "state"
    if s.startswith("RI "):
        return "macro"
    return "municipality"

def extract_code_from_text(txt: str) -> Optional[str]:
    if not txt: return None
    s = str(txt)
    m = re.search(r"(\d+(?:\.\d+){1,2})", s)  # 1.1 / 2.3 / 3.1.1
    if m: return m.group(1)
    m2 = re.search(r"\b(\d{1,2})\b", s)       # 01 / 1
    if m2: return m2.group(1).lstrip("0") or "0"
    return None

# --------- header detection (ROBUSTO) ----------
def find_header_row(df: pd.DataFrame) -> Optional[int]:
    """
    Tenta achar a linha de cabeçalho.
    Critérios: presença de colunas 'Município/Nome/Território/R. Integ'
    e 2 ou mais células com ano (20xx).
    Ignora NaN de forma segura.
    """
    if df is None or df.empty:
        return None

    def has_year_like(vals) -> bool:
        n = 0
        for v in vals:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            try:
                if isinstance(v, (int, np.integer)):
                    iv = int(v)
                    if 1900 <= iv <= 2100:
                        n += 1
                        continue
                s = str(v).strip()
                if re.fullmatch(r"20\d{2}", s):
                    n += 1
                    continue
                if s:
                    fv = float(s.replace(",", "."))
                    if np.isfinite(fv) and fv.is_integer() and 1900 <= int(fv) <= 2100:
                        n += 1
            except Exception:
                continue
        return n >= 2

    # tenta primeiras 25 linhas
    for i in range(min(25, len(df))):
        vals = df.iloc[i].tolist()
        txts = [str(v) for v in vals]
        has_geo = any(re.search(r"(Munic|Nome|Territ|R\.?\s*Integ)", t, re.I) for t in txts)
        if has_geo and has_year_like(vals):
            return i

    # fallback: primeira linha com 2 anos; usa a anterior como header
    for i in range(min(40, len(df))):
        vals = df.iloc[i].tolist()
        if has_year_like(vals):
            return max(0, i - 1)

    return None

# ================= dicionário / overview =================
def load_indicator_dictionary(files: List[Path]) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}

    def harvest_df(df: pd.DataFrame):
        if df is None or df.empty:
            return
        cols = {c: str(c).strip().lower() for c in df.columns}

        def col_like(*tokens):
            for c in df.columns:
                s = cols.get(c, "")
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

            if not code_val and name_val:
                code_val = extract_code_from_text(name_val)

            ods_key = None
            if ods_val:
                m = re.search(r"(\d{1,2})", ods_val)
                if m:
                    ods_key = f"ODS{int(m.group(1)):02d}"
            if not ods_key and code_val and re.match(r"(\d+)", code_val):
                n = int(re.match(r"(\d+)", code_val).group(1))
                ods_key = f"ODS{n:02d}"
            if not ods_key or not code_val:
                continue

            out.setdefault(ods_key, {})
            out[ods_key][code_val] = {
                "title": name_val or f"Indicador {code_val}",
                "desc":  desc_val or "",
                "unit":  unit_val or "",
                "source": src_val or ""
            }

    for f in files:
        try:
            xls = pd.ExcelFile(f)
            for sh in xls.sheet_names:
                df = pd.read_excel(f, sheet_name=sh)
                harvest_df(df)
            print(f"[INFO] Dicionário: {f.name}")
        except Exception as ex:
            print(f"[WARN] Não consegui ler dicionário '{f.name}': {ex}")
    return out


def load_ids_overview(ids_files: List[Path]) -> Tuple[List[dict], Dict[str, str], List[dict]]:
    geos, seen = [], set()
    geo_macro_map: Dict[str, str] = {}
    overview: List[dict] = []

    for f in ids_files:
        try:
            xls = pd.ExcelFile(f)
            plan = xls.sheet_names[0]
            df = pd.read_excel(f, sheet_name=plan)
            df.columns = [str(c).strip() for c in df.columns]

            name_col = next((c for c in df.columns if "Nome" in c or "Território" in c), None)
            if not name_col:
                print(f"[WARN] IDS overview sem coluna de nome em {f.name}")
                continue
            macro_col = next((c for c in df.columns if "R. Integ" in c), None)

            ods_cols = [c for c in df.columns if re.fullmatch(r"ODS\s*\d{1,2}", str(c))]
            ids_col  = next((c for c in df.columns if str(c).strip().upper() == "IDS"), None)

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

            for _, r in df.iterrows():
                nm = safe_str(r.get(name_col))
                if not nm: continue
                for oc in ods_cols:
                    v = coerce_number(r.get(oc, None))
                    if v is not None:
                        n = int(re.findall(r"\d+", oc)[0])
                        overview.append({"geo_id": nm, "ods": f"ODS{n:02d}", "value": float(v)})
                if ids_col:
                    v = coerce_number(r.get(ids_col, None))
                    if v is not None:
                        overview.append({"geo_id": nm, "ods": "IDS", "value": float(v)})

            print(f"[INFO] Overview: {f.name}")
        except Exception as ex:
            print(f"[WARN] Falha em overview '{f.name}': {ex}")

    return geos, geo_macro_map, overview


# ================= coleta de arquivos =================
def collect_sources_by_year() -> Dict[str, Dict[str, List[Path]]]:
    """
    Retorna por ano: {'2023': {'ods_files': [...], 'dict_files': [...], 'ids_files': [...]}, ...}
    Busca dentro de 'data/ODS 2023/ODS 01/*.xlsx' etc e também variações soltas.
    """
    acc: Dict[str, Dict[str, List[Path]]] = {}
    for year_dir in sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and YEAR_DIR_RE.match(p.name)]):
        year = re.findall(r"(20\d{2})", year_dir.name)[0]
        acc.setdefault(year, {"ods_files": [], "dict_files": [], "ids_files": []})

        # dicionário (descrições)
        for f in year_dir.glob("**/*.xlsx"):
            low = f.name.lower()
            if "descritivo" in low and "indicador" in low:
                acc[year]["dict_files"].append(f)

        # overview IDS
        for f in year_dir.glob("**/*.xlsx"):
            if "ids" in f.name.lower() and "municipal" in f.name.lower():
                acc[year]["ids_files"].append(f)

        # arquivos de construção por ODS
        for ods_dir in [p for p in year_dir.iterdir() if p.is_dir() and ODS_DIR_RE.match(p.name)]:
            for f in ods_dir.glob("*.xlsx"):
                acc[year]["ods_files"].append(f)

        # também aceita arquivos soltos “Construção de Indicadores 03.xlsx”
        for f in year_dir.glob("*.xlsx"):
            if FLAT_CONSTR_RE.match(f.name):
                acc[year]["ods_files"].append(f)

    return acc


# ================= parse das planilhas de componentes =================
def parse_components_and_values(
    xlsx_files: List[Path],
    geo_macro_map: Dict[str, str],
    dict_ind: Dict[str, Dict[str, dict]],
) -> Tuple[Dict[str, List[dict]], List[dict], List[int]]:
    components: Dict[str, List[dict]] = {}
    values: List[dict] = []
    years_all: set = set()

    def _ods_key_from_path(p: Path) -> Optional[str]:
        # 1) pela pasta "ODS 11"
        for part in p.parts:
            m = ODS_DIR_RE.match(part)
            if m:
                return f"ODS{int(m.group(1)):02d}"
        # 2) pelo nome "Construção de Indicadores 11.xlsx"
        m2 = FLAT_CONSTR_RE.match(p.name)
        if m2:
            return f"ODS{int(m2.group(1)):02d}"
        return None

    def _parse_file_into(ods_key: str, p: Path):
        nonlocal components, values, years_all

        try:
            xls = pd.ExcelFile(p)
        except Exception as ex:
            print(f"[WARN] Não consegui abrir {p.name}: {ex}")
            return

        comps_for_this_file: List[dict] = []

        for sheet in xls.sheet_names:
            try:
                raw = pd.read_excel(p, sheet_name=sheet, header=None)
            except Exception as ex:
                print(f"[WARN] {p.name} / '{sheet}': não consegui ler ({ex}); pulando.")
                continue

            # limpar linhas totalmente vazias
            raw = raw.dropna(how="all")
            if raw.empty:
                print(f"[INFO] {p.name} / '{sheet}': aba vazia; pulando.")
                continue

            hdr = find_header_row(raw)
            if hdr is None or hdr >= len(raw):
                print(f"[WARN] {p.name} / '{sheet}': não encontrei cabeçalho reconhecível; pulando.")
                continue

            header = list(raw.iloc[hdr].fillna(""))
            tmp = raw.iloc[hdr + 1:].copy()
            tmp.columns = normalize_year_cols(header)

            # renomes usuais
            ren = {}
            for c in tmp.columns:
                s = str(c).strip().lower()
                if s.startswith("nome") or "município" in s:
                    ren[c] = "Nome_Município"
                elif s.startswith("r. integ"):
                    ren[c] = "R. Integ."
                elif "código ibge 01" in s:
                    ren[c] = "Código IBGE 01"
                elif "código ibge 02" in s:
                    ren[c] = "Código IBGE 02"
            tmp = tmp.rename(columns=ren)

            # Mapa macro do município
            if "Nome_Município" in tmp.columns and "R. Integ." in tmp.columns:
                for _, r in tmp[["Nome_Município", "R. Integ."]].dropna().iterrows():
                    nm = safe_str(r["Nome_Município"])
                    mac = safe_str(r["R. Integ."])
                    if nm and mac:
                        geo_macro_map[nm] = mac

            # anos
            year_cols = [c for c in tmp.columns if isinstance(c, int) and 1900 <= c <= 2100]
            if not year_cols:
                print(f"[WARN] {p.name} / '{sheet}': sem colunas de ano; pulando.")
                continue
            years_all.update(year_cols)

            # metadados do componente
            comp_id = re.sub(r"\W+", "", sheet).lower()
            code_guess = extract_code_from_text(sheet)
            meta = dict_ind.get(ods_key, {}).get(code_guess or "", {}) if dict_ind else {}

            comp_meta = {
                "id": comp_id,
                "code": code_guess or "",
                "label": meta.get("title") or sheet,
                "unit":  meta.get("unit", ""),
                "desc":  meta.get("desc", ""),
                "source": meta.get("source", ""),
            }
            comps_for_this_file.append(comp_meta)

            base = tmp.copy()
            if "Nome_Município" in base.columns:
                base = base[base["Nome_Município"].notna()]

            # valores
            for _, row in base.iterrows():
                nm = safe_str(row.get("Nome_Município"))
                if not nm:
                    continue
                kind = classify_geo(nm)
                if kind not in ("state", "macro", "municipality"):
                    continue
                gid = nm
                for y in year_cols:
                    v = coerce_number(row.get(y))
                    if v is None:
                        continue
                    values.append({
                        "ods": ods_key,
                        "component_id": comp_id,
                        "component_label": comp_meta["label"],
                        "unit": comp_meta["unit"],
                        "geo_id": gid,
                        "year": int(y),
                        "value": float(v),
                    })

        if comps_for_this_file:
            components.setdefault(ods_key, [])
            components[ods_key].extend(comps_for_this_file)
            print(f"[INFO] {ods_key}: parseado de {p.name}")

    # ---- Itera arquivos
    for p in xlsx_files:
        ods_key = _ods_key_from_path(p)
        if not ods_key:
            continue
        _parse_file_into(ods_key, p)

    years_sorted = sorted({int(y) for y in years_all})
    return components, values, years_sorted


# ================= main =================
def main():
    # Coleta por ano
    by_year = collect_sources_by_year()
    if not by_year:
        raise RuntimeError("Não encontrei pastas 'ODS 20xx' dentro de data/. Verifique a estrutura.")

    all_geos: List[dict] = []
    all_overview: List[dict] = []
    global_geo_macro_map: Dict[str, str] = {}
    all_components: Dict[str, List[dict]] = {}
    all_values: List[dict] = []
    all_years: set = set()

    for year, packs in by_year.items():
        print(f"\n==== Processando {year} ====")
        dict_ind = load_indicator_dictionary(packs["dict_files"])
        geos, geo_macro_map, overview = load_ids_overview(packs["ids_files"])
        # Merge geos e mapas
        if geos:
            existing = {g["id"] for g in all_geos}
            for g in geos:
                if g["id"] not in existing:
                    all_geos.append(g)
                    existing.add(g["id"])
        global_geo_macro_map.update(geo_macro_map)
        all_overview.extend(overview)

        comps, vals, years_sorted = parse_components_and_values(
            packs["ods_files"], global_geo_macro_map, dict_ind
        )
        # merge
        for k, lst in comps.items():
            all_components.setdefault(k, [])
            all_components[k].extend(lst)
        all_values.extend(vals)
        all_years.update(years_sorted)

    # Bloco ODS
    ods_block = {}
    for i in range(1, 18):
        k = f"ODS{i:02d}"
        pretty = ODS_LABELS.get(k, f"ODS {i:02d}")
        ods_block[k] = {
            "label": f"ODS {i:02d} — {pretty}",
            "components": all_components.get(k, []),
        }

    dataset = {
        "years": sorted(all_years),
        "geos": all_geos,
        "ods": ods_block,
        "values": all_values,
        "overview": all_overview,
        "macro_of_muni": global_geo_macro_map,
    }

    OUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n[OK] Gerado {OUT_PATH}\n"
        f"     geos={len(all_geos)}  ODS={len(ods_block)}  séries={len(all_values)}  years={sorted(all_years)}"
    )


if __name__ == "__main__":
    main()
