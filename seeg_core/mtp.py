from __future__ import annotations
import json
import ast
from typing import Any, Dict
import re
from pathlib import Path
import unicodedata

from .db import list_mtp_questions, list_mtp_dimension_meta


def parse_mtp_answers(raw: Any) -> Dict[str, Any]:
    """Parse heterogeneous MTP answers (json, python-literal, dict, str) into a dict.
    Returns empty dict on failure.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    s = str(raw)
    # try json
    try:
        return json.loads(s)
    except Exception:
        pass
    # try python literal
    try:
        val = ast.literal_eval(s)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def mtp_to_text(d: Dict[str, Any]) -> str:
    """Flatten parsed MTP dict into a readable single text (metier, talent, paradigme)."""
    if not isinstance(d, dict) or not d:
        return ""
    parts = []
    for k in ("metier", "talent", "paradigme"):
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            parts.extend([str(x) for x in v if str(x).strip()])
        elif isinstance(v, dict):
            for _, vv in v.items():
                if isinstance(vv, list):
                    parts.extend([str(x) for x in vv if str(x).strip()])
                elif vv is not None and str(vv).strip():
                    parts.append(str(vv))
        elif str(v).strip():
            parts.append(str(v))
    return "\n".join([p for p in parts if p.strip()])


def _extract_keywords(text: str) -> set[str]:
    if not text:
        return set()
    # simple keyword set: lowercase words of length >= 3
    toks = re.findall(r"[a-zA-ZÀ-ÿ']{3,}", str(text).lower())
    return set(toks)


def _normalize_fr(s: str | None) -> str:
    """Normalize french text: lowercase, strip, remove diacritics, collapse spaces.

    Keeps spaces for substring matching of multi-word indicators.
    """
    if not s:
        return ""
    s = str(s).lower().strip()
    # remove diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # unify apostrophes and dashes
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    # collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s

def compute_mtp_scores(conn, poste: str | None, answer_text: str | None) -> Dict[str, Any]:
    """Compute simple MTP coverage scores per dimension and overall (0-100).

    - Loads indicators for the given poste from SQLite (if available).
    - Scores are based on the proportion of indicator keywords found in the answer_text.

    Returns dict: { 'metier': x, 'talent': y, 'paradigme': z, 'overall': o, 'debug': {...} }
    """
    out = {"metier": None, "talent": None, "paradigme": None, "overall": None, "debug": {}}
    if not poste:
        out["debug"]["reason"] = "missing_poste"
        return out
    if not answer_text or not str(answer_text).strip():
        out["debug"]["reason"] = "missing_answer_text"
        return out

    # 1) Try SQLite meta
    try:
        metas = list_mtp_dimension_meta(conn) if conn is not None else []
    except Exception:
        metas = []
    metas_by_dim = {}
    for m in metas:
        if str(m.get("poste", "")).strip().lower() == str(poste).strip().lower():
            metas_by_dim[m.get("dimension")] = m.get("indicators") or m.get("indicateurs") or m.get("indicator")

    # 2) Fallback: load from scripts/mtp_questions.json if nothing from SQLite
    if not metas_by_dim:
        try:
            base = Path(__file__).resolve().parents[1]  # project root (dashboard)
            json_path = base / "scripts" / "mtp_questions.json"
            if json_path.exists():
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                # fuzzy match poste entry (with aliasing and light plural handling)
                target = _normalize_fr(poste)
                # Known aliases mapping job titles to JSON 'poste'
                aliases = {
                    "directeur des systemes d'information": "Direction des Systèmes d'Informations",
                    "directeur des systemes d informations": "Direction des Systèmes d'Informations",
                    "directeur systemes d'information": "Direction des Systèmes d'Informations",
                    "directeur systemes dinformation": "Direction des Systèmes d'Informations",
                    "directeur audit & controle interne": "Direction de l'Audit & Contrôle Interne",
                    "directeur juridique, communication & rse": "Direction Juridique, Communication & RSE",
                    "directeur exploitation electricite": "Direction Exploitation Electricité",
                    "directeur technique electricite": "Direction Technique Electricité",
                    "directeur exploitation eau": "Direction Exploitation Eau",
                    "directeur technique eau": "Direction Technique Eau",
                    "directeur des moyens generaux": "Direction des Moyens Généraux",
                    "directeur commercial et recouvrement": "Direction Commerciale & Recouvrement",
                    "directeur finances & comptabilite": "Direction Finances & Comptabilité",
                    "coordonnateur des regions": "Coordination Régions",
                }
                if target in aliases:
                    forced = aliases[target]
                else:
                    forced = None
                best_item = None
                best_score = 0.0
                def _tokens(s: str) -> set[str]:
                    toks = re.findall(r"[a-zA-ZÀ-ÿ']{3,}", _normalize_fr(s))
                    base = set()
                    for t in toks:
                        if len(t) > 4 and t.endswith("s"):
                            base.add(t[:-1])
                        base.add(t)
                    return base
                for item in data.get("postes", []):
                    p = item.get("poste")
                    if not p:
                        continue
                    pn = _normalize_fr(p)
                    score = 0.0
                    if forced and pn == _normalize_fr(forced):
                        score = 1.0
                    elif pn == target:
                        score = 1.0
                    elif pn in target or target in pn:
                        score = 0.9
                    else:
                        a = _tokens(pn)
                        b = _tokens(target)
                        inter = len(a & b)
                        uni = len(a | b) or 1
                        score = inter / uni
                    if score > best_score:
                        best_score = score
                        best_item = item
                if best_item and best_score >= 0.34:  # threshold tolerant
                    # metiers vs metier naming
                    for dim_key, dim_name in (("metiers","metier"),("talent","talent"),("paradigme","paradigme")):
                        dim_val = best_item.get(dim_key) or {}
                        if isinstance(dim_val, dict):
                            indic = dim_val.get("indicateurs")
                            if indic:
                                metas_by_dim[dim_name] = indic
                    out["debug"]["matched_poste"] = best_item.get("poste")
                    out["debug"]["match_score"] = round(best_score, 3)
                # if still empty, leave empty
        except Exception:
            pass

    # Build indicator lists per dimension
    # Accept either a list directly or a string with separators (comma/semicolon/newline/et/\/dash)
    def _to_indicator_list(v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            parts = [str(x) for x in v if x is not None and str(x).strip()]
        else:
            s = str(v)
            # also split on french conjunctions ' et ' and slashes/dashes
            parts = re.split(r"[,;\n/]|\bet\b|\s-\s", s, flags=re.IGNORECASE)
        cleaned = []
        for p in parts:
            p = str(p).strip()
            if not p:
                continue
            cleaned.append(p)
        return cleaned

    ind = {
        "metier": _to_indicator_list(metas_by_dim.get("metier")),
        "talent": _to_indicator_list(metas_by_dim.get("talent")),
        "paradigme": _to_indicator_list(metas_by_dim.get("paradigme")),
    }

    # Prepare normalized text for robust matching
    norm_text = _normalize_fr(answer_text)
    kw = _extract_keywords(norm_text)
    dims = {}
    for d, indicators in ind.items():
        if not indicators:
            # If no indicators provided for a dimension:
            # - For 'metier', user wants a 0 score instead of None
            # - For others, keep None
            if d == "metier":
                hits = 0
                checked = []
                score = 0
                dims[d] = score
                out["debug"][d] = {"hits": hits, "total": 0, "matched": checked}
            else:
                dims[d] = None
            continue
        hits = 0
        checked = []
        for token in indicators:
            tok_norm = _normalize_fr(token)
            # If multi-word indicator, prefer substring match on normalized text
            is_multi = bool(re.search(r"\s", tok_norm))
            matched = False
            if is_multi and len(tok_norm) >= 5:
                if tok_norm in norm_text:
                    matched = True
            else:
                words = [w for w in re.findall(r"[a-zA-ZÀ-ÿ']+", tok_norm) if len(w) >= 3]
                if words and all((w in kw) for w in words):
                    matched = True
            if matched:
                hits += 1
                checked.append(token)
        score = int(round(100.0 * hits / max(len(indicators), 1)))
        dims[d] = score
        out["debug"][d] = {"hits": hits, "total": len(indicators), "matched": checked}

    # overall: average of available dimension scores
    vals = [v for v in dims.values() if isinstance(v, (int, float))]
    out.update(dims)
    out["overall"] = int(round(sum(vals) / len(vals))) if vals else None
    return out
