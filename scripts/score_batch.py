#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from typing import Dict, Optional
import pandas as pd
import re
import requests
from io import BytesIO

from seeg_core.config import get_settings
from seeg_core.db import (
    get_supabase,
    get_sqlite,
    ensure_sqlite_scores_schema,
    upsert_sqlite_score,
)
from seeg_core.mtp import parse_mtp_answers, mtp_to_text, compute_mtp_scores
from seeg_core.features import build_candidate_text, compute_job_vectors
from seeg_core.scoring import (
    compute_completeness,
    compute_fit,
    compute_final,
    recommend,
    upsert_score_to_supabase,
)

# Optional heavy dependencies for document extraction/OCR
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    import docx  # python-docx
except Exception:
    docx = None


def _download_bytes(url: str) -> bytes | None:
    try:
        if not isinstance(url, str) or not url:
            return None
        if not url.startswith("http://") and not url.startswith("https://"):
            return None
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None
    return None


def _ocr_image_bytes(data: bytes, lang: str = "eng+fra") -> str:
    if not data or Image is None or pytesseract is None:
        return ""
    try:
        img = Image.open(BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return pytesseract.image_to_string(img, lang=lang) or ""
    except Exception:
        return ""


def _ocr_pdf_with_fitz(data: bytes, dpi: int = 200, max_pages: int = 20, lang: str = "eng+fra") -> str:
    if not data or fitz is None or pytesseract is None or Image is None:
        return ""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return ""
    texts = []
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            if i > max_pages:
                break
            try:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
                t = _ocr_image_bytes(img_bytes, lang=lang)
                if t and t.strip():
                    texts.append(t)
            except Exception:
                continue
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return "\n\n".join(texts)


def _extract_text_from_pdf(data: bytes) -> str:
    if not data:
        return ""
    if pdf_extract_text is None:
        return _ocr_pdf_with_fitz(data)
    try:
        txt = pdf_extract_text(BytesIO(data)) or ""
    except Exception:
        txt = ""
    if not txt or len(txt.strip()) < 50:
        ocr_txt = _ocr_pdf_with_fitz(data)
        if ocr_txt:
            return ocr_txt
    return txt


def _extract_text_from_docx(data: bytes) -> str:
    if not data or docx is None:
        return ""
    try:
        d = docx.Document(BytesIO(data))
        paras = [p.text for p in d.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras)
    except Exception:
        return ""


def _extract_text_from_txt(data: bytes) -> str:
    if not data:
        return ""
    for enc in ("utf-8", "latin1"):
        try:
            return data.decode(enc, errors="ignore")
        except Exception:
            continue
    return ""


def _summarize_text_simple(text: str, max_chars: int = 2000) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_chars:
        return t
    head = t[: max_chars // 2]
    tail = t[- max_chars // 3 :]
    return head + " … " + tail


def _chunk_text(t: str, max_chars: int = 1800, overlap: int = 200) -> list[str]:
    if not t:
        return []
    s = re.sub(r"\s+", " ", str(t)).strip()
    if len(s) <= max_chars:
        return [s]
    chunks = []
    start = 0
    while start < len(s):
        end = min(len(s), start + max_chars)
        chunks.append(s[start:end])
        if end >= len(s):
            break
        start = max(0, end - overlap)
    return chunks


def _summarize_hierarchical(text: str, chunk_size: int = 1800, overlap: int = 200, max_levels: int = 3) -> str:
    if not text:
        return ""
    chunks = _chunk_text(text, max_chars=chunk_size, overlap=overlap)
    summaries = [_summarize_text_simple(c, max_chars=chunk_size // 2) for c in chunks]
    merged = " \n".join(summaries)
    level = 1
    while len(merged) > chunk_size and level < max_levels:
        level += 1
        chunks = _chunk_text(merged, max_chars=chunk_size, overlap=overlap)
        summaries = [_summarize_text_simple(c, max_chars=chunk_size // 2) for c in chunks]
        merged = " \n".join(summaries)
    return _summarize_text_simple(merged, max_chars=chunk_size)


def build_app_docs_text(df_docs: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns: application_id, text
    Extract text from documents pointed by URLs if text columns absent.
    """
    if df_docs is None or df_docs.empty:
        return pd.DataFrame(columns=["application_id", "text"])
    # If a text-like column exists, just select it
    for c in ["content", "text", "document_text", "body"]:
        if c in df_docs.columns and "application_id" in df_docs.columns:
            return df_docs[["application_id", c]].rename(columns={c: "text"})
    # Else try to extract from URLs/paths
    link_col: Optional[str] = None
    for c in ["link", "url", "public_url", "download_url", "path", "storage_path", "file_path", "filename", "name"]:
        if c in df_docs.columns:
            link_col = c
            break
    if link_col is None or "application_id" not in df_docs.columns:
        return pd.DataFrame(columns=["application_id", "text"])
    rows = []
    for app_id, group in df_docs.groupby("application_id"):
        parts = []
        for _, r in group.iterrows():
            url = str(r.get(link_col, "") or "")
            if not url:
                continue
            data = _download_bytes(url)
            if not data:
                continue
            low = url.lower()
            txt = ""
            if low.endswith(".pdf"):
                txt = _extract_text_from_pdf(data)
            elif any(low.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]):
                txt = _ocr_image_bytes(data)
            elif low.endswith(".docx") or low.endswith(".doc"):
                txt = _extract_text_from_docx(data)
            elif low.endswith(".txt"):
                txt = _extract_text_from_txt(data)
            else:
                txt = _extract_text_from_txt(data)
            if txt and txt.strip():
                parts.append(txt)
        if parts:
            rows.append({"application_id": app_id, "text": "\n\n".join(parts)})
    return pd.DataFrame(rows)


def to_df(rows, name: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.attrs["_name"] = name
    return df


def fetch_table_multi(supabase, candidates: list[str]) -> list[dict]:
    """Try to fetch the first existing table among candidates; return [] if none."""
    for t in candidates:
        try:
            data = supabase.table(t).select("*").execute().data or []
            # If request didn't raise and data is list, consider success (even empty list)
            return data
        except Exception:
            continue
    return []


def _strip_obj_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop all-null columns, drop duplicate rows, strip strings."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = _strip_obj_cols(df)
    # drop columns that are entirely NA or empty strings
    to_drop = []
    for c in df.columns:
        col = df[c]
        if col.isna().all():
            to_drop.append(c)
        elif col.dtype == object and (col.fillna("").astype(str).str.strip() == "").all():
            to_drop.append(c)
    if to_drop:
        df = df.drop(columns=to_drop)
    # drop duplicates using only hashable columns to avoid TypeError on dict/list values
    hashable_cols = []
    for c in df.columns:
        try:
            # determine if all values are hashable
            _ = df[c].map(lambda x: True if (x is None or isinstance(x, (int, float, str, bool, bytes, tuple))) else False)
            # also guard against tuples containing unhashables
            if _.all():
                hashable_cols.append(c)
        except Exception:
            continue
    if hashable_cols:
        df = df.drop_duplicates(subset=hashable_cols)
    return df


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # collapse excessive whitespace
    s = " ".join(s.split())
    return s.strip()


def data_quality_report(df_app: pd.DataFrame, df_docs: pd.DataFrame, df_jobs: pd.DataFrame) -> dict:
    rep = {}
    rep["applications"] = len(df_app)
    rep["jobs"] = len(df_jobs)
    rep["documents"] = len(df_docs)
    # application text availability
    for col in ["cover_letter", "motivation", "cv_text", "resume_text", "mtp_text"]:
        if col in df_app.columns:
            nonempty = (df_app[col].fillna("").astype(str).str.strip() != "").sum()
            rep[f"nonempty_{col}"] = int(nonempty)
    # documents flags presence
    if not df_docs.empty and "document_type" in df_docs.columns:
        rep["doc_types"] = df_docs["document_type"].fillna("unknown").astype(str).str.lower().value_counts().to_dict()
    return rep


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SEEG batch scoring for applications")
    parser.add_argument("--limit", type=int, default=200, help="Max applications to process")
    parser.add_argument("--dry-run", action="store_true", help="Do not upsert, just print")
    parser.add_argument("--sqlite-db", type=str, default=None, help="Path to local SQLite database to store scores (optional)")
    args = parser.parse_args(argv)

    settings = get_settings()
    # Always read from Supabase
    supabase = get_supabase()
    # Optionally write to SQLite
    sqlite_conn = None
    if args.sqlite_db:
        sqlite_conn = get_sqlite(args.sqlite_db)
        ensure_sqlite_scores_schema(sqlite_conn)

    # Fetch data
    applications = fetch_table_multi(supabase, ["applications"])[: args.limit]
    documents = fetch_table_multi(supabase, ["application_documents", "documents", "docs"]) 
    positions = fetch_table_multi(supabase, ["job_offers", "positions", "offers", "jobs"]) 
    profiles = fetch_table_multi(supabase, ["candidate_profiles", "profiles"]) 
    candidates_rows = fetch_table_multi(supabase, ["users", "candidates", "applicants"]) 

    # Optional: mtp_answers table; fallback to applications.mtp_answers if exists
    try:
        mtp_rows = supabase.table("mtp_answers").select("*").execute().data or []
    except Exception:
        mtp_rows = []

    df_app = clean_df(to_df(applications, "applications"))
    df_docs = clean_df(to_df(documents, "documents"))
    df_jobs = clean_df(to_df(positions, "positions"))
    df_profiles = clean_df(to_df(profiles, "candidate_profiles"))
    df_cand = clean_df(to_df(candidates_rows, "candidates"))
    df_mtp = to_df(mtp_rows, "mtp_answers")

    # Derive mtp_text on applications
    # 1) If a single JSON-like column exists
    if "mtp_answers" in df_app.columns:
        df_app["mtp_text"] = df_app["mtp_answers"].apply(parse_mtp_answers).apply(mtp_to_text)
    # 2) Else if a separate mtp_answers table exists
    elif not df_mtp.empty and "application_id" in df_mtp.columns and "raw_text" in df_mtp.columns:
        agg = df_mtp.groupby("application_id")["raw_text"].apply(lambda s: "\n".join([str(x) for x in s if pd.notna(x) and str(x).strip()])).reset_index()
        agg["mtp_text"] = agg["raw_text"].apply(lambda x: mtp_to_text(parse_mtp_answers(x)))
        df_app = df_app.merge(agg[["application_id","mtp_text"]], on="application_id", how="left")
    # 3) Else derive from any mtp_* columns present in applications
    elif any(col.startswith("mtp_") for col in df_app.columns):
        mtp_cols = [c for c in df_app.columns if c.startswith("mtp_")]
        def _concat_mtp_row(r: pd.Series) -> str:
            vals = []
            for c in mtp_cols:
                v = r.get(c)
                if pd.notna(v) and str(v).strip():
                    vals.append(str(v))
            return "\n".join(vals)
        df_app["mtp_text"] = df_app.apply(_concat_mtp_row, axis=1)
    else:
        df_app["mtp_text"] = ""

    # Build aggregated docs text per application (download/OCR if needed)
    df_docs_text = build_app_docs_text(df_docs) if not df_docs.empty else pd.DataFrame(columns=["application_id","text"])
    if not df_docs_text.empty:
        df_docs_text = df_docs_text.rename(columns={"text": "document_text"})
        # merge back into df_docs to provide a text column detectable by build_candidate_text
        df_docs_for_features = df_docs.merge(df_docs_text, on="application_id", how="left")
        # prefer explicit 'document_text' naming; build_candidate_text scans common names including 'content','text','document_text','body'
        df_docs_for_features = df_docs_for_features.rename(columns={"document_text": "document_text"})
    else:
        df_docs_for_features = df_docs

    # Build candidate features/texts (now includes documents + mtp_text)
    df_features = build_candidate_text(df_app=df_app, df_app_docs=df_docs_for_features, df_profiles=df_profiles)
    # Normalize candidate_text and filter too-short texts
    if "candidate_text" in df_features.columns:
        df_features["candidate_text"] = df_features["candidate_text"].fillna("").astype(str).apply(normalize_text)
        df_features = df_features[df_features["candidate_text"].str.len() >= 30]

    # Build identity map from candidates table
    identity_map = {}
    identity_full_map = {}
    if not df_cand.empty:
        def _pick(v):
            return v if pd.notna(v) and str(v).strip() else None
        for _, row in df_cand.iterrows():
            cid = row.get("id")
            if cid is None:
                continue
            fn = _pick(row.get("first_name")) or _pick(row.get("prenom"))
            ln = _pick(row.get("last_name")) or _pick(row.get("nom"))
            name = " ".join([x for x in [fn, ln] if x]) if (fn or ln) else _pick(row.get("name"))
            identity_map[cid] = {
                "id": cid,
                "first_name": fn,
                "last_name": ln,
                "name": name,
                "email": _pick(row.get("email")) or _pick(row.get("mail")),
                "phone": _pick(row.get("phone")) or _pick(row.get("mobile")) or _pick(row.get("telephone")),
                "matricule": _pick(row.get("matricule")),
            }
            # keep full raw candidate row as dict for complete traceability
            try:
                identity_full_map[cid] = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            except Exception:
                identity_full_map[cid] = row.to_dict()

    # Completeness from documents flags
    flags_map: Dict[str, Dict[str, bool]] = {}
    if not df_docs.empty and "application_id" in df_docs.columns:
        # try to normalize doc types
        def _flag_from_doc_type(dt: str) -> str:
            s = str(dt).lower()
            if any(k in s for k in ["cv", "resume"]):
                return "cv"
            if any(k in s for k in ["cover", "lettre", "motivation", "lm"]):
                return "lm"
            if any(k in s for k in ["diploma", "diplome", "degree"]):
                return "diploma"
            if any(k in s for k in ["id", "identity", "cni", "piece"]):
                return "id"
            if "mtp" in s or "test" in s or "question" in s or "reponse" in s:
                return "mtp"
            return "other"

        # choose a source column for type
        type_col = None
        for c in ["document_type","doc_type","type","filename","name","storage_path","path"]:
            if c in df_docs.columns:
                type_col = c
                break
        if type_col is None:
            df_docs["_doc_type_norm"] = "other"
        else:
            df_docs["_doc_type_norm"] = df_docs[type_col].apply(_flag_from_doc_type)

        for app_id, group in df_docs.groupby("application_id"):
            flags = {"cv": False, "lm": False, "diploma": False, "id": False, "mtp": False}
            for _, row in group.iterrows():
                fl = row["_doc_type_norm"]
                if fl in flags:
                    flags[fl] = True
            flags_map[app_id] = flags

    # Compute job vectors and text mapping
    # Prefer job_offers rich text fields when present
    job_text_fields = [
        "title", "description", "requirements", "benefits", "skills",
        "department", "location", "profile",
    ]
    job_text_fields = [c for c in job_text_fields if c in df_jobs.columns] or None
    df_jobs_vec = compute_job_vectors(df_jobs, text_fields=job_text_fields)
    if "job_text" in df_jobs_vec.columns:
        df_jobs_vec["job_text"] = df_jobs_vec["job_text"].fillna("").astype(str).apply(normalize_text)
    job_text_map = {r["id"]: r["job_text"] for _, r in df_jobs_vec.iterrows()} if "id" in df_jobs_vec.columns else {}
    job_title_map = {r["id"]: r.get("title") for _, r in df_jobs_vec.iterrows()} if "id" in df_jobs_vec.columns else {}

    # Iterate and score
    processed = 0
    # Data quality summary (printed once in dry-run)
    if args.dry_run:
        summary = data_quality_report(df_app, df_docs, df_jobs)
        print({"data_quality": summary})

    for _, r in df_features.iterrows():
        app_id = r.get("application_id", r.get("id"))
        job_id = r.get("job_offer_id", r.get("position_id"))
        cand_id = r.get("candidate_id", r.get("user_id"))
        cand_text = r.get("candidate_text", "") or ""
        job_text = job_text_map.get(job_id, "")

        flags = flags_map.get(app_id, {"cv": False, "lm": False, "diploma": False, "id": False, "mtp": bool(r.get("mtp_text"))})
        completeness = compute_completeness(flags)
        # Limiter la taille via résumé hiérarchique pour stabiliser le fit sur gros textes
        try:
            text_for_fit = _summarize_hierarchical(cand_text, chunk_size=4000, overlap=300)
        except Exception:
            text_for_fit = _summarize_text_simple(cand_text, max_chars=4000)
        fit = compute_fit(text_for_fit, job_text, settings.vision_text)
        # MTP sub-score (if we have SQLite indicators and a job title)
        poste_title = job_title_map.get(job_id)
        mtp_text_val = r.get("mtp_text")
        try:
            import math
            if mtp_text_val is None or (isinstance(mtp_text_val, float) and math.isnan(mtp_text_val)):
                mtp_text_val = ""
        except Exception:
            if mtp_text_val is None:
                mtp_text_val = ""
        mtp_scores = compute_mtp_scores(sqlite_conn, poste_title, mtp_text_val) if poste_title else {"overall": None}
        mtp_sub = mtp_scores.get("overall")
        # Blend final score with explicit weights
        # final = 0.25*completeness + 0.60*fit + 0.15*mtp (if mtp available), else 0.25*completeness + 0.75*fit
        if isinstance(mtp_sub, (int, float)):
            final = int(round(0.25 * completeness + 0.60 * fit + 0.15 * mtp_sub))
        else:
            final = int(round(0.25 * completeness + 0.75 * fit))
        reco = recommend(final)

        identity = identity_map.get(cand_id) if cand_id is not None else None
        identity_full = identity_full_map.get(cand_id) if cand_id is not None else None

        if args.dry_run:
            dbg = {
                "application_id": app_id,
                "candidate_id": cand_id,
                "job_id": job_id,
                "job_title": poste_title,
                "completeness": completeness,
                "fit": fit,
                "final": final,
                "recommendation": reco,
                "identity_name": identity.get("name") if identity else None,
                "mtp": mtp_scores,
            }
            # extra debug aids
            if poste_title:
                dbg["mtp_input_len"] = len(mtp_text_val or "")
                if isinstance(mtp_scores, dict):
                    dbg["matched_poste"] = mtp_scores.get("debug", {}).get("matched_poste")
                    dbg["match_score"] = mtp_scores.get("debug", {}).get("match_score")
                    dbg["mtp_reason"] = mtp_scores.get("debug", {}).get("reason")
            print(dbg)
        else:
            try:
                payload = {
                    "completeness": completeness,
                    "fit": fit,
                    "final": final,
                    "recommendation": reco,
                    "details": {
                        "flags": flags,
                        "weights": {"final": {"completeness": 0.25, "fit": 0.60 if isinstance(mtp_sub, (int, float)) else 0.75, "mtp": 0.15 if isinstance(mtp_sub, (int, float)) else 0.0}},
                        "mtp": {"scores": mtp_scores},
                        "identity": identity,
                        "identity_full": identity_full,
                    },
                }
                if sqlite_conn is not None:
                    upsert_sqlite_score(sqlite_conn, app_id, payload)
                else:
                    upsert_score_to_supabase(supabase, app_id, payload)
            except Exception as e:
                print(f"Upsert error for application {app_id}: {e}")
        processed += 1

    print(f"Processed {processed} applications")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
