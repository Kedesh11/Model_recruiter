#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from typing import Dict
import pandas as pd

from seeg_core.config import get_settings
from seeg_core.db import (
    get_supabase,
    get_sqlite,
    ensure_sqlite_scores_schema,
    upsert_sqlite_score,
)
from seeg_core.mtp import parse_mtp_answers, mtp_to_text
from seeg_core.features import build_candidate_text, compute_job_vectors
from seeg_core.scoring import (
    compute_completeness,
    compute_fit,
    compute_final,
    recommend,
    upsert_score_to_supabase,
)


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

    # Build candidate features/texts
    df_features = build_candidate_text(df_app=df_app, df_app_docs=None, df_profiles=df_profiles)
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
        fit = compute_fit(cand_text, job_text, settings.vision_text)
        final = compute_final(completeness, fit)
        reco = recommend(final)

        identity = identity_map.get(cand_id) if cand_id is not None else None
        identity_full = identity_full_map.get(cand_id) if cand_id is not None else None

        if args.dry_run:
            print({
                "application_id": app_id,
                "candidate_id": cand_id,
                "completeness": completeness,
                "fit": fit,
                "final": final,
                "recommendation": reco,
                "identity_name": identity.get("name") if identity else None,
            })
        else:
            try:
                payload = {
                    "completeness": completeness,
                    "fit": fit,
                    "final": final,
                    "recommendation": reco,
                    "details": {
                        "flags": flags,
                        "weights": {"final": {"completeness": 0.4, "fit": 0.6}},
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
