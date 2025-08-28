from __future__ import annotations
from typing import List, Optional
import pandas as pd
import numpy as np
from .nlp import text_vector


def _safe_text_from_row(row: pd.Series, fields: List[str]) -> str:
    def _to_text(val) -> str:
        # Convert complex types to readable text
        if val is None:
            return ""
        if isinstance(val, (list, tuple, set, np.ndarray)):
            try:
                return "\n".join([str(x) for x in val if x is not None and str(x).strip()])
            except Exception:
                return str(val)
        if isinstance(val, dict):
            try:
                # join key: value lines for readability
                return "\n".join([f"{k}: {v}" for k, v in val.items() if v is not None and str(v).strip()])
            except Exception:
                return str(val)
        return str(val)

    parts: List[str] = []
    for f in fields:
        if f in row.index:
            v = row.get(f)
            try:
                txt = _to_text(v).strip()
                if txt:
                    parts.append(txt)
            except Exception:
                continue
    return "\n".join(parts)


def compute_job_vectors(df_jobs: pd.DataFrame, text_fields: Optional[List[str]] = None) -> pd.DataFrame:
    if text_fields is None:
        text_fields = ["description", "job_description", "title"]
    df = df_jobs.copy()
    df["job_text"] = df.apply(lambda r: _safe_text_from_row(r, [f for f in text_fields if f in df.columns]), axis=1)
    df["job_vector"] = df["job_text"].apply(text_vector)
    return df


def build_candidate_text(
    df_app: pd.DataFrame,
    df_app_docs: Optional[pd.DataFrame],
    df_profiles: Optional[pd.DataFrame],
    extra_text_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return a DataFrame `df_features` containing at minimum:
    - application id (kept as 'id' if present, else 'application_id')
    - candidate identifier (user_id or candidate_id if present)
    - position identifier (job_offer_id or position_id if present)
    - candidate_text: aggregated text built from application/profile/docs/mtp
    Expects df_app to already include a column 'mtp_text' when available.
    """
    df_app = df_app.copy()
    app_id_col = 'application_id' if 'application_id' in df_app.columns else 'id'

    # Select useful columns from applications
    cols_keep = [c for c in [app_id_col, 'user_id', 'candidate_id', 'job_offer_id', 'position_id', 'cover_letter', 'motivation', 'cv_text', 'cv_content', 'resume_text', 'mtp_text'] if c in df_app.columns]
    df_feat = df_app[cols_keep].copy()

    # Merge profile info if available (e.g., age, years_experience)
    if df_profiles is not None and isinstance(df_profiles, pd.DataFrame):
        # try user_id -> profiles.id
        left_key = 'user_id' if 'user_id' in df_feat.columns else None
        right_key = 'id' if 'id' in df_profiles.columns else None
        if left_key and right_key:
            df_feat = df_feat.merge(df_profiles, left_on=left_key, right_on=right_key, how='left', suffixes=("", "_prof"))

    # Aggregate docs text by application if available
    if df_app_docs is not None and isinstance(df_app_docs, pd.DataFrame):
        doc_text_col = None
        for c in ['content', 'text', 'document_text', 'body']:
            if c in df_app_docs.columns:
                doc_text_col = c
                break
        if doc_text_col and 'application_id' in df_app_docs.columns:
            agg = (df_app_docs.groupby('application_id')[doc_text_col]
                   .apply(lambda s: "\n".join([str(x) for x in s if pd.notna(x) and str(x).strip()]))
                   .reset_index()
                   .rename(columns={doc_text_col: 'doc_text_agg'}))
            df_feat = df_feat.merge(agg, left_on=app_id_col, right_on='application_id', how='left')

    # Build candidate_text columns list
    base_cols = ['cover_letter','motivation','cv_text','cv_content','resume_text','mtp_text','doc_text_agg']
    if extra_text_cols:
        base_cols.extend([c for c in extra_text_cols if c in df_feat.columns])

    def concat_text_row(r: pd.Series) -> str:
        def _to_text(val) -> str:
            if val is None:
                return ""
            if isinstance(val, (list, tuple, set, np.ndarray)):
                try:
                    return "\n".join([str(x) for x in val if x is not None and str(x).strip()])
                except Exception:
                    return str(val)
            if isinstance(val, dict):
                try:
                    return "\n".join([f"{k}: {v}" for k, v in val.items() if v is not None and str(v).strip()])
                except Exception:
                    return str(val)
            return str(val)

        parts: List[str] = []
        for c in base_cols:
            if c in r.index:
                v = r.get(c)
                try:
                    txt = _to_text(v).strip()
                    if txt:
                        parts.append(txt)
                except Exception:
                    continue
        return "\n\n".join(parts)

    df_feat['candidate_text'] = df_feat.apply(concat_text_row, axis=1)
    return df_feat
