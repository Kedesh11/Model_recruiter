import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import unicodedata as ud
import requests
from io import BytesIO
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import docx  # python-docx
except Exception:
    docx = None

# Ensure project root is on sys.path to import local packages when running from streamlit_app/
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from seeg_core.db import (
    get_supabase,
    get_sqlite,
    ensure_sqlite_scores_schema,
    ensure_sqlite_mtp_schema,
    list_mtp_questions,
    upsert_mtp_question,
    delete_mtp_question,
    import_mtp_questions_from_json,
)
from seeg_core.config import get_settings
import scripts.score_batch as score_batch
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="SEEG Recrutement", layout="wide")


def _drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are entirely empty (all NA or blank strings)."""
    if df is None or df.empty:
        return df
    to_drop = []
    for c in df.columns:
        col = df[c]
        try:
            if col.isna().all():
                to_drop.append(c)
            elif col.dtype == object and (col.fillna("").astype(str).str.strip() == "").all():
                to_drop.append(c)
        except Exception:
            continue
    if to_drop:
        return df.drop(columns=to_drop)
    return df


@st.cache_data(ttl=600)
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
        pass
    return None


def _extract_text_from_pdf(data: bytes) -> str:
    if not data:
        return ""
    if pdf_extract_text is None:
        # Pas d'extraction texte native, essayer OCR direct
        return _ocr_pdf_with_fitz(data)
    try:
        txt = pdf_extract_text(BytesIO(data)) or ""
    except Exception:
        txt = ""
    # Si texte vide/faible (souvent un scan), tenter OCR
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


def _summarize_text_simple(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_chars:
        return t
    # Heuristique: garder début + fin
    head = t[: max_chars // 2]
    tail = t[- max_chars // 3 :]
    return head + " … " + tail


def _chunk_text(t: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Découpe un texte en morceaux avec chevauchement simple sur nombre de caractères.
    Normalise les espaces pour limiter les coupures inutiles.
    """
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


def _summarize_hierarchical(text: str, chunk_size: int = 1500, overlap: int = 200, max_levels: int = 3) -> str:
    """Résumé hiérarchique: résume chaque chunk, concatène, puis re-résume si trop long.
    Version locale basée sur _summarize_text_simple (pas d'appel API).
    """
    if not text:
        return ""
    # 1er niveau
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


def _embeddings_openai(texts: list[str], api_key: str, model: str = "text-embedding-3-small") -> list[list[float]]:
    if not texts:
        return []
    if OpenAI is None or not api_key:
        return []
    try:
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        return []


def _embeddings_fallback(texts: list[str], dim: int = 256) -> list[list[float]]:
    # Hashing simple type count-vector -> projection fixe par modulo
    vecs = []
    for t in texts or []:
        v = np.zeros(dim, dtype=np.float32)
        if t:
            for tok in str(t).lower().split():
                idx = (hash(tok) % dim)
                v[idx] += 1.0
        n = np.linalg.norm(v)
        vecs.append((v / n).tolist() if n > 0 else v.tolist())
    return vecs


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def ensure_sqlite_rag_schema(conn):
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_index (
                application_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                text TEXT,
                embedding BLOB,
                PRIMARY KEY (application_id, chunk_id)
            )
            """
        )
        conn.commit()
    except Exception:
        pass


def upsert_rag_for_application(conn, application_id: str, full_text: str, chunk_size: int = 1200, overlap: int = 200):
    if not application_id or not full_text:
        return
    ensure_sqlite_rag_schema(conn)
    chunks = _chunk_text(full_text, max_chars=chunk_size, overlap=overlap)
    # Embeddings (OpenAI si dispo, sinon fallback hashing)
    settings = get_settings()
    embs = _embeddings_openai(chunks, getattr(settings, "openai_api_key", ""))
    if not embs:
        embs = _embeddings_fallback(chunks, dim=256)
    try:
        cur = conn.cursor()
        # Remplacer l'index existant pour cette application
        cur.execute("DELETE FROM rag_index WHERE application_id = ?", (str(application_id),))
        # Insert
        for i, (txt, vec) in enumerate(zip(chunks, embs)):
            arr = np.array(vec, dtype=np.float32)
            cur.execute(
                "INSERT OR REPLACE INTO rag_index(application_id, chunk_id, text, embedding) VALUES (?, ?, ?, ?)",
                (str(application_id), int(i), txt, arr.tobytes()),
            )
        conn.commit()
    except Exception:
        pass


def rag_search(conn, application_id: str, query: str, top_k: int = 3) -> list[dict]:
    if not application_id or not query:
        return []
    ensure_sqlite_rag_schema(conn)
    # Embedding requête
    settings = get_settings()
    q_emb = _embeddings_openai([query], getattr(settings, "openai_api_key", ""))
    if q_emb:
        q = np.array(q_emb[0], dtype=np.float32)
    else:
        q = np.array(_embeddings_fallback([query], dim=256)[0], dtype=np.float32)
    try:
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, text, embedding FROM rag_index WHERE application_id = ?", (str(application_id),))
        rows = cur.fetchall()
        scored = []
        for cid, txt, blob in rows:
            try:
                v = np.frombuffer(blob, dtype=np.float32)
                scored.append((float(_cosine(q, v)), int(cid), txt))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for s, cid, txt in scored[:max(1, int(top_k))]:
            out.append({"chunk_id": cid, "score": s, "text": txt})
        return out
    except Exception:
        return []

@st.cache_data(ttl=900)
def get_application_docs_text(app_id: str, links: list[str] | None) -> dict:
    out = {"full": "", "per_doc": []}
    if not links:
        return out
    texts = []
    for url in links:
        if not isinstance(url, str) or not url.strip():
            continue
        data = _download_bytes(url)
        if not data:
            continue
        txt = ""
        low = url.lower()
        if low.endswith(".pdf"):
            txt = _extract_text_from_pdf(data)
        elif any(low.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]):
            txt = _ocr_image_bytes(data)
        elif low.endswith(".docx") or low.endswith(".doc"):
            txt = _extract_text_from_docx(data)
        elif low.endswith(".txt"):
            txt = _extract_text_from_txt(data)
        else:
            # Tentative générique texte
            txt = _extract_text_from_txt(data)
            if not txt:
                # Essai image OCR générique
                tmp = _ocr_image_bytes(data)
                if tmp:
                    txt = tmp
        if txt and txt.strip():
            texts.append((url, txt))
    # Concat et résumés courts par doc
    out["per_doc"] = [{"url": u, "summary": _summarize_hierarchical(t, chunk_size=1500, overlap=200)} for (u, t) in texts]
    full_text = "\n\n".join([t for (_, t) in texts])
    out["full"] = full_text
    # Résumé global hiérarchique optionnel pour prompts compacts
    try:
        out["summary"] = _summarize_hierarchical(full_text, chunk_size=1800, overlap=200)
    except Exception:
        out["summary"] = _summarize_text_simple(full_text, max_chars=1800)
    return out


def _ocr_image_bytes(data: bytes, lang: str = "eng+fra") -> str:
    if not data or Image is None or pytesseract is None:
        return ""
    try:
        img = Image.open(BytesIO(data))
        # Convertir en RGB pour stabilité
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
        # Matrice de rendu pour DPI souhaité
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

def _derive_age_series(users: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns: user_id, age (int), gender_label (Homme/Femme/Inconnu).
    Tries several birth date column names and computes age in years.
    """
    # Prepare user_id mapping
    df_users = users.copy() if not users.empty else pd.DataFrame()
    if not df_users.empty:
        if "id" in df_users.columns and "user_id" not in df_users.columns:
            df_users = df_users.rename(columns={"id": "user_id"})
    # Build base with user_id
    if not df_users.empty and "user_id" in df_users.columns:
        base = df_users[["user_id"]].drop_duplicates().copy()
    elif not profiles.empty and "user_id" in profiles.columns:
        base = profiles[["user_id"]].drop_duplicates().copy()
    else:
        return pd.DataFrame(columns=["user_id", "age", "gender_label"])  # no identifiers

    # Find birth date column from profiles then users
    date_cols = ["birth_date", "date_naissance", "birthdate", "dob", "date_of_birth"]
    dob = None
    if not profiles.empty:
        for c in date_cols:
            if c in profiles.columns:
                dob = profiles[["user_id", c]].rename(columns={c: "dob"})
                break
    if dob is None and not df_users.empty:
        for c in date_cols:
            if c in df_users.columns:
                # users dataframe keyed by user_id
                src = df_users[["user_id", c]] if "user_id" in df_users.columns else pd.DataFrame()
                if not src.empty:
                    dob = src.rename(columns={c: "dob"})
                    break
    if dob is None:
        base["age"] = np.nan
    else:
        dob["dob"] = pd.to_datetime(dob["dob"], errors="coerce")
        # Compute age
        today = pd.Timestamp.today().normalize()
        dob["age"] = ((today - dob["dob"]) / pd.Timedelta(days=365.25)).astype("float").apply(lambda x: np.floor(x) if pd.notna(x) else np.nan)
        base = base.merge(dob[["user_id", "age"]], on="user_id", how="left")

    # Gender labels using existing helper
    g = _derive_gender_series(df_users, profiles)
    gender_map = pd.DataFrame({"user_id": base["user_id"].values})
    # Try attach gender via profiles.user_id if available
    if not profiles.empty and "gender" in profiles.columns and "user_id" in profiles.columns:
        gg = profiles[["user_id", "gender"]].copy()
        gg["gender"] = gg["gender"].fillna("Inconnu").astype(str).str.lower().map({
            "m": "Homme", "male": "Homme", "homme": "Homme",
            "f": "Femme", "female": "Femme", "femme": "Femme",
        }).fillna("Inconnu")
        gender_map = gender_map.merge(gg, on="user_id", how="left")
        gender_map = gender_map.rename(columns={"gender": "gender_label"})
    elif not df_users.empty and "gender" in df_users.columns:
        gg = df_users[["user_id", "gender"]].copy()
        gg["gender"] = gg["gender"].fillna("Inconnu").astype(str).str.lower().map({
            "m": "Homme", "male": "Homme", "homme": "Homme",
            "f": "Femme", "female": "Femme", "femme": "Femme",
        }).fillna("Inconnu")
        gender_map = gender_map.merge(gg, on="user_id", how="left")
        gender_map = gender_map.rename(columns={"gender": "gender_label"})
    else:
        gender_map["gender_label"] = "Inconnu"

    out = base.merge(gender_map, on="user_id", how="left")
    return out

@st.cache_data(ttl=300)
def load_data(sqlite_db: str | None = None):
    sb = get_supabase()
    # Core tables
    apps = sb.table("applications").select("*").execute().data or []
    scores = []
    # Prefer SQLite if specified and exists
    if sqlite_db and os.path.exists(sqlite_db):
        try:
            conn = get_sqlite(sqlite_db)
            ensure_sqlite_scores_schema(conn)
            df_sql = pd.read_sql_query("SELECT * FROM scores", conn)
            scores_df = df_sql
        except Exception:
            scores_df = pd.DataFrame()
    else:
        try:
            scores = sb.table("scores").select("*").execute().data or []
        except Exception:
            scores = []
    # Users/candidates
    try:
        users = sb.table("users").select("*").execute().data or []
    except Exception:
        users = []
    if not users:
        try:
            users = sb.table("candidates").select("*").execute().data or []
        except Exception:
            users = []
    # Candidate profiles (for gender and extra info)
    try:
        profiles = sb.table("candidate_profiles").select("*").execute().data or []
    except Exception:
        profiles = []
    # Positions/job offers
    positions = []
    for t in ["job_offers", "positions", "jobs", "offers"]:
        try:
            positions = sb.table(t).select("*").execute().data or []
            if isinstance(positions, list):
                break
        except Exception:
            continue
    # Documents by application
    documents = []
    for t in ["application_documents", "documents", "docs"]:
        try:
            documents = sb.table(t).select("*").execute().data or []
            if isinstance(documents, list):
                break
        except Exception:
            continue
    # Fallback: if no scores available yet, try protocol_evaluations as score source
    if (not isinstance(locals().get("scores_df", None), pd.DataFrame) or scores_df.empty) and not scores:
        try:
            proto = sb.table("protocol_evaluations").select("*").execute().data or []
        except Exception:
            proto = []
        if proto:
            dfp = pd.DataFrame(proto)
            # Map to a scores-like structure
            # Expect columns: application_id, overall_score
            cols = {}
            if "overall_score" in dfp.columns:
                cols["final"] = dfp["overall_score"].astype(float)
            elif "overall" in dfp.columns:
                cols["final"] = dfp["overall"].astype(float)
            if "application_id" in dfp.columns:
                cols["application_id"] = dfp["application_id"]
            scores_df = pd.DataFrame(cols) if cols else pd.DataFrame()
        else:
            scores_df = pd.DataFrame()
    elif not isinstance(locals().get("scores_df", None), pd.DataFrame):
        scores_df = pd.DataFrame(scores)

    return (
        pd.DataFrame(apps),
        scores_df,
        pd.DataFrame(users),
        pd.DataFrame(positions),
        pd.DataFrame(documents),
        pd.DataFrame(profiles),
    )


def build_enriched(apps: pd.DataFrame, scores: pd.DataFrame, users: pd.DataFrame, positions: pd.DataFrame, documents: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    df = apps.copy()
    # Merge scores
    if not scores.empty:
        df = df.merge(scores, left_on="id", right_on="application_id", how="left", suffixes=("","_score"))
    # Merge users/candidates
    if not users.empty and "id" in users.columns:
        left_key = "candidate_id" if "candidate_id" in df.columns else "user_id"
        if left_key in df.columns:
            df = df.merge(users.rename(columns={"id":"candidate_ref"}), left_on=left_key, right_on="candidate_ref", how="left")
    # Merge profiles (gender, etc.)
    if not profiles.empty:
        key_users = "candidate_ref" if "candidate_ref" in df.columns else ("user_id" if "user_id" in df.columns else None)
        if key_users and "user_id" in profiles.columns:
            df = df.merge(profiles, left_on=key_users, right_on="user_id", how="left", suffixes=("","_profile"))
    # Merge positions
    if not positions.empty and "id" in positions.columns:
        pos_key = "job_offer_id" if "job_offer_id" in df.columns else "position_id"
        if pos_key in df.columns:
            df = df.merge(positions.rename(columns={"id":"position_ref"}), left_on=pos_key, right_on="position_ref", how="left", suffixes=("","_pos"))
    # Attach documents aggregate
    if not documents.empty and "application_id" in documents.columns:
        # Provide a simple list of links/paths per application
        link_col = None
        for c in ["link", "url", "public_url", "download_url", "path", "storage_path", "file_path", "filename", "name"]:
            if c in documents.columns:
                link_col = c
                break
        doc_agg = documents.groupby("application_id").apply(
            lambda g: [str(x) for x in (g[link_col] if link_col in g.columns else g.index) if str(x).strip()],
            include_groups=False,
        ).reset_index(name="doc_links")
        df = df.merge(doc_agg, left_on="id", right_on="application_id", how="left")
    return df


def _derive_gender_series(users: pd.DataFrame, profiles: pd.DataFrame) -> pd.Series:
    # Prefer candidate_profiles.gender if available
    if not profiles.empty and "gender" in profiles.columns and "user_id" in profiles.columns:
        s = profiles["gender"].fillna("Inconnu").astype(str)
        s = s.str.lower().map({"m": "Homme", "male": "Homme", "homme": "Homme", "f": "Femme", "female": "Femme", "femme": "Femme"}).fillna("Inconnu")
        return s
    if not users.empty:
        for c in ["gender", "sexe", "genre"]:
            if c in users.columns:
                s = users[c].fillna("Inconnu").astype(str)
                s = s.str.lower().map({"m": "Homme", "male": "Homme", "homme": "Homme", "f": "Femme", "female": "Femme", "femme": "Femme"}).fillna("Inconnu")
                return s
        return pd.Series(["Inconnu"] * len(users))
    return pd.Series([], dtype=str)


def page_home():
    st.title("SEEG Recrutement — Accueil")
    st.caption("Vue synthétique du pipeline de candidatures et des indicateurs clés")
    # Style global léger (cartes KPIs)
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #ececec; border-radius: 12px;
            padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,.04);
        }
        /* Forcer le texte des métriques en noir (label, valeur, delta) */
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #000 !important;
        }
        /* Certaines versions encapsulent le contenu dans des divs internes */
        [data-testid="stMetric"] * { color: #000; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    sqlite_db = st.session_state.get("sqlite_db")
    apps, scores, users, positions, documents, profiles = load_data(sqlite_db)
    df = build_enriched(apps, scores, users, positions, documents, profiles)

    # KPIs
    # Déterminer clés utiles
    cand_key = "candidate_id" if "candidate_id" in apps.columns else ("user_id" if "user_id" in apps.columns else None)
    pos_key = "job_offer_id" if "job_offer_id" in apps.columns else ("position_id" if "position_id" in apps.columns else None)

    # Calculs
    total_applications = int(len(apps))
    if cand_key:
        try:
            total_candidates = int(apps[cand_key].nunique())
        except Exception:
            total_candidates = int(len(users)) if not isinstance(users, list) else len(users)
    else:
        total_candidates = int(len(users)) if not isinstance(users, list) else len(users)
    multi_cnt = 0
    multi_df = pd.DataFrame()
    if cand_key and pos_key and not apps.empty:
        try:
            tmp = apps[[cand_key, pos_key]].dropna()
            grp = tmp.groupby(cand_key)[pos_key].nunique().reset_index(name="nb_postes")
            multi_df = grp[grp["nb_postes"] > 1].sort_values("nb_postes", ascending=False)
            multi_cnt = int((grp["nb_postes"] > 1).sum())
        except Exception:
            multi_cnt = 0

    # Organisation en onglets: Vue globale (KPIs + Graphs) et Tableau
    tab1, tab2 = st.tabs(["Vue globale", "Tableau"])

    with tab1:
        # Ligne métriques
        colm1, colm2, colm3, colm4 = st.columns(4)
        with colm1:
            st.metric("Nombre total de candidats", total_candidates)
        with colm2:
            st.metric("Nombre de candidatures", total_applications)
        with colm3:
            st.metric("Nombre de candidats multi-postes", multi_cnt)
        with colm4:
            # Afficher aussi la proportion multi-postes si possible
            try:
                ratio = (multi_cnt / total_candidates * 100.0) if total_candidates else 0.0
                st.metric("% multi-postes", f"{ratio:.1f}%")
            except Exception:
                st.metric("% multi-postes", "-")

        st.divider()
        # Ligne graphiques: Répartition H/F et Candidats multi-postes
        colg1, colg2 = st.columns(2)
        with colg1:
            g = _derive_gender_series(users, profiles)
            gcount = g.value_counts().reindex(["Homme","Femme","Inconnu"], fill_value=0)
            fig_g = px.pie(values=gcount.values, names=gcount.index, title="Répartition H/F")
            fig_g.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_g, use_container_width=True)
        with colg2:
            # Construire labels des candidats multi-postes
            if not multi_df.empty:
                # Joindre noms si disponibles
                names_map = None
                try:
                    if not users.empty and "id" in users.columns:
                        # Essayer colonnes de nom usuelles
                        name_col = None
                        for k in ["name", "first_name", "prenom", "last_name", "nom"]:
                            if k in users.columns:
                                name_col = k
                                break
                        if name_col:
                            names_map = users[["id", name_col]].rename(columns={"id": cand_key, name_col: "candidate_name"})
                except Exception:
                    names_map = None
                df_bar = multi_df.copy()
                df_bar = df_bar.head(20)
                if names_map is not None and cand_key in names_map.columns:
                    try:
                        df_bar = df_bar.merge(names_map, on=cand_key, how="left")
                    except Exception:
                        df_bar["candidate_name"] = df_bar[cand_key].astype(str)
                if "candidate_name" not in df_bar.columns:
                    df_bar["candidate_name"] = df_bar[cand_key].astype(str)
                fig_multi = px.bar(df_bar, x="candidate_name", y="nb_postes", title="Candidats multi-postes (Top 20)")
                fig_multi.update_layout(xaxis_title="Candidat", yaxis_title="Nombre de postes", template="plotly_white", height=380)
                st.plotly_chart(fig_multi, use_container_width=True)
            else:
                st.info("Aucun candidat multi-postes identifié")

        # Candidatures par poste (bar chart)
        if pos_key and not apps.empty:
            try:
                st.subheader("Candidatures par poste")
                vc = apps[pos_key].value_counts().reset_index()
                vc.columns = [pos_key, "nb_candidatures"]
                # Joindre titre du poste si possible
                label_col = None
                if positions is not None and hasattr(positions, 'empty') and not positions.empty:
                    title_candidates = ["title", "job_title", "position_title", "name", "libelle"]
                    for c in title_candidates:
                        if c in positions.columns:
                            label_col = c
                            break
                    if label_col and "id" in positions.columns:
                        vc = vc.merge(positions[["id", label_col]].rename(columns={"id": pos_key}), on=pos_key, how="left")
                if label_col and label_col in vc.columns:
                    vc["poste_label"] = vc[label_col].fillna(vc[pos_key].astype(str))
                else:
                    vc["poste_label"] = vc[pos_key].astype(str)
                vc = vc.sort_values("nb_candidatures", ascending=False).head(20)
                fig_pos = px.bar(vc, x="poste_label", y="nb_candidatures", title="Top postes par nombre de candidatures")
                fig_pos.update_layout(xaxis_title="Poste", yaxis_title="Candidatures", template="plotly_white", height=420)
                st.plotly_chart(fig_pos, use_container_width=True)
            except Exception:
                pass

        # Nouveaux graphiques clés
        st.divider()
        st.subheader("Analyses complémentaires")
        colx1, colx2 = st.columns(2)

        # 1) Histogramme des scores finaux
        with colx1:
            try:
                if 'final' in df.columns:
                    df_scores = pd.to_numeric(df['final'], errors='coerce').dropna()
                    if len(df_scores) > 0:
                        fig_hist = px.histogram(df, x='final', nbins=20, title='Distribution des scores finaux')
                        fig_hist.update_layout(xaxis_title='Score final', yaxis_title='Effectif', template='plotly_white', height=360)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.info("Aucun score exploitable pour l'histogramme")
                else:
                    st.info("Colonne 'final' absente pour l'histogramme")
            except Exception:
                st.info("Impossible d'afficher l'histogramme des scores")

        # 2) Boxplot des scores par poste
        with colx2:
            try:
                # Déterminer un label de poste dans df
                label_candidates = [
                    'title', 'job_title', 'position_title',
                    'title_pos', 'job_title_pos', 'position_title_pos',
                ]
                poste_label_col = None
                for c in label_candidates:
                    if c in df.columns:
                        poste_label_col = c
                        break
                if poste_label_col and 'final' in df.columns:
                    dtmp = df[[poste_label_col, 'final']].dropna()
                    dtmp = dtmp[dtmp[poste_label_col].astype(str).str.strip() != '']
                    if not dtmp.empty:
                        fig_box = px.box(dtmp, x=poste_label_col, y='final', title='Scores par poste (boxplot)')
                        fig_box.update_layout(xaxis_title='Poste', yaxis_title='Score final', template='plotly_white', height=360)
                        fig_box.update_xaxes(tickangle=-35)
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.info("Pas de données suffisantes pour le boxplot par poste")
                else:
                    st.info("Colonnes poste/score manquantes pour le boxplot")
            except Exception:
                st.info("Impossible d'afficher le boxplot des scores par poste")

        # 3) Évolution mensuelle des candidatures
        try:
            if not apps.empty:
                date_col = None
                for c in ['created_at', 'applied_at', 'submitted_at', 'date']:
                    if c in apps.columns:
                        date_col = c
                        break
                if date_col is not None:
                    at = pd.to_datetime(apps[date_col], errors='coerce')
                    s = at.dt.to_period('M').value_counts().sort_index()
                    if len(s) > 0:
                        ts_df = s.rename_axis('mois').reset_index(name='candidatures')
                        ts_df['mois'] = ts_df['mois'].astype(str)
                        fig_ts = px.bar(ts_df, x='mois', y='candidatures', title='Évolution mensuelle des candidatures')
                        fig_ts.update_layout(xaxis_title='Mois', yaxis_title='Candidatures', template='plotly_white', height=380)
                        st.plotly_chart(fig_ts, use_container_width=True)
                    else:
                        st.info("Aucune donnée temporelle exploitable pour l'évolution mensuelle")
                else:
                    st.info("Aucune colonne de date (created_at/applied_at/...) trouvée pour l'évolution mensuelle")
        except Exception:
            st.info("Impossible d'afficher l'évolution des candidatures")

        # 4) Heatmap des corrélations entre scores
        try:
            score_cols = [c for c in ['final', 'fit', 'completeness'] if c in df.columns]
            if len(score_cols) >= 2:
                dnum = df[score_cols].apply(pd.to_numeric, errors='coerce')
                if dnum.dropna(how='all').shape[0] > 0:
                    corr = dnum.corr()
                    fig_corr = px.imshow(corr, title='Corrélations des scores', color_continuous_scale='Blues', zmin=-1, zmax=1)
                    fig_corr.update_layout(template='plotly_white', height=420)
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Pas assez de données numériques pour la corrélation")
            else:
                st.info("Colonnes de scores insuffisantes pour la corrélation")
        except Exception:
            st.info("Impossible d'afficher la heatmap de corrélations")

        # 5) Scores moyens par poste
        try:
            label_candidates = [
                'title', 'job_title', 'position_title',
                'title_pos', 'job_title_pos', 'position_title_pos',
            ]
            poste_label_col = None
            for c in label_candidates:
                if c in df.columns:
                    poste_label_col = c
                    break
            if poste_label_col and 'final' in df.columns:
                dtmp = df[[poste_label_col, 'final']].copy()
                dtmp['final'] = pd.to_numeric(dtmp['final'], errors='coerce')
                dtmp = dtmp.dropna()
                dtmp = dtmp[dtmp[poste_label_col].astype(str).str.strip() != '']
                if not dtmp.empty:
                    agg = dtmp.groupby(poste_label_col)['final'].agg(['mean','count']).reset_index()
                    agg = agg.sort_values('mean', ascending=False).head(15)
                    agg = agg.rename(columns={'mean': 'score_moyen', 'count': 'n'})
                    fig_mean = px.bar(agg, x=poste_label_col, y='score_moyen', title='Score moyen par poste (Top 15)')
                    fig_mean.update_layout(xaxis_title='Poste', yaxis_title='Score moyen', template='plotly_white', height=420)
                    fig_mean.update_xaxes(tickangle=-35)
                    st.plotly_chart(fig_mean, use_container_width=True)
                else:
                    st.info("Pas de données suffisantes pour les scores moyens par poste")
            else:
                st.info("Colonnes poste/score manquantes pour les scores moyens par poste")
        except Exception:
            st.info("Impossible d'afficher les scores moyens par poste")

    with tab2:
        # Filtres + Tableau
        st.subheader("Liste des candidats")
        col1, col2 = st.columns(2)
        with col1:
            lastname_q = st.text_input("Nom contient")
        with col2:
            firstname_q = st.text_input("Prénom contient")

        # Filtre d'affichage: masquer colonnes techniques
        def _filter_display_columns(df_in: pd.DataFrame) -> pd.DataFrame:
            hidden_exact = {
                "id",
                "job_offer_id",
                "cover_letter",
                "reference_contact",
                "reference_contacts",
                "role",
                "application_id_x",
                "created_at_score",
                "updated_at_score",
                "candidate_ref",
                "created_at_y",
                "updated_at_y",
                "user_id_x",
                "position_ref",
                "created_at_pos",
                "updated_at_pos",
                "date_limit",
                "start_date",
                "application_id",
                "user_id_y",
                "doc_links",
            }
            cols = []
            for c in df_in.columns:
                # created/updated timestamps
                if c == "created_at" or c.endswith("_created_at") or c.startswith("created_at"):
                    continue
                if c == "updated_at" or c.endswith("_updated_at") or c.startswith("updated_at") or c.startswith("update_at"):
                    continue
                # ids
                if c in hidden_exact:
                    continue
                if c == "id" or ("_id" in c) or c.startswith("id_"):
                    continue
                cols.append(c)
            # Assurer qu'il reste au moins quelque chose
            return df_in[cols] if cols else df_in

        base = df.copy()
        # Dédupliquer: 1 ligne par candidat (garder la plus récente)
        try:
            # Déterminer clé candidat
            cand_key = None
            for k in ["candidate_ref", "user_id", "candidate_id"]:
                if k in base.columns:
                    cand_key = k
                    break
            # Déterminer colonne de récence
            date_cols = [c for c in ["created_at", "submitted_at", "applied_at", "updated_at"] if c in base.columns]
            if date_cols:
                # Créer une colonne de tri récence
                sort_col = None
                for c in date_cols:
                    try:
                        base[c] = pd.to_datetime(base[c], errors="coerce")
                    except Exception:
                        pass
                sort_col = date_cols[0]
                for c in date_cols:
                    # choisir la première colonne non nulle la plus informative
                    if base[c].notna().sum() > base[sort_col].notna().sum():
                        sort_col = c
                base = base.sort_values(by=[sort_col], ascending=False)
            elif "id" in base.columns:
                # fallback: id décroissant
                base = base.sort_values(by=["id"], ascending=False)
            # Drop duplicates
            if cand_key:
                base = base.drop_duplicates(subset=[cand_key], keep="first")
            else:
                # fallback sans clé: essayer email sinon Nom+Prénom
                if "email" in base.columns:
                    base = base.drop_duplicates(subset=["email"], keep="first")
                else:
                    # on dédupliquera après construction disp par (Lastname, firstname)
                    pass
        except Exception:
            pass
        # Préparer une clé utilisateur pour éventuel âge
        key_users = None
        for k in ["candidate_ref", "user_id"]:
            if k in base.columns:
                key_users = k
                break
        # Dériver l'âge via helper si possible
        age_map = None
        try:
            ages_df = _derive_age_series(users, profiles)
            if not ages_df.empty and key_users and "user_id" in ages_df.columns:
                age_map = ages_df[["user_id", "age"]].drop_duplicates()
                base = base.merge(age_map, left_on=key_users, right_on="user_id", how="left")
        except Exception:
            pass

        def pick_from_identity(row, key):
            try:
                d = row.get("details")
                if isinstance(d, dict):
                    ident = d.get("identity") or {}
                    v = ident.get(key)
                    if v is not None and str(v).strip():
                        return str(v).strip()
            except Exception:
                pass
            return None

        def compute_firstname(row):
            return (
                pick_from_identity(row, "first_name")
                or row.get("first_name")
                or row.get("prenom")
                or None
            )

        def compute_lastname(row):
            return (
                pick_from_identity(row, "last_name")
                or row.get("last_name")
                or row.get("nom")
                or None
            )

        def compute_email(row):
            return (
                pick_from_identity(row, "email")
                or row.get("email")
                or row.get("mail")
                or None
            )

        def compute_phone(row):
            return (
                pick_from_identity(row, "phone")
                or row.get("phone")
                or row.get("mobile")
                or row.get("telephone")
                or None
            )

        def compute_matricule(row):
            return (
                pick_from_identity(row, "matricule")
                or row.get("matricule")
                or None
            )

        def compute_years_exp(row):
            for c in [
                "years_experience", "experience_years", "annees_experience", "years_of_experience",
                "years_exp", "exp_years",
            ]:
                v = row.get(c)
                if v is not None and str(v).strip():
                    try:
                        return float(pd.to_numeric(v, errors="coerce"))
                    except Exception:
                        return None
            # Essai depuis profiles si présent dans row fusionné
            for c in ["experience", "experiences"]:
                v = row.get(c)
                if v is not None and str(v).strip():
                    try:
                        # extraire nombre si textuel
                        m = re.search(r"(\d+[\.,]?\d*)", str(v))
                        if m:
                            return float(m.group(1).replace(",", "."))
                    except Exception:
                        pass
            return None

        def compute_current_post(row):
            for c in [
                "current_post", "current_position", "poste_actuel", "poste", "job", "profession",
            ]:
                v = row.get(c)
                if v is not None and str(v).strip():
                    return str(v).strip()
            # fallback: poste visé (depuis position)
            for c in ["title", "job_title", "position_title", "title_pos", "job_title_pos", "position_title_pos"]:
                v = row.get(c)
                if v is not None and str(v).strip():
                    return str(v).strip()
            return None

        # Construire DataFrame d'affichage avec colonnes demandées
        disp = pd.DataFrame()
        disp["Lastname"] = base.apply(compute_lastname, axis=1)
        disp["firstname"] = base.apply(compute_firstname, axis=1)
        # Age: préférer colonne fusionnée, sinon None
        if "age" in base.columns:
            try:
                disp["age"] = pd.to_numeric(base["age"], errors="coerce").round().astype("Int64")
            except Exception:
                disp["age"] = pd.to_numeric(base["age"], errors="coerce")
        else:
            disp["age"] = pd.Series([pd.NA] * len(base), dtype="Int64")
        disp["matricule"] = base.apply(compute_matricule, axis=1)
        disp["email"] = base.apply(compute_email, axis=1)
        disp["phone"] = base.apply(compute_phone, axis=1)
        disp["years_experience"] = base.apply(compute_years_exp, axis=1)
        disp["current_post"] = base.apply(compute_current_post, axis=1)

        # Dédup fallback si pas de clé: par (Lastname, firstname)
        try:
            if disp.shape[0] > 0 and ("Lastname" in disp.columns and "firstname" in disp.columns):
                disp = disp.drop_duplicates(subset=["Lastname", "firstname"], keep="first")
        except Exception:
            pass

        # Appliquer filtres sur Nom/Prénom
        disp_mask = pd.Series([True] * len(disp))
        if lastname_q:
            disp_mask &= disp["Lastname"].fillna("").str.contains(lastname_q, case=False, na=False)
        if firstname_q:
            disp_mask &= disp["firstname"].fillna("").str.contains(firstname_q, case=False, na=False)
        disp = disp[disp_mask]

        # Trier A -> Z par Nom puis Prénom (case-insensitive, NaN en dernier)
        if not disp.empty:
            disp = disp.sort_values(
                by=["Lastname", "firstname"],
                key=lambda s: s.fillna("").map(lambda x: ud.normalize('NFKD', x).encode('ASCII','ignore').decode('ASCII').lower()),
                na_position='last'
            )

        # Style bouton pour l'action "Voir détails"
        st.markdown(
            """
            <style>
            /* Bouton visuel pour la colonne Voir détails */
            .voir-details-btn {
                display: inline-block;
                background: #0d6efd;
                color: #fff !important;
                padding: 4px 10px;
                border-radius: 6px;
                text-decoration: none;
                font-weight: 600;
                border: 1px solid #0b5ed7;
                cursor: pointer;
            }
            .voir-details-btn:hover {
                background: #0b5ed7;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Identifier la colonne d'ID utilisable pour la navigation
        nav_id_col = None
        for _c in ["id", "application_id", "application_id_x", "app_id", "applicationId"]:
            if _c in base.columns:
                nav_id_col = _c
                break

        # Colonne action "Voir détails" (si un identifiant est disponible)
        if nav_id_col is not None:
            try:
                ids_aligned = base.loc[disp.index, nav_id_col].astype(str)
                disp["Voir"] = ids_aligned
            except Exception:
                pass

        # Afficher un identifiant séquentiel 0..N-1 correspondant au nombre de candidats affichés
        try:
            seq_ids = pd.RangeIndex(start=0, stop=len(disp))
            if "id" in disp.columns:
                disp.drop(columns=["id"], inplace=True)
            disp.insert(0, "id", seq_ids)
        except Exception:
            pass

        # Rendu du tableau (toujours affiché)
        column_cfg = {}
        if "Voir" in disp.columns:
            column_cfg["Voir"] = st.column_config.TextColumn("Voir détails")
        st.data_editor(
            disp.reset_index(drop=True),
            width='stretch',
            column_config=column_cfg if column_cfg else None,
            disabled=[c for c in disp.columns if c != "Voir"],
            hide_index=True,
        )

        # Navigation contrôlée (même onglet) via selectbox + bouton (toujours visibles)
        try:
            def _label_from_disp(row: pd.Series) -> str:
                nom = str(row.get("Lastname") or "").strip()
                pr = str(row.get("firstname") or "").strip()
                em = str(row.get("email") or "").strip()
                name = (nom + " " + pr).strip() or None
                if name and em:
                    return f"{name} ({em})"
                return name or em or "Candidat"

            labels = [ _label_from_disp(disp.loc[i]) for i in disp.index ] if not disp.empty else []
            sel_label = st.selectbox("Sélectionner un candidat", options=labels, key="select_cand_table") if labels else None

            # Associer label -> id si possible
            mapping = {}
            if nav_id_col is not None and labels:
                try:
                    ids_series = base.loc[disp.index, nav_id_col].astype(str)
                    mapping = dict(zip(labels, ids_series.values))
                except Exception:
                    mapping = {}

            # Bouton toujours visible, désactivé si pas de sélection ou pas d'ID dispo
            col_btn1, _ = st.columns([1,6])
            with col_btn1:
                disabled_btn = not sel_label or not mapping
                if st.button("Voir détails", key="btn_voir_details", disabled=disabled_btn):
                    sel_id = mapping.get(sel_label)
                    if sel_id:
                        try:
                            st.query_params["page"] = "Candidat"
                            st.query_params["app_id"] = sel_id
                        except Exception:
                            st.markdown(f"[Voir détails](/?page=Candidat&app_id={sel_id})")
                        st.rerun()
            if nav_id_col is None:
                st.info("Aucun identifiant de candidature n'a été trouvé dans les données (colonnes attendues: id/application_id).")
        except Exception:
            pass

def page_candidate():
    st.title("Détail Candidat")
    sqlite_db = st.session_state.get("sqlite_db")
    apps, scores, users, positions, documents, profiles = load_data(sqlite_db)
    df = build_enriched(apps, scores, users, positions, documents, profiles)

    # Sélection par nom/prénom (et titre de poste), tout en conservant l'app_id en interne
    def _get_name_from_row(r: pd.Series) -> str:
        # 1) details.identity.name si présent
        try:
            if isinstance(r.get("details"), dict):
                nm = r["details"].get("identity", {}).get("name")
                if nm and str(nm).strip():
                    return str(nm).strip()
        except Exception:
            pass
        # 2) colonnes utilisateurs classiques
        for k in ["name", "first_name", "prenom", "last_name", "nom"]:
            v = r.get(k)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        return "(Nom inconnu)"

    def _get_job_from_row(r: pd.Series) -> str:
        for k in ["title", "job_title", "position_title"]:
            v = r.get(k)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        return ""

    if "id" in df.columns:
        df = df.copy()
        df["_id_str"] = df["id"].astype(str)
        df["_label"] = df.apply(lambda r: f"{_get_name_from_row(r)}" + (f" — {_get_job_from_row(r)}" if _get_job_from_row(r) else ""), axis=1)
        labels = df["_label"].tolist()
        # Pré-sélection via ?app_id=...
        qp_app = None
        try:
            qp_app = st.query_params.get("app_id")
        except Exception:
            qp_app = None
        pre_idx = 0
        if qp_app and qp_app in df["_id_str"].values:
            pre_idx = df.index.get_loc(df[df["_id_str"] == qp_app].index[0])
        selected_label = st.selectbox("Sélectionner un candidat", options=labels, index=(pre_idx if labels else 0))
        if not selected_label:
            st.info("Aucune candidature")
            return
        selected_id = df.loc[df["_label"] == selected_label, "_id_str"].iloc[0]
        row = df[df["_id_str"] == selected_id].iloc[0]
    else:
        st.info("Aucune candidature")
        return

    # Bloc identités
    st.subheader("Informations personnelles")
    identity = {}
    if isinstance(row.get("details"), dict) and isinstance(row["details"].get("identity"), dict):
        identity = row["details"]["identity"]
    # fallback depuis colonnes users
    def _pick(*keys):
        for k in keys:
            if k in row.index and pd.notna(row[k]) and str(row[k]).strip():
                return row[k]
        return None
    cols = st.columns(4)
    with cols[0]:
        st.write("Nom:", identity.get("name") or _pick("name", "first_name", "prenom"))
    with cols[1]:
        st.write("Email:", identity.get("email") or _pick("email", "mail"))
    with cols[2]:
        st.write("Téléphone:", identity.get("phone") or _pick("phone", "mobile", "telephone"))
    with cols[3]:
        st.write("Matricule:", identity.get("matricule") or _pick("matricule"))

    # Scores gauges (préparer les 3 premières figures)
    st.subheader("Scores")
    comp_val = float(row.get("completeness", 0) or 0)
    fit_val_pre = float(row.get("fit", 0) or 0)
    final_val = float(row.get("final", 0) or 0)
    fig_comp = go.Figure(go.Indicator(mode="gauge+number", value=comp_val, title={'text': "Complétude"}, gauge={'axis': {'range': [0,100]}}))
    fig_fit0 = go.Figure(go.Indicator(mode="gauge+number", value=fit_val_pre, title={'text': "Adéquation"}, gauge={'axis': {'range': [0,100]}}))
    fig_final = go.Figure(go.Indicator(mode="gauge+number", value=final_val, title={'text': "Score final"}, gauge={'axis': {'range': [0,100]}}))

    # Score d'adhérence MTP au poste (calcul à la volée, sans toucher aux scores persistés)
    st.subheader("Adhérence MTP au poste")

    def _extract_mtp_overall(r: pd.Series) -> float | None:
        try:
            d = r.get("details") if "details" in r.index else None
            if isinstance(d, dict):
                mtp = d.get("mtp") or {}
                if isinstance(mtp, dict):
                    sc = mtp.get("scores") or {}
                    if isinstance(sc, dict):
                        ov = sc.get("overall")
                        if ov is None:
                            return None
                        try:
                            return float(ov)
                        except Exception:
                            return None
        except Exception:
            return None
        return None

    def _doc_support_from_flags(r: pd.Series) -> float:
        # Mesure simple basée sur la présence des documents clés: cv, lm, diploma (+id facultatif)
        keys = ["cv", "lm", "diploma"]
        try:
            d = r.get("details") if "details" in r.index else None
            flags = d.get("flags", {}) if isinstance(d, dict) else {}
        except Exception:
            flags = {}
        have = 0
        total = len(keys)
        for k in keys:
            try:
                if bool(flags.get(k)):
                    have += 1
            except Exception:
                pass
        # Échelle 0..100
        return 100.0 * (have / total) if total else 0.0

    fit_val = float(row.get("fit", 0) or 0)
    mtp_overall = _extract_mtp_overall(row)
    doc_support = _doc_support_from_flags(row)

    # Pondérations dynamiques: privilégier fit (inclut vision SEEG), puis MTP, puis documents.
    weights = {
        "fit": 0.55,
        "mtp": 0.30,
        "docs": 0.15,
    }
    # Si MTP indisponible, renormaliser sur fit+docs
    av = {"fit": True, "mtp": isinstance(mtp_overall, (int, float)), "docs": True}
    tot = sum([w for k, w in weights.items() if av[k]]) or 1.0
    w_fit = weights["fit"] / tot
    w_mtp = (weights["mtp"] / tot) if av["mtp"] else 0.0
    w_docs = weights["docs"] / tot

    adherence = (w_fit * fit_val) + (w_mtp * (mtp_overall or 0.0)) + (w_docs * doc_support)
    adherence = float(max(0.0, min(100.0, adherence)))

    # Figure d'adhérence (pour la rangée unifiée)
    fig_ad = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adherence,
        title={'text': "Adhérence MTP"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    # Figure de conformité (pour la rangée unifiée) sera créée après calcul compliance_score

    # Commentaire court sur l'adhérence
    lines_adh = []
    lines_adh.append(f"- Fit (poste + vision): {fit_val:.1f}% (poids {w_fit*100:.0f}%)")
    if av["mtp"]:
        lines_adh.append(f"- MTP global: {float(mtp_overall):.1f}% (poids {w_mtp*100:.0f}%)")
    else:
        lines_adh.append("- MTP global: non disponible (indicateurs/questions manquants)")
    lines_adh.append(f"- Support documentaire: {doc_support:.1f}% (poids {w_docs*100:.0f}%)")

    # Interprétation brève
    if adherence >= 80:
        verdict = "Très forte adhérence au poste"
    elif adherence >= 60:
        verdict = "Adhérence correcte, à confirmer"
    else:
        verdict = "Adhérence faible"
    st.markdown("\n".join([
        f"**Synthèse adhérence**: {verdict}",
        "Détails:",
        *lines_adh,
    ]))

    # Conformité du dossier (modèle déterministe renforcé, sans IA)
    st.subheader("Conformité")
    def _extract_flags(r: pd.Series) -> dict:
        try:
            d = r.get("details") if "details" in r.index else None
            if isinstance(d, dict) and isinstance(d.get("flags"), dict):
                return dict(d.get("flags"))
        except Exception:
            pass
        return {}

    flags_all = _extract_flags(row)
    links = []
    try:
        if "doc_links" in row.index and isinstance(row.get("doc_links"), (list, tuple)):
            links = [str(u).lower() for u in row.get("doc_links")]
    except Exception:
        links = []

    # Lecture directe depuis la table documents (DB) pour l'application sélectionnée
    doc_types_present = set()
    try:
        app_id_cur = row.get("id")
        df_docs_app = pd.DataFrame()
        if app_id_cur is not None and isinstance(documents, pd.DataFrame) and not documents.empty and "application_id" in documents.columns:
            df_docs_app = documents[documents["application_id"].astype(str) == str(app_id_cur)].copy()

        def _norm_doc_type(val: str) -> str | None:
            if not isinstance(val, str):
                return None
            s = val.strip().lower()
            if not s:
                return None
            # Normalisation large
            if any(k in s for k in ["cv", "resume", "curriculum"]):
                return "cv"
            if any(k in s for k in ["diploma", "diplome", "degree", "licence", "master", "mba", "bachelor"]):
                return "diploma"
            if any(k in s for k in ["lettre", "motivation", "cover-letter", "cover_letter", "coverletter"]):
                return "lm"
            return None

        # 1) Détecter via colonnes de type explicites si disponibles
        if not df_docs_app.empty:
            type_cols = [c for c in ["type", "doc_type", "category", "kind", "label", "document_type", "name"] if c in df_docs_app.columns]
            for c in type_cols:
                for v in df_docs_app[c].astype(str).tolist():
                    t = _norm_doc_type(v)
                    if t:
                        doc_types_present.add(t)

            # 2) Fallback sur noms/liens si aucun type trouvé
            if not doc_types_present:
                link_cols = [c for c in ["link", "url", "public_url", "download_url", "path", "storage_path", "file_path", "filename", "name"] if c in df_docs_app.columns]
                try:
                    links_from_db = []
                    for c in link_cols:
                        links_from_db.extend([str(x) for x in df_docs_app[c].dropna().astype(str).tolist() if str(x).strip()])
                    if links_from_db:
                        # alimenter aussi links pour la détection heuristique existante
                        links.extend([x.lower() for x in links_from_db])
                        for x in links_from_db:
                            t = _norm_doc_type(str(x))
                            if t:
                                doc_types_present.add(t)
                except Exception:
                    pass
    except Exception:
        pass

    # Dictionnaire de synonymes/variantes pour la détection par liens
    synonyms = {
        "cv": ["cv", "resume", "curriculum", "curriculum vitae"],
        "diploma": ["diploma", "diplome", "degree", "licence", "master", "mba", "bachelor"],
        "lm": ["lettre de motivation", "lettre", "cover-letter", "cover_letter", "coverletter", "motivation"],
    }

    def _present_norm(key: str) -> bool:
        # 1) Documents DB: si un type normalisé correspond, considérer présent
        if key in doc_types_present:
            return True
        # 2) Flag explicite (hérité des détails) – fallback
        if bool(flags_all.get(key)):
            return True
        # 3) Heuristique liens/nom de fichier – fallback
        terms = synonyms.get(key, [])
        for s in links:
            if any(t in s for t in terms):
                return True
        return False

    required = ["cv", "diploma", "lm"]
    present = {k: _present_norm(k) for k in required}
    missing = [k for k, v in present.items() if not v]
    compliance_score = 100.0 * (sum(1 for v in present.values() if v) / len(required)) if required else 0.0
    is_compliant = len(missing) == 0

    # Détection facultative d'un certificat supplémentaire (heuristique sur les liens)
    has_certificate = any(any(t in s for t in ["certif", "certificate", "certificat"]) for s in links)

    # Créer la figure de jauge pour conformité (pour la rangée unifiée)
    fig_conf = go.Figure(go.Indicator(
        mode="gauge+number",
        value=compliance_score,
        title={'text': "Conformité du dossier"},
        gauge={'axis': {'range': [0, 100]}}
    ))

    # Afficher toutes les jauges sur UNE seule ligne (5 colonnes)
    cols_all = st.columns(5)
    cols_all[0].plotly_chart(fig_comp, use_container_width=True)
    cols_all[1].plotly_chart(fig_fit0, use_container_width=True)
    cols_all[2].plotly_chart(fig_final, use_container_width=True)
    cols_all[3].plotly_chart(fig_ad, use_container_width=True)
    cols_all[4].plotly_chart(fig_conf, use_container_width=True)

    # Détail en puces (checklist)
    label_map = {"cv": "CV", "diploma": "Diplôme", "lm": "Lettre de motivation"}
    lines_conf = []
    for k in required:
        ok = present.get(k, False)
        lines_conf.append(("✅" if ok else "❌") + f" {label_map.get(k, k)}")
    # Facultatif
    lines_conf.append(("✅" if has_certificate else "➖") + " Certificat supplémentaire (facultatif)")
    # Score en clair
    lines_conf.append(f"Score de conformité: {compliance_score:.0f}%")
    st.markdown("\n".join([f"- {x}" for x in lines_conf]))

    # Synthèse d'adhérence générée par IA (optionnelle)
    settings_adh = get_settings()
    if getattr(settings_adh, "openai_api_key", "") and OpenAI is not None:
        with st.expander("Synthèse adhérence (IA)"):
            if st.button("Générer la synthèse d'adhérence", key="adh_synth_btn"):
                try:
                    client = OpenAI(api_key=settings_adh.openai_api_key)
                    # Extraire infos MTP détaillées si dispo
                    mtp_scores_full = {}
                    mtp_dims_txt = []
                    try:
                        d = row.get("details") if "details" in row.index else None
                        if isinstance(d, dict):
                            mtp = d.get("mtp") or {}
                            if isinstance(mtp, dict):
                                sc = mtp.get("scores") or {}
                                if isinstance(sc, dict):
                                    mtp_scores_full = sc
                                    for k in ["metier","talent","paradigme"]:
                                        v = sc.get(k)
                                        if isinstance(v, (int, float)):
                                            mtp_dims_txt.append(f"{k}: {float(v):.1f}%")
                    except Exception:
                        pass

                    name = identity.get("name") or _pick("name", "first_name", "prenom", "last_name", "nom") or "Candidat"
                    poste = _get_job_from_row(row)

                    # Préparer info sur documents détectés depuis la DB pour éviter toute hallucination
                    documents_present = {
                        "cv": present.get("cv", False),
                        "diploma": present.get("diploma", False),
                        "lm": present.get("lm", False),
                    }

                    sys_prompt = (
                        "Tu es un assistant RH francophone. Objectif: produire une synthèse d'adhérence au poste, courte et actionnable. "
                        "Règles strictes: ne cite aucun document ni extrait si les pièces ne sont pas présentes; n'invente jamais de contenu absent. "
                        "Si des documents manquent, mentionne simplement l'absence (sans spéculer) et propose une action (demander la pièce). "
                        "Structure attendue en 5–8 puces: (1) Verdict clair (forte/correcte/faible), (2) Décomposition chiffrée par composante (fit, MTP, documents) avec poids, "
                        "(3) Forces majeures, (4) Risques/manquements, (5) Prochaine étape concrète. Ne répète pas les champs bruts; synthétise."
                    )
                    user_prompt = (
                        f"Profil: {name}\n"
                        f"Poste: {poste}\n"
                        f"Adhérence calculée (0–100): {adherence:.1f}\n"
                        f"Composantes (poids): fit={w_fit*100:.0f}%, mtp={w_mtp*100:.0f}%, docs={w_docs*100:.0f}%\n"
                        f"Valeurs: fit={fit_val:.1f}%, mtp_overall={'NA' if mtp_overall is None else f'{float(mtp_overall):.1f}%'}, docs={doc_support:.1f}%\n"
                        + ("MTP par dimension: " + ", ".join(mtp_dims_txt) + "\n" if mtp_dims_txt else "")
                        + f"Documents présents (depuis DB): {json.dumps(documents_present)}\n"
                        + ("Aucun document n'est disponible. Évite toute référence à des pièces.\n" if not any(documents_present.values()) else "")
                        + "Critères d'interprétation: >=80 très forte, 60–79 correcte, <60 faible. Rends une synthèse en puces courtes."
                    )
                    try:
                        resp = client.chat.completions.create(
                            model=getattr(settings_adh, "openai_model", "gpt-4o-mini") or "gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.2,
                            max_tokens=250,
                        )
                        content = resp.choices[0].message.content if resp and resp.choices else None
                    except Exception:
                        content = None
                    if content:
                        st.markdown(content)
                    else:
                        st.info("Impossible d'obtenir une synthèse IA pour l'instant.")
                except Exception as e:
                    st.info(f"IA désactivée/indisponible: {e}")

    # Radar chart sur complétude par composantes
    flags = {}
    if isinstance(row.get("details"), dict) and isinstance(row["details"].get("flags"), dict):
        flags = row["details"]["flags"]
    if flags:
        st.subheader("Composantes de complétude")
        axes = list(flags.keys())
        vals = [100 if bool(flags[k]) else 0 for k in axes]
        vals.append(vals[0])
        axes.append(axes[0])
        fig_r = go.Figure(data=go.Scatterpolar(r=vals, theta=axes, fill='toself'))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False)
        st.plotly_chart(fig_r, width='stretch')

    # Commentaire du modèle
    st.subheader("Commentaire du modèle")
    def _comment_from_scores(r: pd.Series) -> str:
        comp = float(r.get("completeness", 0) or 0)
        fit = float(r.get("fit", 0) or 0)
        final = float(r.get("final", 0) or 0)
        reco = r.get("recommendation")
        if not isinstance(reco, str) or not reco.strip():
            # Appliquer les mêmes seuils que seeg_core.scoring.recommend
            if final >= 80:
                reco = "Fortement recommandé"
            elif final >= 60:
                reco = "À considérer"
            else:
                reco = "Non recommandé"

        # Positionnement vs cohorte
        pct_txt = ""
        try:
            if 'final' in scores.columns:
                base = scores['final'].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                if len(base) >= 5:
                    rank = (base <= final).mean() * 100.0
                    pct_txt = f"(percentile ~{rank:.0f} parmi les candidats scorés)"
        except Exception:
            pct_txt = ""

        # Détails manquants à partir de flags
        missing = []
        d = r.get("details")
        if isinstance(d, dict) and isinstance(d.get("flags"), dict):
            for k, v in d["flags"].items():
                try:
                    if not bool(v):
                        missing.append(k)
                except Exception:
                    pass

        # Verdict + raisons
        lines = []
        lines.append(f"**Verdict**: {reco} {pct_txt}")
        lines.append(f"- Score final: {final:.1f}% | Adéquation: {fit:.1f}% | Complétude: {comp:.1f}%")

        # Facteurs clés
        key_drivers = []
        if fit >= 75:
            key_drivers.append("forte adéquation au poste")
        elif fit >= 60:
            key_drivers.append("adéquation correcte, perfectible")
        else:
            key_drivers.append("adéquation faible")
        if comp >= 75:
            key_drivers.append("dossier complet")
        elif comp >= 50:
            key_drivers.append("dossier partiellement complet")
        else:
            key_drivers.append("dossier incomplet")
        lines.append("- Facteurs clés: " + ", ".join(key_drivers) + ".")

        # Risques / manquements
        if missing:
            lines.append("- Risques: éléments manquants — " + ", ".join(sorted(set(missing))) + ".")

        # Prochaines étapes recommandées
        next_steps = []
        if final >= 80:
            next_steps.append("Planifier un entretien de validation (technique + culture).")
            if missing:
                next_steps.append("Demander les pièces manquantes avant l'entretien.")
        elif final >= 60:
            next_steps.append("Shortlister et organiser un entretien exploratoire.")
            if comp < 70:
                next_steps.append("Compléter le dossier (CV, LM, diplômes, MTP) pour affiner l'évaluation.")
            if fit < 70:
                next_steps.append("Vérifier des exemples concrets alignés avec le poste durant l'entretien.")
        elif final >= 40:
            next_steps.append("Garder en vivier ou demander des précisions ciblées si le poste le permet.")
            if fit < 55:
                next_steps.append("Évaluer la pertinence sur d'autres postes mieux alignés.")
        else:
            next_steps.append("Ne pas poursuivre pour ce poste, sauf éléments exceptionnels à reconsidérer.")
        lines.append("- Prochaines étapes: " + " ".join(next_steps))

        return "\n".join(lines)

    st.markdown(_comment_from_scores(row))

    # Génération optionnelle par ChatGPT
    settings = get_settings()
    if getattr(settings, "openai_api_key", "") and OpenAI is not None:
        with st.expander("Commentaire généré par IA"):
            model = getattr(settings, "openai_model", "gpt-4o-mini") or "gpt-4o-mini"
            # Bouton pour éviter des appels coûteux automatiques
            if st.button("Générer l'analyse IA"):
                try:
                    client = OpenAI(api_key=settings.openai_api_key)
                    # Préparer un contexte compact
                    name = identity.get("name") or _pick("name", "first_name", "prenom", "last_name", "nom") or "Candidat"
                    poste = _get_job_from_row(row)
                    comp = float(row.get("completeness", 0) or 0)
                    fit = float(row.get("fit", 0) or 0)
                    final = float(row.get("final", 0) or 0)
                    reco = row.get("recommendation") or ""
                    flags_d = row.get("details", {}).get("flags", {}) if isinstance(row.get("details"), dict) else {}
                    # Ajouter contexte MTP + base locale SQLite si disponible
                    mtp_summary = []
                    mtp_resp_summary = []
                    try:
                        sqlite_db = st.session_state.get("sqlite_db")
                        if sqlite_db:
                            conn = get_sqlite(sqlite_db)
                            ensure_sqlite_mtp_schema(conn)
                            # Questions MTP pour le poste courant
                            if poste:
                                qs = [q for q in list_mtp_questions(conn, active_only=True) if q.get("poste") == poste]
                                # Limiter à 5 éléments représentatifs (par dimension et ordre)
                                for q in qs[:5]:
                                    mtp_summary.append(f"[{q.get('dimension')}] {q.get('question')}")
                        # Résumés de réponses MTP si présents dans details
                        d = row.get("details") if "details" in row.index else None
                        if isinstance(d, dict):
                            mtp_ans = d.get("mtp_answers") or d.get("mtp")
                            if isinstance(mtp_ans, dict):
                                # Prendre quelques réponses clés
                                for k, v in list(mtp_ans.items())[:5]:
                                    if v is not None and str(v).strip():
                                        mtp_resp_summary.append(f"{k}: {str(v)[:180]}")
                        # Documents: extraire et résumer si des liens sont associés à la candidature
                        doc_links = []
                        try:
                            if "doc_links" in row.index and isinstance(row.get("doc_links"), (list, tuple)):
                                doc_links = [str(u) for u in row.get("doc_links") if u]
                        except Exception:
                            doc_links = []
                        docs_text = {"full": "", "per_doc": []}
                        app_id_str = str(row.get("id") or selected_id)
                        if doc_links:
                            docs_text = get_application_docs_text(app_id_str, doc_links)
                        # Construire un résumé compact des documents pour le prompt
                        docs_summary_lines = []
                        if docs_text.get("per_doc"):
                            for it in docs_text["per_doc"][:3]:
                                u = it.get("url", "")
                                s = it.get("summary", "")[:400]
                                if s:
                                    docs_summary_lines.append(f"- {u}: {s}")
                        # Indexation RAG des documents (plein texte)
                        rag_lines = []
                        try:
                            if sqlite_db and docs_text.get("full"):
                                upsert_rag_for_application(conn, app_id_str, docs_text["full"], chunk_size=1200, overlap=200)
                                # Construire une requête RAG basée sur le poste + fiche de poste
                                query_parts = []
                                if poste:
                                    query_parts.append(str(poste))
                                if 'job_summary' in locals() and job_summary:
                                    query_parts.append(job_summary)
                                query_parts.append("expériences pertinentes compétences réalisations responsabilités")
                                rag_query = " \n".join([p for p in query_parts if p])
                                hits = rag_search(conn, app_id_str, rag_query, top_k=3)
                                for h in hits:
                                    txt = str(h.get('text','')).strip()
                                    if txt:
                                        rag_lines.append(f"- (score {h.get('score',0):.2f}) {txt[:400]}")
                        except Exception:
                            pass
                        # Fiche de poste: agréger les champs pertinents et résumer
                        def _field_val(r: pd.Series, key: str) -> str | None:
                            try:
                                v = r.get(key)
                                if v is None or str(v).strip() == "":
                                    v = r.get(f"{key}_pos")
                                if v is None:
                                    return None
                                s = str(v).strip()
                                return s if s else None
                            except Exception:
                                return None
                        job_fields = [
                            "title", "job_title", "position_title",
                            "description", "mission", "missions",
                            "requirements", "skills", "competences",
                            "responsibilities", "responsabilites",
                            "profil", "experience", "experiences",
                            "education", "formation",
                        ]
                        job_text_parts = []
                        for k in job_fields:
                            val = _field_val(row, k)
                            if val:
                                job_text_parts.append(val)
                        job_text_full = "\n".join(dict.fromkeys([p for p in job_text_parts if p]))  # dédupliquer en gardant l'ordre
                        job_summary = ""
                        if job_text_full:
                            try:
                                job_summary = _summarize_hierarchical(job_text_full, chunk_size=900, overlap=150)
                            except Exception:
                                job_summary = _summarize_text_simple(job_text_full, max_chars=900)
                    except Exception:
                        pass
                    # Construire prompt
                    sys_prompt = (
                        "Tu es un assistant RH francophone pour SEEG, utilisé en direct pendant l'entretien. "
                        "Objectif: aider le recruteur à juger si le candidat est le meilleur pour le poste. "
                        "Priorité d'analyse: (1) réponses MTP du candidat, (2) informations personnelles pertinentes, (3) alignement avec la stratégie/vision SEEG. "
                        "Rends un avis clair, précis et concis. Ne répète pas les champs bruts; synthétise en 6–10 lignes max: verdict (meilleur candidat ou non), raisons (adéquation/complétude), risques/manquements, recommandations concrètes et prochaines étapes."
                    )
                    user_prompt = (
                        f"Profil: {name}\n"
                        f"Poste: {poste}\n"
                        f"Scores: final={final:.1f}, fit={fit:.1f}, completeness={comp:.1f}\n"
                        f"Recommendation: {reco}\n"
                        f"Flags manquants: {', '.join([k for k, v in flags_d.items() if not bool(v)]) if flags_d else 'aucun'}\n"
                        f"Vision entreprise (résumé): {get_settings().vision_text[:600]}\n"
                        + ("Fiche de poste — résumé:\n" + job_summary + "\n" if job_summary else "")
                        + ("MTP — questions clés:\n" + "\n".join([f"- {x}" for x in mtp_summary]) + "\n" if mtp_summary else "")
                        + ("MTP — réponses (extraits):\n" + "\n".join([f"- {x}" for x in mtp_resp_summary]) + "\n" if mtp_resp_summary else "")
                        + ("Documents — résumés:\n" + "\n".join(docs_summary_lines) + "\n" if docs_summary_lines else "")
                        + ("Documents — extraits RAG:\n" + "\n".join(rag_lines) + "\n" if rag_lines else "")
                        + "Produit l'analyse au format décisionnel, en puces courtes."
                    )
                    try:
                        # OpenAI Python SDK v1 chat.completions
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.2,
                            max_tokens=350,
                        )
                        content = resp.choices[0].message.content if resp and resp.choices else None
                    except Exception:
                        # Fallback to responses API if needed
                        content = None
                    if content:
                        st.markdown(content)
                    else:
                        st.info("Impossible d'obtenir une réponse IA pour le moment.")
                except Exception as e:
                    st.info(f"IA désactivée/indisponible: {e}")

        # Note: génération multi-candidats retirée à la demande — analyse uniquement pour le candidat sélectionné (par application_id)

    # Explications détaillées si disponibles, sinon heuristique simple
    d = row.get("details") if "details" in row.index else None
    strengths = weaknesses = keywords = explanations = None
    if isinstance(d, dict):
        explanations = d.get("explanations") or d.get("rationale") or d.get("analysis")
        strengths = d.get("strengths") or d.get("highlights") or d.get("top_skills")
        weaknesses = d.get("weaknesses") or d.get("gaps") or d.get("missing_skills")
        keywords = d.get("keywords") or d.get("top_keywords")

    # has_any = False
    # if any([explanations, strengths, weaknesses, keywords]):
    #     has_any = True
    #     st.subheader("Explications du modèle")
    #     if explanations:
    #         if isinstance(explanations, (list, tuple)):
    #             st.markdown("\n".join([f"- {str(x)}" for x in explanations if x is not None and str(x).strip()]))
    #         else:
    #             st.markdown(str(explanations))
    #     if strengths:
    #         st.markdown("**Points forts identifiés :**")
    #         if isinstance(strengths, (list, tuple)):
    #             st.markdown("\n".join([f"- {str(x)}" for x in strengths if x is not None and str(x).strip()]))
    #         else:
    #             st.markdown(f"- {str(strengths)}")
    #     if weaknesses:
    #         st.markdown("**Axes d'amélioration :**")
    #         if isinstance(weaknesses, (list, tuple)):
    #             st.markdown("\n".join([f"- {str(x)}" for x in weaknesses if x is not None and str(x).strip()]))
    #         else:
    #             st.markdown(f"- {str(weaknesses)}")
    #     if keywords:
    #         st.markdown("**Mots-clés pertinents :**")
    #         if isinstance(keywords, (list, tuple, set)):
    #             st.markdown(", ".join([str(x) for x in keywords if x is not None and str(x).strip()]))
    #         else:
    #             st.markdown(str(keywords))

    # if not has_any:
    #     # Heuristique courte basée sur scores/flags
    #     comp = float(row.get("completeness", 0) or 0)
    #     fit = float(row.get("fit", 0) or 0)
    #     final = float(row.get("final", 0) or 0)
    #     flags = {}
    #     if isinstance(d, dict) and isinstance(d.get("flags"), dict):
    #         flags = d.get("flags", {})
    #     auto = []
    #     if fit >= 75:
    #         auto.append("- Le contenu du dossier correspond bien aux exigences du poste (haute adéquation).")
    #     elif fit >= 55:
    #         auto.append("- Adéquation correcte avec le poste, mais certains éléments pourraient mieux correspondre.")
    #     else:
    #         auto.append("- Adéquation au poste faible; le profil semble éloigné des attentes exprimées.")
    #     if comp < 50:
    #         missing = [k for k, v in (flags or {}).items() if not bool(v)]
    #         if missing:
    #             auto.append("- Dossier incomplet (manque: " + ", ".join(sorted(set(missing))) + ").")
    #         else:
    #             auto.append("- Dossier incomplet; ajouter des éléments (CV, LM, diplômes, réponses MTP) pourrait aider.")
    #     if final >= 80:
    #         auto.append("- Très bon candidat selon le modèle.")
    #     elif final >= 60:
    #         auto.append("- Candidat à considérer avec compléments éventuels.")
    #     else:
    #         auto.append("- Candidat actuellement non recommandé au vu des critères du modèle.")
    #     with st.expander("Explications du modèle"):
    #         st.markdown("\n".join(auto))


    # (Bloc Insights résiduel supprimé)

 
def _slugify(text: str) -> str:
    text = str(text)
    text = (
        text.replace("'", " ")
        .replace("’", " ")
        .replace("é", "e").replace("è", "e").replace("ê", "e")
        .replace("à", "a").replace("â", "a")
        .replace("î", "i").replace("ï", "i")
        .replace("ô", "o").replace("ö", "o")
        .replace("û", "u").replace("ü", "u")
    )
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-+", "-", text)
    return text


def page_mtp():
    st.title("MTP — Gestion des questions")
    sqlite_db = st.session_state.get("sqlite_db", "scores.db")
    conn = get_sqlite(sqlite_db)
    ensure_sqlite_mtp_schema(conn)

    # Import prédéfini depuis scripts/mtp_questions.json
    st.subheader("Import JSON des questions (préparé)")
    col_i1, col_i2 = st.columns([1,2])
    with col_i1:
        if st.button("Importer jeu de questions fourni"):
            try:
                json_path = os.path.join(PROJECT_ROOT, "scripts", "mtp_questions.json")
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                res = import_mtp_questions_from_json(conn, data, active=1)
                st.success(f"Import réussi: {res.get('imported', 0)} questions")
                st.rerun()
            except Exception as e:
                st.error(f"Échec import: {e}")
    with col_i2:
        st.info("Le fichier source est `scripts/mtp_questions.json`. Modifiez-le puis ré-importez si besoin.")

    st.subheader("Ajouter / Mettre à jour une question")
    with st.form("mtp_upsert"):
        poste = st.text_input("Poste", placeholder="ex: Direction des Systèmes d'Informations")
        dimension = st.selectbox("Dimension", options=["metier", "talent", "paradigme"])
        qorder = st.number_input("Ordre (qN)", min_value=1, max_value=99, value=1)
        qtext = st.text_area("Intitulé de la question")
        code_custom = st.text_input("Code (optionnel)", help="Laisser vide pour générer automatiquement")
        active = st.checkbox("Active", value=True)
        submitted = st.form_submit_button("Enregistrer")
        if submitted:
            try:
                code = code_custom.strip() if code_custom.strip() else f"{_slugify(poste)}.{dimension}.q{int(qorder)}"
                payload = {
                    "code": code,
                    "poste": poste.strip(),
                    "dimension": dimension,
                    "question_order": int(qorder),
                    "question": qtext.strip(),
                    "active": 1 if active else 0,
                }
                if not payload["poste"] or not payload["question"]:
                    st.warning("Poste et question sont obligatoires.")
                else:
                    upsert_mtp_question(conn, payload)
                    st.success(f"Question enregistrée: {code}")
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur sauvegarde: {e}")

    st.subheader("Questions existantes")
    qs = list_mtp_questions(conn, active_only=False)
    if not qs:
        st.info("Aucune question en base")
    else:
        dfq = pd.DataFrame(qs)
        st.dataframe(dfq.sort_values(["poste", "dimension", "question_order"]) , use_container_width=True)
        # Suppression rapide par code
        with st.expander("Supprimer une question"):
            code_del = st.text_input("Code à supprimer")
            if st.button("Supprimer"):
                try:
                    delete_mtp_question(conn, code_del.strip())
                    st.success("Supprimée")
                    st.rerun()
                except Exception as e:
                    st.error(f"Échec suppression: {e}")


PAGES = {
    "Accueil": page_home,
    "Candidat": page_candidate,
}


def main():
    st.sidebar.title("Navigation")
    pages = list(PAGES.keys())
    # Respecter ?page=... en query param
    try:
        qp_page = st.query_params.get("page")
    except Exception:
        qp_page = None
    default_page = qp_page if qp_page in pages else pages[0]
    page = st.sidebar.radio("Page", pages, index=pages.index(default_page))

    # Scoring controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Calcul des scores")
    sqlite_db = st.sidebar.text_input("Base SQLite (.db)", value=st.session_state.get("sqlite_db", "scores.db"))
    st.session_state["sqlite_db"] = sqlite_db
    limit = st.sidebar.number_input("Limite", min_value=10, max_value=2000, value=200, step=10)
    dry = st.sidebar.checkbox("Dry-run (sans upsert)", value=False)
    if st.sidebar.button("Exécuter le modèle"):
        with st.spinner("Exécution du scoring en cours..."):
            try:
                args = ["--limit", str(int(limit))]
                if dry:
                    args.append("--dry-run")
                if sqlite_db:
                    args += ["--sqlite-db", sqlite_db]
                rc = score_batch.main(args)
                st.success(f"Scoring terminé (code {rc}).")
            except Exception as e:
                st.error(f"Erreur d'exécution: {e}")
        # Refresh cached data
        try:
            load_data.clear()
        except Exception:
            pass
        st.rerun()

    if st.sidebar.button("Rafraîchir données"):
        try:
            load_data.clear()
        except Exception:
            pass
        st.rerun()
    PAGES[page]()


if __name__ == "__main__":
    main()
