import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is on sys.path to import local packages when running from streamlit_app/
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from seeg_core.db import get_supabase, get_sqlite, ensure_sqlite_scores_schema
from seeg_core.config import get_settings
import scripts.score_batch as score_batch

st.set_page_config(page_title="SEEG Recrutement", layout="wide")

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
    sqlite_db = st.session_state.get("sqlite_db")
    apps, scores, users, positions, documents, profiles = load_data(sqlite_db)
    df = build_enriched(apps, scores, users, positions, documents, profiles)

    # KPIs
    colk1, colk2, colk3 = st.columns(3)
    with colk1:
        st.metric("Nombre de candidatures", len(apps))
    with colk2:
        g = _derive_gender_series(users, profiles)
        gcount = g.value_counts().reindex(["Homme","Femme","Inconnu"], fill_value=0)
        fig_g = px.pie(values=gcount.values, names=gcount.index, title="Répartition H/F")
        st.plotly_chart(fig_g, width='stretch')
    with colk3:
        # nb candidats ayant postulé à plusieurs postes
        cand_key = "candidate_id" if "candidate_id" in apps.columns else "user_id" if "user_id" in apps.columns else None
        pos_key = "job_offer_id" if "job_offer_id" in apps.columns else "position_id" if "position_id" in apps.columns else None
        multi_cnt = 0
        if cand_key and pos_key:
            tmp = apps[[cand_key, pos_key]].dropna()
            multi_cnt = (tmp.groupby(cand_key)[pos_key].nunique() > 1).sum()
        st.metric("Candidats multi-postes", int(multi_cnt))

    # Filtres
    st.subheader("Liste des candidats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        name_q = st.text_input("Nom/Prénom contient")
    with col2:
        job_q = st.text_input("Poste contient")
    with col3:
        min_score = st.number_input("Score final min", min_value=0, max_value=100, value=0)
    with col4:
        max_score = st.number_input("Score final max", min_value=0, max_value=100, value=100)

    view = df.copy()
    name_cols = [c for c in ["first_name","last_name","prenom","nom","name"] if c in view.columns]
    if name_q and name_cols:
        mask = False
        for c in name_cols:
            mask = mask | view[c].astype(str).str.contains(name_q, case=False, na=False)
        view = view[mask]
    if job_q:
        for c in ["title", "job_title", "position_title"]:
            if c in view.columns:
                view = view[view[c].astype(str).str.contains(job_q, case=False, na=False)]
                break
    if "final" in view.columns:
        view = view[(view["final"].fillna(0) >= min_score) & (view["final"].fillna(0) <= max_score)]

    st.dataframe(view, width='stretch', height=480)


def page_candidate():
    st.title("Détail Candidat")
    sqlite_db = st.session_state.get("sqlite_db")
    apps, scores, users, positions, documents, profiles = load_data(sqlite_db)
    df = build_enriched(apps, scores, users, positions, documents, profiles)

    app_ids = df["id"].astype(str).tolist() if "id" in df.columns else []
    selected = st.selectbox("Sélectionner une candidature", options=app_ids)
    if not selected:
        st.info("Aucune candidature")
        return
    row = df[df["id"].astype(str) == selected].iloc[0]

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

    # Scores gauges
    st.subheader("Scores")
    gcols = st.columns(3)
    for i, metric in enumerate([("Complétude","completeness"), ("Adéquation","fit"), ("Score final","final")]):
        label, key = metric
        val = float(row.get(key, 0) or 0)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': label}, gauge={'axis': {'range': [0,100]}}))
        gcols[i].plotly_chart(fig, width='stretch')

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

    # Documents
    st.subheader("Documents")
    links = row.get("doc_links") if "doc_links" in row.index else None
    if isinstance(links, list) and links:
        for url in links:
            st.markdown(f"- [Ouvrir]({url})")
    else:
        st.info("Aucun document")


def page_insights():
    st.title("Insights")
    sqlite_db = st.session_state.get("sqlite_db")
    apps, scores, users, positions, _, profiles = load_data(sqlite_db)
    if scores.empty:
        st.info("Aucun score disponible")
        return

    # Distribution des scores finaux
    st.subheader("Distribution des scores finaux")
    fig_hist = px.histogram(scores, x=scores["final"].fillna(0), nbins=20, title="Histogramme des scores")
    st.plotly_chart(fig_hist, width='stretch')

    # Distribution par poste (box plot)
    st.subheader("Distribution des scores par poste")
    # relier scores -> applications -> positions
    df_join = apps.merge(scores, left_on="id", right_on="application_id", how="inner")
    pos_key = "job_offer_id" if "job_offer_id" in df_join.columns else "position_id" if "position_id" in df_join.columns else None
    title_col = None
    if not positions.empty:
        if pos_key and "id" in positions.columns:
            df_join = df_join.merge(positions[["id","title"]] if "title" in positions.columns else positions[["id"]], left_on=pos_key, right_on="id", how="left")
            if "title" in df_join.columns:
                title_col = "title"
    df_join["poste"] = df_join[title_col] if title_col else df_join.get(pos_key, "poste")
    fig_box = px.box(df_join, x="poste", y="final", points="all")
    st.plotly_chart(fig_box, width='stretch')

    # Classement des candidats (top 20)
    st.subheader("Top candidats")
    # récupérer identité depuis details.identity si existant
    def _name_from_details(d):
        if isinstance(d, dict) and isinstance(d.get("identity"), dict):
            return d["identity"].get("name")
        return None
    scores["candidate_name"] = scores["details"].apply(_name_from_details) if "details" in scores.columns else None
    top = scores.sort_values("final", ascending=False).head(20)
    fig_top = px.bar(top, x="candidate_name", y="final")
    st.plotly_chart(fig_top, width='stretch')

    # Indicateurs globaux
    st.subheader("Indicateurs globaux")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Taux moyen de complétude", f"{scores['completeness'].fillna(0).mean():.1f}%")
    with c2:
        st.metric("Score final moyen", f"{scores['final'].fillna(0).mean():.1f}%")
    with c3:
        reco_rate = (scores["recommendation"].astype(str) == "Fortement recommandé").mean() * 100 if "recommendation" in scores.columns else 0
        st.metric("% Fortement recommandés", f"{reco_rate:.1f}%")


PAGES = {
    "Accueil": page_home,
    "Candidat": page_candidate,
    "Insights": page_insights,
}


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Page", list(PAGES.keys()))

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
