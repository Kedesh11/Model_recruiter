import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def inject_global_style():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #ececec; border-radius: 12px;
            padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,.04);
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #000 !important;
        }
        [data-testid="stMetric"] * { color: #000; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def batch_scoring_expander(positions: pd.DataFrame, apps: pd.DataFrame, documents: pd.DataFrame, sqlite_db: str | None, get_scoring_service):
    with st.expander("Batch scoring (recalcul et enregistrement)", expanded=False):
        disabled = get_scoring_service is None
        if disabled:
            st.caption("Service de scoring indisponible (module non importable)")
        # Sélecteur de poste optionnel
        pos_choices = ["Tous"]
        pos_map: dict[str, str] = {}
        try:
            if isinstance(positions, pd.DataFrame) and not positions.empty and "id" in positions.columns:
                label_candidates = ["title", "job_title", "position_title", "name", "libelle"]
                lab_col = next((c for c in label_candidates if c in positions.columns), None)
                for _, r in positions.iterrows():
                    pid = str(r.get("id"))
                    lab = str(r.get(lab_col)) if lab_col else pid
                    label = f"{lab} ({pid})" if lab else pid
                    pos_choices.append(label)
                    pos_map[label] = pid
        except Exception:
            pass
        pos_sel = st.selectbox("Poste cible (optionnel)", options=pos_choices, index=0)
        target_pid = pos_map.get(pos_sel)
        if st.button("Lancer le recalcul batch", disabled=disabled, key="btn_batch_score"):
            try:
                svc = get_scoring_service(sqlite_db or "scores.db") if get_scoring_service else None
                if svc is None:
                    st.error("ScoringService indisponible")
                else:
                    n = svc.score_from_dataframes(apps, documents, positions, filter_position_id=target_pid)
                    st.success(f"Recalcul terminé: {n} candidatures scorées/persistées")
                    st.rerun()
            except Exception as e:
                st.error(f"Échec du recalcul batch: {e}")


def render_identity(identity: dict, row: pd.Series) -> None:
    """Affiche le bloc Informations personnelles pour un candidat."""
    st.subheader("Informations personnelles")
    def _pick(*keys):
        for k in keys:
            try:
                if k in row.index and pd.notna(row[k]) and str(row[k]).strip():
                    return row[k]
            except Exception:
                continue
        return None
    cols = st.columns(4)
    with cols[0]:
        st.write("Nom:", (identity or {}).get("name") or _pick("name", "first_name", "prenom"))
    with cols[1]:
        st.write("Email:", (identity or {}).get("email") or _pick("email", "mail"))
    with cols[2]:
        st.write("Téléphone:", (identity or {}).get("phone") or _pick("phone", "mobile", "telephone"))
    with cols[3]:
        st.write("Matricule:", (identity or {}).get("matricule") or _pick("matricule"))


def _indicator_fig(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=float(value or 0.0), title={'text': title}, gauge={'axis': {'range': [0, 100]}}))
    return fig


def plot_score_gauges_row(completeness: float, fit: float, final: float, adherence: float, compliance: float) -> None:
    """Affiche en une rangée les 5 jauges principales."""
    st.subheader("Scores")
    figs = [
        _indicator_fig(completeness, "Complétude"),
        _indicator_fig(fit, "Adéquation"),
        _indicator_fig(final, "Score final"),
        _indicator_fig(adherence, "Adhérence MTP"),
        _indicator_fig(compliance, "Conformité du dossier"),
    ]
    cols = st.columns(5)
    for i, f in enumerate(figs):
        cols[i].plotly_chart(f, use_container_width=True)


def render_compliance_checklist(required: list[str], present: dict, has_certificate: bool, compliance_score: float) -> None:
    """Affiche la checklist de conformité (CV, diplôme, LM, etc.)."""
    label_map = {"cv": "CV", "diploma": "Diplôme", "lm": "Lettre de motivation"}
    lines = []
    for k in required:
        ok = bool(present.get(k, False))
        lines.append(("✅" if ok else "❌") + f" {label_map.get(k, k)}")
    lines.append(("✅" if has_certificate else "➖") + " Certificat supplémentaire (facultatif)")
    lines.append(f"Score de conformité: {float(compliance_score or 0):.0f}%")
    st.markdown("\n".join([f"- {x}" for x in lines]))


def render_completeness_radar(flags: dict) -> None:
    """Affiche un radar de complétude à partir d'un dict de flags booléens."""
    if not isinstance(flags, dict) or not flags:
        return
    st.subheader("Composantes de complétude")
    axes = list(flags.keys())
    vals = [100 if bool(flags[k]) else 0 for k in axes]
    vals.append(vals[0])
    axes.append(axes[0])
    fig = go.Figure(data=go.Scatterpolar(r=vals, theta=axes, fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
    st.plotly_chart(fig, width='stretch')
