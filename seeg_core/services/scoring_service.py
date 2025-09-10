from __future__ import annotations
from typing import List, Optional, Dict

import pandas as pd

from seeg_core.domain.candidate import Application, ApplicationDocument
from seeg_core.domain.job import JobOffer
from seeg_core.domain.score import Score
from seeg_core.services.features_service import FeaturesService, get_features_service
from seeg_core.adapters.sqlite_repository import SQLiteWriteRepository
from seeg_core.utils.logging import get_logger


logger = get_logger(__name__)


class ScoringService:
    """Service de scoring simple basé sur des features heuristiques.

    Cette implémentation est minimaliste pour permettre un branchement progressif.
    Elle pourra être remplacée par un moteur plus avancé (ML, RAG, LLM, etc.).
    """

    def __init__(self, features: Optional[FeaturesService] = None, sqlite_db: str = "scores.db") -> None:
        self.features = features or get_features_service()
        self.writer = SQLiteWriteRepository(sqlite_db)

    def _compute_from_features(
        self,
        ftext_len: int,
        nb_docs: int,
        kw_hits: int,
        kw_coverage: int = 0,
        jaccard_title_sim: float = 0.0,
        semantic_sim: float = 0.0,
        doc_types: Optional[List[str]] = None,
    ) -> tuple[float, float, float, str]:
        # Complétude (0..1): texte + nb docs + présence pièces clés
        comp_text = min(ftext_len / 5000.0, 1.0)   # 5k chars ~ complet
        comp_docs = min(nb_docs / 3.0, 1.0)        # 3 docs ~ complet
        doc_types = doc_types or []
        req = {"cv", "diploma", "lm"}
        have = len(req & set(doc_types))
        comp_docs_types = have / max(1, len(req))
        completeness = max(0.0, min(0.5 * comp_text + 0.2 * comp_docs + 0.3 * comp_docs_types, 1.0))

        # Fit (0..1): sémantique (embeddings), occurrences, couverture, similarité titre
        fit_hits = min(kw_hits / 20.0, 1.0)             # 20 hits => max
        fit_cov = min(kw_coverage / 10.0, 1.0)          # 10 mots-clés couverts => max
        fit_title = max(0.0, min(jaccard_title_sim * 4.0, 1.0))  # Jaccard 0.25 ~ max
        fit_sem = max(0.0, min(semantic_sim, 1.0))
        fit = max(0.0, min(0.35 * fit_hits + 0.15 * fit_cov + 0.1 * fit_title + 0.4 * fit_sem, 1.0))

        # Score final
        final = max(0.0, min(0.5 * completeness + 0.5 * fit, 1.0))

        # Recommandation
        if final >= 0.75:
            rec = "Avancer en entretien"
        elif final >= 0.5:
            rec = "À approfondir"
        else:
            rec = "Réserve"
        return completeness, fit, final, rec

    def score_application(self, app: Application, docs: List[ApplicationDocument], job: Optional[JobOffer]) -> Score:
        f = self.features.compute(app, docs, job)
        completeness, fit, final, rec = self._compute_from_features(
            f.text_length, f.nb_documents, f.keyword_hits, f.keyword_coverage, f.jaccard_title_sim, f.semantic_sim, f.doc_types
        )
        sc = Score(
            application_id=app.id,
            completeness=float(completeness * 100.0),
            fit=float(fit * 100.0),
            final=float(final * 100.0),
            recommendation=rec,
        )
        logger.debug(
            "score_computed",
            extra={
                "extra": {
                    "application_id": app.id,
                    "completeness": completeness,
                    "fit": fit,
                    "final": final,
                    "recommendation": rec,
                }
            },
        )
        return sc

    def score_and_persist(self, app: Application, docs: List[ApplicationDocument], job: Optional[JobOffer]) -> Score:
        sc = self.score_application(app, docs, job)
        # Persiste dans SQLite (idempotent par application_id)
        self.writer.upsert_score(sc)
        logger.info("score_persisted", extra={"extra": {"application_id": app.id, "final": sc.final}})
        return sc

    # ------------------------- Batch scoring helpers -------------------------
    def _build_maps_from_frames(
        self,
        apps_df: pd.DataFrame,
        documents_df: pd.DataFrame,
        positions_df: pd.DataFrame,
    ) -> tuple[List[Application], Dict[str, List[ApplicationDocument]], Dict[str, JobOffer]]:
        apps_list: List[Application] = []
        docs_map: Dict[str, List[ApplicationDocument]] = {}
        jobs_map: Dict[str, JobOffer] = {}

        # Applications
        for _, r in apps_df.iterrows():
            app_id = str(r.get("id"))
            cand_key = "candidate_id" if "candidate_id" in apps_df.columns else ("user_id" if "user_id" in apps_df.columns else None)
            pos_key = "job_offer_id" if "job_offer_id" in apps_df.columns else ("position_id" if "position_id" in apps_df.columns else None)
            apps_list.append(Application(
                id=app_id,
                candidate_id=str(r.get(cand_key)) if cand_key else "",
                job_offer_id=str(r.get(pos_key)) if pos_key else "",
            ))

        # Documents groupés
        if isinstance(documents_df, pd.DataFrame) and not documents_df.empty and "application_id" in documents_df.columns:
            type_cols = [c for c in ["type", "doc_type", "category", "kind", "label", "document_type", "name"] if c in documents_df.columns]
            text_col = "text" if "text" in documents_df.columns else None
            url_col = None
            for c in ["link", "url", "public_url", "download_url", "path", "storage_path", "file_path", "filename", "name"]:
                if c in documents_df.columns:
                    url_col = c
                    break
            for app_id, g in documents_df.groupby(documents_df["application_id"].astype(str)):
                cur: List[ApplicationDocument] = []
                for idx, r in g.iterrows():
                    doc_id = str(r.get("id") or f"doc_{idx}")
                    dtype = "unknown"
                    for c in type_cols:
                        v = str(r.get(c) or "").strip()
                        if v:
                            dtype = v
                            break
                    txt = str(r.get(text_col)) if text_col else None
                    url = str(r.get(url_col)) if url_col else None
                    cur.append(ApplicationDocument(id=doc_id, application_id=str(app_id), doc_type=dtype, text=txt or None, url=url or None))
                docs_map[str(app_id)] = cur

        # Jobs map
        if isinstance(positions_df, pd.DataFrame) and not positions_df.empty and "id" in positions_df.columns:
            title_candidates = ["title", "job_title", "position_title", "name", "libelle"]
            for _, r in positions_df.iterrows():
                pos_id = str(r.get("id"))
                title = None
                for c in title_candidates:
                    if c in positions_df.columns and isinstance(r.get(c), str) and r.get(c).strip():
                        title = r.get(c)
                        break
                desc = r.get("description") if "description" in positions_df.columns else None
                loc = r.get("location") if "location" in positions_df.columns else None
                jobs_map[pos_id] = JobOffer(id=pos_id, title=title, description=desc, location=loc)

        return apps_list, docs_map, jobs_map

    def score_from_dataframes(
        self,
        apps_df: pd.DataFrame,
        documents_df: pd.DataFrame,
        positions_df: pd.DataFrame,
        filter_position_id: Optional[str] = None,
    ) -> int:
        """Calcule et persiste les scores pour tout/applications filtrées. Retourne le nombre scoré."""
        if apps_df is None or len(apps_df) == 0:
            return 0
        apps_list, docs_map, jobs_map = self._build_maps_from_frames(apps_df, documents_df, positions_df)
        # Filtre optionnel
        if filter_position_id:
            pos_key = "job_offer_id" if "job_offer_id" in apps_df.columns else ("position_id" if "position_id" in apps_df.columns else None)
            if pos_key:
                apps_list = [a for a in apps_list if a.job_offer_id == str(filter_position_id)]
        count = 0
        for app in apps_list:
            docs = docs_map.get(app.id, [])
            job = jobs_map.get(app.job_offer_id)
            try:
                self.score_and_persist(app, docs, job)
                count += 1
            except Exception:
                continue
        return count


def get_scoring_service(sqlite_db: str = "scores.db") -> ScoringService:
    return ScoringService(sqlite_db=sqlite_db)
