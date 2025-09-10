from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import re

from seeg_core.domain.candidate import Application, ApplicationDocument
from seeg_core.domain.job import JobOffer
from seeg_core.utils.logging import get_logger
from seeg_core.nlp import st_text_vector, cosine_sim


logger = get_logger(__name__)


@dataclass
class Features:
    application_id: str
    text_length: int
    nb_documents: int
    keyword_hits: int
    job_keywords: List[str]
    keyword_coverage: int = 0
    jaccard_title_sim: float = 0.0
    semantic_sim: float = 0.0  # cosine similarity between job text and candidate text (0..1)
    doc_types: List[str] = None  # normalized doc types present

    def to_dict(self) -> Dict[str, float | int | str | List[str]]:
        return {
            "application_id": self.application_id,
            "text_length": int(self.text_length),
            "nb_documents": int(self.nb_documents),
            "keyword_hits": int(self.keyword_hits),
            "job_keywords": list(self.job_keywords),
            "keyword_coverage": int(self.keyword_coverage),
            "jaccard_title_sim": float(self.jaccard_title_sim),
            "semantic_sim": float(self.semantic_sim),
            "doc_types": list(self.doc_types or []),
        }


class FeaturesService:
    """Extraction de features basiques à partir des documents de candidature et de l'offre.

    Cette implémentation volontairement simple servira de base, remplaçable par
    une version plus avancée (TF-IDF, embeddings, RAG, etc.).
    """

    def __init__(self) -> None:
        pass

    def _extract_keywords_from_job(self, job: Optional[JobOffer]) -> List[str]:
        if job is None:
            return []
        text = " ".join(
            [
                str(job.title or ""),
                str(job.description or ""),
                str(job.location or ""),
            ]
        ).lower()
        # heuristique simple: top tokens alphanumériques de 4+ caractères
        toks = re.findall(r"[a-zA-Zàâäéèêëîïôöùûüç0-9]{4,}", text)
        uniq = []
        seen = set()
        for t in toks:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
            if len(uniq) >= 15:
                break
        return uniq

    def _tokens(self, text: str) -> Set[str]:
        toks = re.findall(r"[a-zA-Zàâäéèêëîïôöùûüç0-9]{3,}", (text or "").lower())
        return set(toks)

    def _norm_doc_type(self, val: str) -> Optional[str]:
        if not isinstance(val, str):
            return None
        s = val.strip().lower()
        if not s:
            return None
        if any(k in s for k in ["cv", "resume", "curriculum"]):
            return "cv"
        if any(k in s for k in ["diploma", "diplome", "degree", "licence", "master", "mba", "bachelor"]):
            return "diploma"
        if any(k in s for k in ["lettre", "motivation", "cover-letter", "cover_letter", "coverletter"]):
            return "lm"
        return None

    def compute(self, app: Application, docs: List[ApplicationDocument], job: Optional[JobOffer]) -> Features:
        # concat texte des documents (si présent)
        texts = [d.text for d in docs if isinstance(d.text, str) and d.text.strip()]
        full = "\n\n".join(texts)
        kwds = self._extract_keywords_from_job(job)
        # compter occurrences simples de mots-clés (sans pondération)
        low = full.lower()
        hits = 0
        for k in kwds:
            try:
                hits += low.count(k)
            except Exception:
                continue
        # couverture: combien de mots-clés uniques sont présents au moins une fois
        tokset = self._tokens(full)
        coverage = sum(1 for k in kwds if k in tokset)

        # similarité Jaccard entre titre de job et premiers 1000 chars du texte
        title = (job.title or "") if job else ""
        tA = self._tokens(title)
        tB = self._tokens(full[:1000]) if full else set()
        jac = (len(tA & tB) / len(tA | tB)) if (tA or tB) else 0.0

        # similarité sémantique (embeddings): job(title+description) vs texte candidate (tronqué)
        sem = 0.0
        try:
            job_text = ((job.title or "") + "\n" + (job.description or "")) if job else ""
            cand_text = full if len(full) <= 3000 else full[:2500] + " " + full[-500:]
            vj = st_text_vector(job_text)
            vc = st_text_vector(cand_text)
            sem_raw = cosine_sim(vj, vc)
            # cosine in [-1,1] -> clip to [0,1]
            sem = max(0.0, min((sem_raw + 1.0) / 2.0, 1.0))
        except Exception:
            sem = 0.0

        # types de documents détectés depuis les meta des docs
        dtypes: Set[str] = set()
        for d in docs:
            t = self._norm_doc_type(getattr(d, "doc_type", None) or getattr(d, "url", None) or "")
            if t:
                dtypes.add(t)

        f = Features(
            application_id=app.id,
            text_length=len(full),
            nb_documents=len(texts),
            keyword_hits=hits,
            job_keywords=kwds,
            keyword_coverage=coverage,
            jaccard_title_sim=float(jac),
            semantic_sim=float(sem),
            doc_types=sorted(list(dtypes)),
        )
        logger.debug("features_computed", extra={"extra": f.to_dict()})
        return f


def get_features_service() -> FeaturesService:
    return FeaturesService()
