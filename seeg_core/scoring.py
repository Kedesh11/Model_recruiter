from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from .nlp import text_vector, cosine_sim


def compute_completeness(flags: Dict[str, bool], weights: Optional[Dict[str, float]] = None) -> float:
    """Compute completeness score [0..100] from boolean flags and weights.
    Default weights: cv 0.30, lm 0.20, diploma 0.20, id 0.10, mtp 0.20
    """
    default = {'cv': 0.30, 'lm': 0.20, 'diploma': 0.20, 'id': 0.10, 'mtp': 0.20}
    w = weights or default
    total = sum(w.values()) or 1.0
    score = 0.0
    for k, p in w.items():
        score += (1.0 if bool(flags.get(k, False)) else 0.0) * p
    return round(100.0 * score / total, 2)


def compute_fit(candidate_text: str, job_text: str, vision_text: str, alpha_job: float = 0.6, alpha_vision: float = 0.4) -> float:
    """Compute fit score [0..100] using spaCy vectors and cosine similarity.
    fit = alpha_job * cos(cand, job) + alpha_vision * cos(cand, vision)
    """
    cand_vec = text_vector(candidate_text or "")
    job_vec = text_vector(job_text or "")
    vis_vec = text_vector(vision_text or "")
    sim_job = float(cosine_sim(cand_vec, job_vec))
    sim_vis = float(cosine_sim(cand_vec, vis_vec))
    fit = max(0.0, min(1.0, alpha_job * sim_job + alpha_vision * sim_vis))
    return round(100.0 * fit, 2)


def compute_final(completeness: float, fit: float, w_c: float = 0.4, w_f: float = 0.6) -> float:
    val = w_c * (float(completeness) / 100.0) + w_f * (float(fit) / 100.0)
    val = max(0.0, min(1.0, val))
    return round(100.0 * val, 2)


def recommend(final: float, t_strong: float = 80.0, t_consider: float = 60.0) -> str:
    if final >= t_strong:
        return "Fortement recommandé"
    if final >= t_consider:
        return "À considérer"
    return "Non recommandé"


def upsert_score_to_supabase(supabase, application_id: str, payload: Dict[str, Any]):
    data = {
        "application_id": application_id,
        "completeness": float(payload.get("completeness", 0.0)),
        "fit": float(payload.get("fit", 0.0)),
        "final": float(payload.get("final", 0.0)),
        "recommendation": payload.get("recommendation", ""),
        "details": payload.get("details", {}),
    }
    return supabase.table("scores").upsert(data, on_conflict="application_id").execute()
