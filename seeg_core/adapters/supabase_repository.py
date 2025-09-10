from __future__ import annotations
from typing import Iterable, List

from seeg_core.ports.repositories import ReadRepository
from seeg_core.domain.candidate import Application, ApplicationDocument
from seeg_core.domain.job import JobOffer
from seeg_core.db import get_supabase


class SupabaseReadRepository(ReadRepository):
    """Implémentation ReadRepository via Supabase.

    Essaye des noms de tables alternatifs pour robustesse (compatibilité existante).
    """

    def __init__(self) -> None:
        self.sb = get_supabase()

    # --- Helpers ---
    def _fetch_table_first(self, candidates: List[str]) -> list[dict]:
        for t in candidates:
            try:
                data = self.sb.table(t).select("*").execute().data or []
                if isinstance(data, list):
                    return data
            except Exception:
                continue
        return []

    # --- API ---
    def list_applications(self) -> Iterable[Application]:
        rows = self._fetch_table_first(["applications", "candidatures", "application"])
        out: List[Application] = []
        for r in rows:
            app_id = str(r.get("id", ""))
            cand_id = str(r.get("candidate_id") or r.get("user_id") or "")
            job_id = str(r.get("job_offer_id") or r.get("position_id") or "")
            if not app_id:
                continue
            out.append(Application(id=app_id, candidate_id=cand_id, job_offer_id=job_id))
        return out

    def list_application_documents(self) -> Iterable[ApplicationDocument]:
        rows = self._fetch_table_first(["application_documents", "documents", "docs"])
        out: List[ApplicationDocument] = []
        link_col = None
        if rows:
            # detect link/url column
            for c in [
                "link",
                "url",
                "public_url",
                "download_url",
                "path",
                "storage_path",
                "file_path",
                "filename",
                "name",
            ]:
                if c in rows[0]:
                    link_col = c
                    break
        for r in rows:
            doc_id = str(r.get("id") or r.get("doc_id") or "")
            app_id = str(r.get("application_id") or r.get("app_id") or "")
            doc_type = str(r.get("type") or r.get("doc_type") or "document")
            txt = r.get("text")
            url = r.get(link_col) if link_col else None
            if not doc_id:
                # fallback to composed id
                doc_id = f"{app_id}:{link_col}:{url}" if (app_id and url) else (app_id or "")
            if not app_id:
                continue
            out.append(
                ApplicationDocument(
                    id=doc_id or app_id,
                    application_id=app_id,
                    doc_type=doc_type,
                    text=txt if isinstance(txt, str) else None,
                    url=str(url) if url else None,
                )
            )
        return out

    def list_job_offers(self) -> Iterable[JobOffer]:
        rows = self._fetch_table_first(["job_offers", "positions", "jobs", "offers"])
        out: List[JobOffer] = []
        for r in rows:
            jid = str(r.get("id", ""))
            title = r.get("title") or r.get("job_title") or r.get("name")
            desc = r.get("description") or r.get("desc")
            loc = r.get("location") or r.get("lieu")
            if not jid:
                continue
            out.append(JobOffer(id=jid, title=str(title) if title else None, description=str(desc) if desc else None, location=str(loc) if loc else None))
        return out
