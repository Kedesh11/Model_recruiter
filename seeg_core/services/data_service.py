from __future__ import annotations
from typing import List, Tuple

from seeg_core.adapters.supabase_repository import SupabaseReadRepository
from seeg_core.domain.candidate import Application, ApplicationDocument
from seeg_core.domain.job import JobOffer


class DataService:
    """Service de lecture de données à partir de Supabase (read-only)."""

    def __init__(self) -> None:
        self.repo = SupabaseReadRepository()

    def list_core(self) -> Tuple[List[Application], List[ApplicationDocument], List[JobOffer]]:
        apps = list(self.repo.list_applications())
        docs = list(self.repo.list_application_documents())
        jobs = list(self.repo.list_job_offers())
        return apps, docs, jobs


def get_data_service() -> DataService:
    return DataService()
