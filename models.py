from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class QueryBody(BaseModel):
    query: str
    k: int = 4
    history: Optional[List[dict]] = None
    model: Optional[str] = None
    org: Optional[str] = None
    category: Optional[str] = None
    doc_code: Optional[str] = None
    owner: Optional[str] = None


class ZipRequest(BaseModel):
    paths: List[str] = []


class IngestDocument(BaseModel):
    sp_site_id: Optional[str] = None
    sp_list_id: Optional[str] = None
    sp_item_id: Optional[str] = None
    sp_drive_id: Optional[str] = None
    sp_file_id: Optional[str] = None
    sp_web_url: str

    etag: Optional[str] = None
    version_label: Optional[str] = None

    title: Optional[str] = None
    doc_code: Optional[str] = None
    org_code: Optional[str] = None
    org: Optional[str] = None
    category: Optional[str] = None
    owner: Optional[str] = None
    version: Optional[str] = None
    revision_date: Optional[str] = None
    latest_review_date: Optional[str] = None
    document_review_date: Optional[str] = None
    review_approval_date: Optional[str] = None
    keywords: Optional[List[str]] = None
    enterprise_keywords: Optional[List[str]] = None
    association_ids: Optional[List[str]] = None
    domain: Optional[str] = "HR"
    allowed_groups: Optional[List[str]] = None

    file_name: Optional[str] = None
    content_bytes: Optional[str] = None
    text_content: Optional[str] = None

    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    persist: Optional[bool] = False
