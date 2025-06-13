from datetime import datetime
from typing import Any, Dict, Optional, List
from sqlmodel import SQLModel, Field
class RawSignal(SQLModel, table=True):
    id: str = Field(primary_key=True)
    source: str
    keyword: Optional[str] = None
    captured_at: datetime
    payload: Dict[str, Any] = Field(sa_column_kwargs={'type_':'jsonb'})
    created_at: datetime = Field(default_factory=datetime.utcnow)
class ClassifiedSignal(SQLModel, table=True):
    id: str = Field(primary_key=True)
    raw_signal_id: str = Field(foreign_key='rawsignal.id')
    company_name: str
    score: int = 0
    tags: List[str] = Field(default=[], sa_column_kwargs={'type_':'jsonb'})
    classification: str = 'frio'
    processed_at: datetime = Field(default_factory=datetime.utcnow)