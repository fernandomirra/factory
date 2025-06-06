# Bootstrap script to recreate the Radar SAP MVP project on Windows.
# Usage:
#   1. Open PowerShell
#   2. Navigate to an empty directory
#   3. Run:  .\bootstrap_project.ps1
# The script will re-create folders/files exactly as designed and then
# compress everything into radar-sap.zip for easy sharing.

$ErrorActionPreference = 'Stop'

$root = "radar-sap"
if (Test-Path $root) { Remove-Item $root -Recurse -Force }

# Helper to write file safely
function Write-File($Path, $Content) {
  $dir = Split-Path $Path -Parent
  if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  Set-Content -Path $Path -Value $Content -NoNewline -Encoding UTF8
}

# ---------- root-level files ----------
Write-File "$root\requirements.txt" @"
fastapi==0.111.0
uvicorn[standard]==0.30.0
sqlmodel==0.0.16
asyncpg==0.29.0
psycopg[binary]==3.1.19
elasticsearch==8.12.1
pika==1.3.2
playwright==1.44.0
pydantic-settings==2.3.1
python-dotenv==1.0.1
pyyaml==6.0.1
streamlit==1.31.0

# Dev
aiofiles==23.2.1
pytest==8.2.0
ruff==0.4.4
mypy==1.10.0
types-requests==2.31.0.20240406
alembic==1.13.1
"@

Write-File "$root\.env" @"
DATABASE_URL=postgresql+psycopg://radar:radar@localhost:5432/radar
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
ELASTICSEARCH_URL=http://localhost:9200
API_BASE=http://localhost:8000
"@

Write-File "$root\Makefile" @"
# Makefile – developer shortcuts for MVP

PROJECT_NAME = radar-sap

.PHONY: dev stop logs lint test type-check build clean

## Spin up local stack (docker + API hot-reload)
dev:
	docker compose up -d
	@echo "✅ Stack is up. API on http://localhost:8000/docs"

## Stop and clean containers
stop:
	docker compose down --volumes --remove-orphans

## Tail logs of all services
logs:
	docker compose logs -f

## Lint with ruff
lint:
	ruff src tests

## Run unit tests
test:
	pytest -q

## Static typing
type-check:
	mypy src

## Build docker image
build:
	docker build -t $(PROJECT_NAME):local .

## Remove __pycache__ and other artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
"@

Write-File "$root\docker-compose.yml" @"
version: "3.9"

services:
  postgres:
    image: postgres:15-alpine
    container_name: radar_postgres
    environment:
      POSTGRES_USER: radar
      POSTGRES_PASSWORD: radar
      POSTGRES_DB: radar
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "radar"]
      interval: 10s
      timeout: 5s
      retries: 5

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.1
    container_name: radar_es
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - esdata:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9200/_cluster/health"]
      interval: 20s
      timeout: 10s
      retries: 5

  rabbitmq:
    image: rabbitmq:3-management-alpine
    container_name: radar_rmq
    ports:
      - "5672:5672"   # AMQP
      - "15672:15672" # HTTP management UI
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
  esdata:
"@

Write-File "$root\README.md" @"
# Radar Inteligente de Oportunidades SAP

## Visão Geral
Plataforma automatizada que detecta sinais públicos (LinkedIn, vagas) indicando oportunidades de projetos SAP.

## Instalação

### Pré-requisitos
- Python 3.11+
- Docker e Docker Compose
- Playwright

### Setup
```bash
# Instalar dependências
python -m pip install -r requirements.txt

# Instalar browsers Playwright
python -m playwright install chromium

# Iniciar serviços (Postgres, RabbitMQ, Elasticsearch)
docker compose up -d

# Criar tabelas
alembic revision --autogenerate -m "init tables"
alembic upgrade head
```

## Execução
```bash
# Terminal 1: Worker para persistência
python -m src.workers.raw_signal_consumer

# Terminal 2: Worker para classificação
python -m src.workers.classifier_worker

# Terminal 3: Crawler (gera sinais)
python -m src.ingest.linkedin_jobs --keywords "SAP S/4HANA" --location "Brazil"

# Terminal 4: Dashboard
streamlit run src.dashboard_app
```

## Acessando
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- RabbitMQ: http://localhost:15672 (guest/guest)
"@

# ---------- src files ----------
Write-File "$root\src\__init__.py" ""

Write-File "$root\src\db.py" @"
"""Database utilities: engine creation and helper for migrations."""
from __future__ import annotations

import os

from sqlmodel import SQLModel, create_engine

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://radar:radar@localhost:5432/radar")
engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)


def init_db() -> None:
    """Create tables if they don't exist (dev convenience)."""
    from src import models  # noqa: F401  # ensure models imported

    SQLModel.metadata.create_all(engine)
"@

Write-File "$root\src\models.py" @"
"""Shared SQLModel models for the Radar SAP MVP."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlmodel import Field, SQLModel


class RawSignal(SQLModel, table=True):
    """Table storing the unprocessed signals captured by crawlers."""

    id: str = Field(primary_key=True, index=True, description="UUID of the signal")
    source: str = Field(index=True, description="origin of the signal, e.g., 'linkedin_jobs'")
    keyword: Optional[str] = Field(default=None, index=True)
    captured_at: datetime = Field(description="Timestamp when the crawler captured the page")
    payload: Dict[str, Any] = Field(sa_column_kwargs={"type_": "jsonb"})
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class ClassifiedSignal(SQLModel, table=True):
    """Table storing signals after classification and enrichment."""

    id: str = Field(primary_key=True, index=True)
    raw_signal_id: str = Field(foreign_key="rawsignal.id", index=True)
    company_name: str = Field(index=True)
    company_id: Optional[str] = Field(default=None, index=True, description="CNPJ or external ID")
    score: int = Field(default=0, index=True, description="0-100 score, higher = more relevant")
    tags: list[str] = Field(default=[], sa_column_kwargs={"type_": "jsonb"})
    classification: str = Field(default="unclassified", index=True)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default={}, sa_column_kwargs={"type_": "jsonb"})
"@

Write-File "$root\src\main.py" @"
"""FastAPI entrypoint exposing classified signals.

Mounted at `src.main:app` (referenced in docker-compose).

Endpoints
---------
GET /health        → healthcheck
GET /signals       → list classified signals with filters & pagination

Future endpoints can be added under /signals/{id}, /export, etc.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from src.db import engine, init_db
from src.models import ClassifiedSignal

init_db()
app = FastAPI(title="Radar SAP – Signals API", version="0.1.0")


# Dependency -------------------------------------------------------------

def get_session():  # noqa: D401
    with Session(engine) as session:
        yield session


# Response schemas -------------------------------------------------------
class SignalOut(BaseModel):
    id: str
    company_name: str
    score: int
    classification: str
    tags: List[str]
    title: Optional[str] = None
    captured_at: Optional[str] = None
    url: Optional[str] = None


class PaginatedResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: List[SignalOut]


# Routes ----------------------------------------------------------------


@app.get("/health")
def health():  # noqa: D401
    return {"status": "ok"}


@app.get("/signals", response_model=PaginatedResponse)
def list_signals(
    min_score: int = Query(0, ge=0, description="Filter min score"),
    classification: Optional[str] = Query(None, regex="^(quente|morno|frio)$"),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None, description="Search by title or company"),
    session: Session = Depends(get_session),
):
    stmt = select(ClassifiedSignal).where(ClassifiedSignal.score >= min_score)
    if classification:
        stmt = stmt.where(ClassifiedSignal.classification == classification)
    if search:
        pattern = f"%{search.lower()}%"
        stmt = stmt.where(
            ClassifiedSignal.company_name.ilike(pattern)  # type: ignore[arg-type]
            | ClassifiedSignal.metadata["title"].as_string().ilike(pattern)  # type: ignore[index]
        )

    total = session.exec(select([ClassifiedSignal.id]).from_statement(stmt)).count()  # type: ignore[arg-type]
    stmt = stmt.order_by(ClassifiedSignal.score.desc()).offset(offset).limit(limit)
    results = session.exec(stmt).all()

    items: List[SignalOut] = []
    for s in results:
        title = None
        url = None
        captured_at = None
        # Raw info stored in metadata due to normalized model; adjust if changed
        if isinstance(s.metadata, dict):
            title = s.metadata.get("title")
            url = s.metadata.get("url")
            captured_at = s.metadata.get("captured_at")
        items.append(
            SignalOut(
                id=s.id,
                company_name=s.company_name,
                score=s.score,
                classification=s.classification,
                tags=s.tags,
                title=title,
                url=url,
                captured_at=captured_at,
            )
        )

    return PaginatedResponse(total=total, offset=offset, limit=limit, items=items)
"@

Write-File "$root\src\dashboard_app.py" @"
"""Streamlit dashboard for viewing classified SAP opportunity signals.

Run locally with:
    streamlit run src/dashboard_app.py

It queries the FastAPI backend (default http://localhost:8000) and displays
results in a paginated table with export CSV button.
"""
from __future__ import annotations

import io
import os
import requests
import pandas as pd
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
PAGE_LIMIT = 100

st.set_page_config(page_title="Radar SAP Dashboard", layout="wide")
st.title("📡 Radar SAP – Oportunidades Classificadas")

# Sidebar filters -------------------------------------------------------
with st.sidebar:
    st.header("Filtros")
    min_score = st.slider("Score mínimo", 0, 100, 30, 5)
    classification = st.multiselect("Classificação", ["quente", "morno", "frio"], default=["quente", "morno"])
    search = st.text_input("Buscar título ou empresa")
    st.markdown("---")

current_offset = st.session_state.get("offset", 0)

# Build query params
params = {
    "min_score": min_score,
    "limit": PAGE_LIMIT,
    "offset": current_offset,
}
if search:
    params["search"] = search
if classification and len(classification) < 3:
    # API only accepts single classification; request separately then concat
    dfs = []
    total = 0
    for cls in classification:
        params_cls = params.copy()
        params_cls["classification"] = cls
        r = requests.get(f"{API_BASE}/signals", params=params_cls, timeout=15)
        r.raise_for_status()
        data = r.json()
        dfs.append(pd.DataFrame(data["items"]))
        total += data["total"]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
else:
    if len(classification) == 1:
        params["classification"] = classification[0]
    r = requests.get(f"{API_BASE}/signals", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["items"])
    total = data["total"]

st.caption(f"Total sinais encontrados: {total}")

if df.empty:
    st.warning("Nenhum sinal encontrado com os filtros selecionados.")
else:
    df_sort = df.sort_values(by="score", ascending=False)
    st.dataframe(
        df_sort,
        use_container_width=True,
        column_config={
            "url": st.column_config.LinkColumn("Link"),
        },
    )

    # Export ------------------------------------------------------------
    csv_buf = io.StringIO()
    df_sort.to_csv(csv_buf, index=False)
    st.download_button("📥 Exportar CSV", data=csv_buf.getvalue(), file_name="signals.csv", mime="text/csv")

    # Pagination controls ----------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Anterior", disabled=current_offset == 0):
            st.session_state.offset = max(current_offset - PAGE_LIMIT, 0)
            st.experimental_rerun()
    with col2:
        st.markdown(f"Página **{current_offset // PAGE_LIMIT + 1}**")
    with col3:
        if st.button("Próxima ➡️", disabled=len(df) < PAGE_LIMIT):
            st.session_state.offset = current_offset + PAGE_LIMIT
            st.experimental_rerun()
"@

# ---------- ingest ----------
Write-File "$root\src\ingest\__init__.py" ""

Write-File "$root\src\ingest\linkedin_jobs.py" @"
"""LinkedIn Jobs Ingestor

Usage:
    python -m src.ingest.linkedin_jobs --keywords "SAP S/4HANA" "Consultor SAP" --location "Brazil" --max-pages 2

The script:
1. Launches a headless Chromium via Playwright.
2. Searches LinkedIn Jobs for each keyword/location page by page.
3. Extracts job cards (title, company, place, job url, listed_at).
4. Publishes each raw signal as JSON into RabbitMQ queue `raw_signals`.

Requirements:
    - Environment variable `RABBITMQ_URL` (default: amqp://guest:guest@localhost:5672/)
    - Optional LinkedIn authentication via cookie header: set `LINKEDIN_COOKIE` env var with `li_at=...;` string to bypass login wall.

NOTE: This PoC respects robots.txt rate-limits but LinkedIn may block scraping in production; consider the Partner Jobs API or legitimate data providers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import List

import pika  # type: ignore
from playwright.sync_api import Playwright, sync_playwright

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = "raw_signals"


class RabbitPublisher:
    """Lightweight wrapper around pika BlockingConnection."""

    def __init__(self, amqp_url: str, queue: str = QUEUE_NAME) -> None:
        params = pika.URLParameters(amqp_url)
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=queue, durable=True)
        self._queue = queue

    def publish(self, message: dict) -> None:
        self._channel.basic_publish(
            exchange="",
            routing_key=self._queue,
            body=json.dumps(message).encode(),
            properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),  # persistent
        )

    def close(self) -> None:
        if self._connection.is_open:
            self._connection.close()


def build_search_url(keyword: str, location: str | None = None, start: int = 0) -> str:
    base = "https://www.linkedin.com/jobs/search/?keywords="
    url = f"{base}{keyword.replace(' ', '%20')}"
    if location:
        url += f"&location={location.replace(' ', '%20')}"
    if start:
        url += f"&start={start}"
    return url


def scrape_page(page, keyword: str) -> List[dict]:
    """Return list of job dicts from the current loaded page."""
    cards = page.locator(".jobs-search-results__list-item")
    items: List[dict] = []
    for i in range(cards.count()):
        card = cards.nth(i)
        try:
            title = card.locator(".base-search-card__title").inner_text().strip()
            company = card.locator(".base-search-card__subtitle").inner_text().strip()
            place = card.locator(".job-search-card__location").inner_text().strip()
            link = card.locator("a.base-card__full-link").get_attribute("href") or ""
            posted = card.locator("time").get_attribute("datetime") or ""
        except Exception:  # noqa: BLE001
            continue
        items.append(
            {
                "id": str(uuid.uuid4()),
                "source": "linkedin_jobs",
                "keyword": keyword,
                "captured_at": datetime.now(tz=timezone.utc).isoformat(),
                "payload": {
                    "title": title,
                    "company": company,
                    "location": place,
                    "url": link,
                    "posted_at": posted,
                },
            }
        )
    return items


def run(playwright: Playwright, *, keywords: List[str], location: str | None, max_pages: int) -> None:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()

    # Optionally set auth cookie
    cookie_header = os.getenv("LINKEDIN_COOKIE")
    if cookie_header:
        context.add_init_script(
            "Object.defineProperty(document, 'cookie', { get: () => \"" + cookie_header + "\" });"
        )

    page = context.new_page()
    publisher = RabbitPublisher(RABBITMQ_URL)

    for kw in keywords:
        for page_idx in range(max_pages):
            url = build_search_url(kw, location, start=page_idx * 25)
            print(f"[crawler] GET {url}")
            page.goto(url, wait_until="domcontentloaded")
            # Scroll to load lazy content
            page.mouse.wheel(0, 5000)
            time.sleep(2)
            records = scrape_page(page, kw)
            print(f"[crawler] → {len(records)} jobs found (page {page_idx})")
            for rec in records:
                publisher.publish(rec)
            # Basic rate-limit
            time.sleep(3)

    publisher.close()
    browser.close()


def main(argv: List[str] | None = None) -> None:  # noqa: D401
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="LinkedIn Jobs crawler → RabbitMQ")
    parser.add_argument("--keywords", nargs="+", required=True)
    parser.add_argument("--location", default=None)
    parser.add_argument("--max-pages", type=int, default=1, help="Pages per keyword (25 results each)")
    args = parser.parse_args(argv)

    with sync_playwright() as pw:
        run(pw, keywords=args.keywords, location=args.location, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
"@

# ---------- workers ----------
Write-File "$root\src\workers\__init__.py" ""

Write-File "$root\src\workers\raw_signal_consumer.py" @"
"""RabbitMQ → PostgreSQL consumer for RawSignal.

Launch:
    python -m src.workers.raw_signal_consumer

Env vars (optional):
    RABBITMQ_URL   default amqp://guest:guest@localhost:5672/
    QUEUE_NAME     default raw_signals
    BATCH_SIZE     default 100  (flush db every N msgs)
"""
from __future__ import annotations

import json
import os
import signal
import sys
from contextlib import contextmanager
from typing import Any, Dict

import pika  # type: ignore
from sqlmodel import Session

from src.db import engine, init_db
from src.models import RawSignal

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = os.getenv("QUEUE_NAME", "raw_signals")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))


@contextmanager
def pika_connection(url: str):  # noqa: D401
    params = pika.URLParameters(url)
    conn = pika.BlockingConnection(params)
    try:
        yield conn
    finally:
        if conn.is_open:
            conn.close()


def persist_batch(batch: list[Dict[str, Any]]) -> None:
    if not batch:
        return
    with Session(engine) as session:
        objs = [RawSignal(**msg) for msg in batch]
        session.add_all(objs)
        session.commit()
    print(f"[worker] 🚀 persisted {len(batch)} signals to Postgres")


def main() -> None:  # noqa: D401
    print("[worker] starting raw_signal_consumer …")
    init_db()
    buffer: list[Dict[str, Any]] = []

    def handle_sigterm(sig, frame):  # noqa: D401
        print("[worker] SIGTERM received, flushing …")
        persist_batch(buffer)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    with pika_connection(RABBITMQ_URL) as conn:
        channel = conn.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)

        def callback(ch, method, properties, body):  # noqa: D401
            try:
                msg = json.loads(body)
                buffer.append(msg)
                if len(buffer) >= BATCH_SIZE:
                    persist_batch(buffer.copy())
                    buffer.clear()
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as exc:  # noqa: BLE001
                print(f"[worker] error processing msg: {exc}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        channel.basic_qos(prefetch_count=BATCH_SIZE)
        channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
        print(f"[worker] consuming from '{QUEUE_NAME}' … Ctrl+C to stop")
        try:
            channel.start_consuming()
        finally:
            persist_batch(buffer)


if __name__ == "__main__":
    main()
"@

Write-File "$root\src\workers\classifier_worker.py" @"
"""Worker that consumes raw_signals, applies regex rules, writes classified_signals."""
from __future__ import annotations

import json
import os
import re
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pika  # type: ignore
from pydantic import BaseModel
from sqlmodel import Session

from src.db import engine, init_db
from src.models import ClassifiedSignal, RawSignal

RULES_PATH = Path(os.getenv("RULES_PATH", "config/classifier_rules.yaml"))
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
INPUT_QUEUE = os.getenv("INPUT_QUEUE", "raw_signals")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

try:
    import yaml  # noqa: WPS433 external dep
except ImportError:  # pragma: no cover
    print("[classifier] missing PyYAML; install to run classifier")
    sys.exit(1)


class Rule(BaseModel):
    pattern: str
    tags: List[str]
    score: int
    regex: re.Pattern[str] | None = None

    def compile(self) -> None:  # noqa: D401
        self.regex = re.compile(self.pattern, re.IGNORECASE)

    def apply(self, title: str) -> tuple[int, list[str]]:  # noqa: D401
        if self.regex and self.regex.search(title):
            return self.score, self.tags.copy()
        return 0, []


def load_rules(path: Path) -> List[Rule]:
    data = yaml.safe_load(path.read_text())
    rules = [Rule(**item) for item in data]
    for r in rules:
        r.compile()
    return rules


RULES = load_rules(RULES_PATH)
print(f"[classifier] Loaded {len(RULES)} rules from {RULES_PATH}")


def classify(signal: RawSignal) -> ClassifiedSignal:
    total_score = 0
    all_tags: list[str] = []
    title = signal.payload.get("title", "") if isinstance(signal.payload, dict) else ""

    for rule in RULES:
        inc, tags = rule.apply(title)
        if inc:
            total_score += inc
            all_tags.extend(tags)

    classification = "frio"
    if total_score >= 60:
        classification = "quente"
    elif total_score >= 30:
        classification = "morno"

    return ClassifiedSignal(
        id=signal.id,
        raw_signal_id=signal.id,
        company_name=signal.payload.get("company", ""),
        score=total_score,
        tags=list(set(all_tags)),
        classification=classification,
        processed_at=datetime.utcnow(),
    )


def main() -> None:  # noqa: D401
    print("[classifier] starting classifier worker …")
    init_db()
    buffer: list[ClassifiedSignal] = []

    def flush():  # noqa: D401
        if not buffer:
            return
        with Session(engine) as session:
            session.add_all(buffer.copy())
            session.commit()
        print(f"[classifier] ✅ persisted {len(buffer)} classified signals")
        buffer.clear()

    def handle_term(sig, frame):  # noqa: D401
        print("[classifier] SIGTERM received, flushing …")
        flush()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    chan = conn.channel()
    chan.queue_declare(queue=INPUT_QUEUE, durable=True)

    def callback(ch, method, props, body):  # noqa: D401
        raw_dict: Dict[str, Any] = json.loads(body)
        raw = RawSignal(**raw_dict)
        classified = classify(raw)
        buffer.append(classified)
        if len(buffer) >= BATCH_SIZE:
            flush()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    chan.basic_qos(prefetch_count=BATCH_SIZE)
    chan.basic_consume(queue=INPUT_QUEUE, on_message_callback=callback)
    print(f"[classifier] consuming {INPUT_QUEUE} …")
    try:
        chan.start_consuming()
    finally:
        flush()
        conn.close()


if __name__ == "__main__":
    main()
"@

# ---------- config ----------
Write-File "$root\config\classifier_rules.yaml" @"
# Simple rule set for heuristic classifier

# Each rule has:
#   pattern: regex applied to payload.title (case-insensitive)
#   tags: list of tag strings
#   score: integer to add if rule matches

- pattern: "S/4HANA|S4HANA|S4 HANA"
  tags: ["s4hana", "migration"]
  score: 50

- pattern: "SAP BASIS|Basis Administrator"
  tags: ["basis"]
  score: 20

- pattern: "FiCO|FICO|Finance & Controlling"
  tags: ["fico"]
  score: 15

- pattern: "Roll-out|Deployment"
  tags: ["rollout"]
  score: 10

- pattern: "Arquiteto|Architect|Solution Manager"
  tags: ["architect"]
  score: 40

- pattern: "Migração|Migration|Conversão|Conversion"
  tags: ["migration"]
  score: 35

- pattern: "MM|Materials Management|Gestão de Materiais"
  tags: ["mm"]
  score: 15

- pattern: "SD|Sales & Distribution|Vendas"
  tags: ["sd"]
  score: 15

- pattern: "PP|Production Planning|Planejamento de Produção"
  tags: ["pp"]
  score: 15

- pattern: "HCM|Human Capital|Recursos Humanos|RH"
  tags: ["hcm"]
  score: 15

- pattern: "ABAP|Desenvolvedor|Developer"
  tags: ["abap", "development"]
  score: 10

- pattern: "BW|Business Warehouse|Hana Analytics"
  tags: ["bw", "analytics"]
  score: 20

- pattern: "CIO|CTO|Diretor TI|IT Director"
  tags: ["executive"]
  score: 30

- pattern: "ECC|R/3"
  tags: ["legacy"]
  score: 25

- pattern: "Implementação|Implementation|Implantação"
  tags: ["implementation"]
  score: 30

- pattern: "Fiori|UI5|User Experience|UX"
  tags: ["fiori", "ux"]
  score: 20

- pattern: "Cloud|S/4HANA Cloud|SAP Cloud"
  tags: ["cloud"]
  score: 25

- pattern: "Consultor Sênior|Senior Consultant|Especialista|Specialist"
  tags: ["senior"]
  score: 15

- pattern: "Projeto|Project|Programa|Program"
  tags: ["project"]
  score: 10
"@

# ---------- alembic ----------
Write-File "$root\src\alembic.ini" @"
# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = src/migrations

# template used to generate migration files
# file_template = %%(rev)s_%%(slug)s

# timezone to use when rendering the date
# within the migration file as well as the filename.
# string value is passed to dateutil.tz.gettz()
# leave blank for localtime
# timezone =

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version location specification; this defaults
# to src/migrations/versions.  When using multiple version
# directories, initial revisions must be specified with --version-path
# version_locations = %(here)s/bar %(here)s/bat src/migrations/versions

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = postgresql+psycopg://radar:radar@localhost:5432/radar

[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
# hooks=black
# black.type=console_scripts
# black.entrypoint=black
# black.options=-l 79

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"@

Write-File "$root\src\migrations\__init__.py" ""

Write-File "$root\src\migrations\env.py" @"
"""Alembic environment to autogenerate migrations from SQLModel."""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

# add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.db import DB_URL  # noqa: E402
from src import models  # noqa: F401, E402

# Alembic Config object
config = context.config
config.set_main_option("sqlalchemy.url", DB_URL)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:  # noqa: D401
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:  # noqa: D401
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
"@

Write-File "$root\src\migrations\script.py.mako" @"
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
"@

Write-File "$root\src\migrations\versions\__init__.py" ""

# ---------- devcontainer ----------
Write-File "$root\.devcontainer\devcontainer.json" @"
{
  "name": "Radar SAP Dev",
  "image": "mcr.microsoft.com/devcontainers/python:3.11-bullseye",
  "workspaceFolder": "/workspaces/radar-sap",
  "features": {
    "ghcr.io/devcontainers/features/docker-outside-of-docker:1": {},
    "ghcr.io/devcontainers/features/git:1": {},
    "ghcr.io/devcontainers/features/pipx:1": {}
  },
  "postCreateCommand": "pip install -r requirements.txt && playwright install chromium && docker compose up -d postgres rabbitmq elasticsearch",
  "forwardPorts": [
    5432,
    5672,
    15672,
    8000,
    8501
  ],
  "remoteUser": "vscode",
  "customizations": {
    "vscode": {
      "settings": {
        "python.pythonPath": "/usr/local/bin/python",
        "terminal.integrated.defaultProfile.linux": "bash"
      },
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "esbenp.prettier-vscode",
        "streetsidesoftware.code-spell-checker"
      ]
    }
  }
}
"@

# ---------- zip ----------
if (Test-Path "$root.zip") { Remove-Item "$root.zip" -Force }
Compress-Archive -Path $root -DestinationPath "$root.zip"
Write-Host "`nProject created and zipped to $root.zip" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Extract the ZIP" -ForegroundColor Cyan
Write-Host "2. Run 'docker compose up -d'" -ForegroundColor Cyan
Write-Host "3. Run 'python -m pip install -r requirements.txt'" -ForegroundColor Cyan
Write-Host "4. Run 'python -m playwright install chromium'" -ForegroundColor Cyan
Write-Host "5. Run 'alembic revision --autogenerate -m \"init tables\"'" -ForegroundColor Cyan
Write-Host "6. Run 'alembic upgrade head'" -ForegroundColor Cyan
Write-Host "7. Start workers and dashboard in separate terminals" -ForegroundColor Cyan
