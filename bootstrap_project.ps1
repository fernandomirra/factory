# -------- bootstrap_radar_sap.ps1 --------
$ErrorActionPreference = 'Stop'
$root = "radar-sap"
if (Test-Path $root) { Remove-Item $root -Recurse -Force }

function W ($p,$c) { $d=Split-Path $p -Parent; if(!(Test-Path $d)){New-Item -ItemType Directory -Path $d -Force|Out-Null}; Set-Content -Path $p -Value $c -NoNewline -Encoding UTF8 }

# = requirements.txt =
W "$root\requirements.txt" @"
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
pytest==8.2.0
ruff==0.4.4
mypy==1.10.0
types-requests==2.31.0.20240406
alembic==1.13.1
"@

# = .env =
W "$root\.env" @"
DATABASE_URL=postgresql+psycopg://radar:radar@localhost:5432/radar
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
ELASTICSEARCH_URL=http://localhost:9200
API_BASE=http://localhost:8000
"@

# = docker-compose.yml =
W "$root\docker-compose.yml" @"
version: '3.9'
services:
  postgres:
    image: postgres:15-alpine
    ports: ['5432:5432']
    environment:
      POSTGRES_USER: radar
      POSTGRES_PASSWORD: radar
      POSTGRES_DB: radar
  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports: ['5672:5672','15672:15672']
"@

# = Makefile (atalhos básicos) =
W "$root\Makefile" @"
dev: ; docker compose up -d
stop: ; docker compose down --volumes --remove-orphans
logs: ; docker compose logs -f
"@

# = src/db.py =
W "$root\src\db.py" @"
from sqlmodel import SQLModel, create_engine
import os
engine = create_engine(os.getenv('DATABASE_URL','sqlite:///local.db'), echo=False)
def init_db():
    from src import models  # noqa
    SQLModel.metadata.create_all(engine)
"@

# = src/models.py =
W "$root\src\models.py" @"
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
"@

# = src/main.py (API mínima p/ saúde) =
W "$root\src\main.py" @"
from fastapi import FastAPI
from src.db import init_db
init_db()
app = FastAPI()
@app.get('/health')
def health(): return {'status':'ok'}
"@

# (resto dos arquivos: workers, crawler, dashboard, rules YAML)
# Para manter o script curto, se precisar de todos os módulos avançados
# copie-os do exemplo completo já enviado ou acrescente depois.

# zip opcional
Compress-Archive -Path $root -DestinationPath "$root.zip"
Write-Host \"Projeto gerado em $root e zip criado $root.zip\" -ForegroundColor Green
# -------- fim do script ---------