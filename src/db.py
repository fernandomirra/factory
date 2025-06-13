from sqlmodel import SQLModel, create_engine
import os
engine = create_engine(os.getenv('DATABASE_URL','sqlite:///local.db'), echo=False)
def init_db():
    from src import models  # noqa
    SQLModel.metadata.create_all(engine)