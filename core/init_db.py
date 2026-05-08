import sys
from pathlib import Path

# Tell Python where the root of the project is
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.database import engine
from core.models import Base

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Done!")