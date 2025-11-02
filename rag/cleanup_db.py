import shutil
import os

CHROMA_PATH = "chroma_db"

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
    print(f"Deleted existing database at {CHROMA_PATH}")
else:
    print(f"No database found at {CHROMA_PATH}")

print("You can now run ingest_database.py to create a fresh database")