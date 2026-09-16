"""Run this from the repository root to see exactly why a DL model isn't ready.

    python check_dl_handoff.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from deep_learning import DEEP_LEARNING_DIR, MANIFEST_PATH, load_deep_learning_handoff

print(f"Looking for manifest at: {MANIFEST_PATH}")
print(f"Manifest exists: {MANIFEST_PATH.is_file()}")
print(f"DL directory: {DEEP_LEARNING_DIR}")
print(f"DL directory exists: {DEEP_LEARNING_DIR.is_dir()}")
if DEEP_LEARNING_DIR.is_dir():
    print("Contents:")
    for path in sorted(DEEP_LEARNING_DIR.rglob("*")):
        print(f"  {path.relative_to(DEEP_LEARNING_DIR)}")

print()
mlp, ae = load_deep_learning_handoff()

for status in (mlp, ae):
    print(f"--- {status.name} ---")
    print(f"ready: {status.ready}")
    if status.messages:
        for message in status.messages:
            print(f"  - {message}")
    if status.metadata:
        print(f"  artifact: {status.metadata.get('artifact')}")
        print(f"  preprocessing_artifact: {status.metadata.get('preprocessing_artifact')}")
    print()