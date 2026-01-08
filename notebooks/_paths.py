from pathlib import Path
import os

def set_repo_root() -> Path:
    root = Path.cwd()
    while root != root.parent and not (root / ".git").exists():
        root = root.parent
    os.chdir(root)
    return root
