import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "mini_infer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
