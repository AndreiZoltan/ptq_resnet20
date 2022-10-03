import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(r"{}".format(PARENT_DIR))
sys.path.append(r"{}/{}".format(PARENT_DIR, "src"))
