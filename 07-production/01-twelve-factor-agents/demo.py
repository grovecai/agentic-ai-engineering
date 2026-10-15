"""Explicit configuration, stateless request, and narrow tool contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service import Settings, SupportService

if __name__ == "__main__":
    print(SupportService(Settings.from_env()).answer("Can I get a refund after 20 days?"))
