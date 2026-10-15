"""Bound input and keep the tool registry fixed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service import Settings, SupportService, authorize, lookup

if __name__ == "__main__":
    print("Unknown tool target:", lookup("../../.env"))
    try:
        authorize("wrong", "expected")
    except PermissionError:
        print("Rejected an invalid application key")
    service = SupportService(Settings.from_env())
    print(service.answer("What is the refund window?"))
