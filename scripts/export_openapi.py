"""
Export both service contracts.

The order API contract is the interface between the Railway gateway and the
Vercel function. Exporting it in CI means a breaking change shows up as a diff
in a reviewable artifact rather than a 500 during a live call.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
for path in (
    ROOT / "packages" / "order-contracts",
    ROOT / "services" / "order-api",
    ROOT / "services" / "gateway",
):
    sys.path.insert(0, str(path))

from api.index import app as order_app  # noqa: E402
from app.main import app as gateway_app  # noqa: E402

out = ROOT / "contracts"
out.mkdir(exist_ok=True)
for name, application in (("openapi", order_app), ("gateway-openapi", gateway_app)):
    target = out / f"{name}.json"
    target.write_text(json.dumps(application.openapi(), indent=2))
    print(f"Wrote {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")
