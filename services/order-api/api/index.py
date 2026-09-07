"""
Vercel entrypoint for the order service.

Vercel's Python runtime serves an exported ASGI app directly, so one file plus
a rewrite in vercel.json covers every route — far better than one handler file
per endpoint, which would duplicate the store bootstrap on each cold start.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent.parent / "packages" / "order-contracts"))

from fastapi import FastAPI  # noqa: E402
from order_api.router import router  # noqa: E402
from order_api.store import get_store  # noqa: E402

app = FastAPI(
    title="Order API",
    version="0.1.0",
    description="Stateless order lookup and mutation service. Deployed to Vercel.",
)
app.include_router(router)


@app.get("/api/health")
def health() -> dict:
    from order_api.kv import get_backend

    store = get_store()
    return {
        "status": "ok",
        "backend": type(get_backend()).__name__,
        "orders": len(store.all_ids()),
    }
