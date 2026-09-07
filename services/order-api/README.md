# Order API

Stateless order lookup and mutation service. Deployed to Vercel; genuinely
serverless-shaped — spiky call volume, no long-lived connections, no shared
process state.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/api/orders?phone=` | Caller's open orders — the fuzzy ID search space |
| GET | `/api/orders/{id}` | Full WMS record with tracking chain |
| POST | `/api/orders/{id}/reschedule` | Requires `Idempotency-Key` |
| POST | `/api/orders/{id}/address` | Requires `Idempotency-Key` |
| GET | `/api/health` | Reports which KV backend is live |

## State

Serverless functions are stateless, so an in-memory dict silently drops writes
between invocations. Set Upstash credentials in the Vercel project and the store
writes through to serverless Redis instead:

```
UPSTASH_REDIS_REST_URL=...
UPSTASH_REDIS_REST_TOKEN=...
```

Without them the store falls back to memory — correct locally and under a single
long-lived container, wrong on Vercel. `/api/health` tells you which is active,
so a misconfigured deploy is visible rather than subtly broken.

Fixtures self-seed on first request when the KV is empty.

## Deploy

```bash
cd services/order-api
vercel --prod
```
