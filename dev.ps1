<#
.SYNOPSIS
  Windows dev helper for the Voice Concierge monorepo.

.DESCRIPTION
  Wraps the PYTHONPATH wiring that the monorepo layout needs. On Windows the
  path separator is ';' not ':' — copying the Linux commands verbatim produces
  one malformed path and a confusing ModuleNotFoundError, so this script exists
  mainly to stop that happening.

.EXAMPLE
  .\dev.ps1 setup      # venv + dependencies
  .\dev.ps1 test       # pytest + ruff
  .\dev.ps1 gateway    # run gateway on :7860
  .\dev.ps1 orders     # run order API on :8000
  .\dev.ps1 split      # gateway talking to a running order API
  .\dev.ps1 demo       # exercise the constraint logic
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'test', 'gateway', 'orders', 'split', 'demo', 'clean')]
    [string]$Command = 'test'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot

# Semicolon separator — this is the Windows-specific bit.
$env:PYTHONPATH = @(
    (Join-Path $Root 'packages\order-contracts'),
    (Join-Path $Root 'services\order-api'),
    (Join-Path $Root 'services\gateway')
) -join ';'

function Assert-Venv {
    if (-not $env:VIRTUAL_ENV) {
        Write-Warning "No virtualenv active. Run: .\.venv\Scripts\Activate.ps1"
    }
}

switch ($Command) {
    'setup' {
        python -m venv .venv
        Write-Host "`nNow run:" -ForegroundColor Cyan
        Write-Host "  .\.venv\Scripts\Activate.ps1"
        Write-Host "  pip install -r requirements-dev.txt"
        Write-Host "  Copy-Item .env.example .env"
    }

    'test' {
        Assert-Venv
        pytest -q
        ruff check services packages tests scripts
    }

    'gateway' {
        Assert-Venv
        Write-Host "Gateway -> http://localhost:7860/docs" -ForegroundColor Green
        uvicorn app.main:app --reload --port 7860 --app-dir services/gateway
    }

    'orders' {
        Assert-Venv
        Write-Host "Order API -> http://localhost:8000/docs" -ForegroundColor Green
        uvicorn api.index:app --reload --port 8000 --app-dir services/order-api
    }

    'split' {
        Assert-Venv
        # Point the gateway at a separately running order API, mirroring the
        # deployed Railway -> Vercel hop.
        $env:ORDER_API_BASE_URL = 'http://127.0.0.1:8000'
        Write-Host "Gateway -> order API at $env:ORDER_API_BASE_URL" -ForegroundColor Green
        uvicorn app.main:app --reload --port 7860 --app-dir services/gateway
    }

    'demo' {
        Assert-Venv
        $d3 = (Get-Date).AddDays(3).ToString('yyyy-MM-dd')

        Write-Host "`n--- Open orders for caller 9990000001 ---" -ForegroundColor Cyan
        Invoke-RestMethod 'http://localhost:8000/api/orders?phone=9990000001' |
            Format-Table order_id, status, items_preview

        Write-Host "--- Reschedule into the saturated slot ($d3 morning) ---" -ForegroundColor Cyan
        $body = @{ new_date = $d3; window = 'morning' } | ConvertTo-Json
        $r = Invoke-RestMethod -Method Post `
            -Uri 'http://localhost:8000/api/orders/DLV1002/reschedule' `
            -Headers @{ 'Idempotency-Key' = 'demo-1' } `
            -ContentType 'application/json' -Body $body
        Write-Host "ok: $($r.ok)  refusal: $($r.refusal_code)"
        $r.alternatives | ForEach-Object { Write-Host "  offers: $($_.date) $($_.window)" }

        Write-Host "`n--- Already delivered (expect TERMINAL_STATE) ---" -ForegroundColor Cyan
        $r2 = Invoke-RestMethod -Method Post `
            -Uri 'http://localhost:8000/api/orders/DLV1005/reschedule' `
            -Headers @{ 'Idempotency-Key' = 'demo-2' } `
            -ContentType 'application/json' -Body $body
        Write-Host "refusal: $($r2.refusal_code) - $($r2.message)"
    }

    'clean' {
        Get-ChildItem -Path $Root -Include __pycache__, .pytest_cache, .ruff_cache `
            -Recurse -Directory | Remove-Item -Recurse -Force
        Write-Host "Cleaned." -ForegroundColor Green
    }
}
