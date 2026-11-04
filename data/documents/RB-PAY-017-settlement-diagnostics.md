---
doc_id: RB-PAY-017
title: Runbook — Settlement Worker & Redis Pool Diagnostics
department: payments
document_type: runbook
access_level: internal
created_date: 2025-03-15
tags: [runbook, settlement, redis]
---

# RB-PAY-017 — Settlement Diagnostics

## When to use
Gateway timeouts, elevated `/upi/collect` latency, settlement lag alerts.

## Steps
1. Check Grafana dashboard `payments-settlement` — worker queue depth.
2. Redis: `INFO clients` — compare connected_clients vs maxclients.
3. If pool > 70%: scale workers cautiously; prefer raising max_connections first.
4. Verify circuit breaker state on ApexPay gateway.
5. Page Payments On-call if SEV-2+ and not recovered in 15 minutes.

## Do NOT
- Flush Redis ledger cache during business hours without Change Ticket.
- Disable idempotency checks.
