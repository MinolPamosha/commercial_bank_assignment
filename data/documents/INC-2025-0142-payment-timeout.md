---
doc_id: INC-2025-0142
title: Payment Gateway Timeout — UPI Settlement Lag
department: payments
document_type: incident
access_level: confidential
created_date: 2025-03-12
tags: [upi, timeout, settlement, outage]
---

# Incident INC-2025-0142 — Payment Gateway Timeout

**Severity:** SEV-2  
**Service:** ApexPay UPI Gateway  
**Duration:** 2025-03-12 14:22 IST → 16:05 IST  
**Impact:** ~8% of UPI push transactions timed out after 12s; card payments unaffected.

## Timeline
- 14:22 — Alert: p95 latency > 8s on `/upi/collect`
- 14:31 — On-call confirmed elevated timeouts in settlement worker pool
- 14:55 — Scaled settlement workers 8 → 24; partial recovery
- 15:40 — Identified saturated Redis connection pool to ledger cache
- 16:05 — Pool size raised; circuit breaker reset; SEV closed

## Root Cause
Settlement workers exhausted Redis connections after a config change lowered `max_connections` from 200 to 50 during a cost-optimization PR. Under peak festival traffic, workers blocked waiting for connections, causing upstream gateway timeouts.

## Recurring Theme
Connection-pool misconfiguration under load has appeared in INC-2024-0891 and INC-2025-0033.

## Remediation
1. Revert Redis pool to 200 with alert at 70% utilization.
2. Add canary check for pool saturation in release gates.
3. Update runbook RB-PAY-017 with pool diagnostics.
