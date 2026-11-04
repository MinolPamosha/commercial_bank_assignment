---
doc_id: INC-2025-0033
title: NEFT Duplicate Credits during Failover
department: payments
document_type: incident
access_level: confidential
created_date: 2025-01-18
tags: [neft, duplicate, failover, outage]
---

# Incident INC-2025-0033 — NEFT Duplicate Credits

**Severity:** SEV-1  
**Service:** NEFT Clearing Adapter  
**Duration:** 2025-01-18 09:10 IST → 11:45 IST  
**Impact:** 214 customers received duplicate credits totaling ₹18.4L; reversed within T+1.

## Root Cause
Active-passive failover replayed in-flight NEFT ack messages because idempotency keys were stored only in the primary region's Redis without cross-region replication.

## Recurring Theme
Idempotency gaps during regional failover (also noted in INC-2024-1102 card auth retries).

## Actions
- Replicate idempotency keys to both regions with 72h TTL.
- Add dual-write verification job.
- Customer remediation playbook executed under Policy POL-OPS-04.
