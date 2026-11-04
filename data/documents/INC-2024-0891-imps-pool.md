---
doc_id: INC-2024-0891
title: IMPS Connection Pool Exhaustion
department: payments
document_type: incident
access_level: internal
created_date: 2024-11-02
tags: [imps, pool, timeout, outage]
---

# Incident INC-2024-0891 — IMPS Connection Pool Exhaustion

**Severity:** SEV-2  
**Service:** IMPS Switch Connector  
**Root Cause:** HikariCP pool sized for weekday traffic; Diwali campaign traffic spiked 3.2x. Threads blocked; customers saw "temporarily unavailable".

## Lessons
- Festival capacity planning must include connector pools, not only API pods.
- Same class of failure as later Redis pool incident INC-2025-0142.
