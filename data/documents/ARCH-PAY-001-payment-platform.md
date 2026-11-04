---
doc_id: ARCH-PAY-001
title: Architecture — Apex Payment Platform
department: payments
document_type: architecture
access_level: internal
created_date: 2025-02-01
tags: [architecture, payments, upi, neft, imps]
---

# Apex Payment Platform Architecture

## Components
- **API Edge:** Kong → ApexPay Gateway
- **Rails:** UPI, IMPS, NEFT/RTGS adapters
- **Ledger Cache:** Redis (active-active planned)
- **Settlement Workers:** Kubernetes HPA on queue depth
- **Idempotency Store:** Redis with cross-region replication (post INC-2025-0033)

## SLOs
- UPI success rate ≥ 99.5%
- p95 auth latency ≤ 2.5s
- Settlement lag ≤ 5 minutes at p99

## Failure Domains
Regional failover, partner network, Redis pool, worker saturation.
