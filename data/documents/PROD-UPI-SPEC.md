---
doc_id: PROD-UPI-SPEC
title: Product Spec — Apex UPI Collect
department: product
document_type: product_spec
access_level: internal
created_date: 2025-01-05
tags: [product, upi]
---

# Product Spec — Apex UPI Collect

Merchants initiate collect requests via ApexPay. Timeouts surface as `GATEWAY_TIMEOUT` after 12 seconds. Retry guidance: exponential backoff, honor idempotency key for 24h. Festival mode raises rate limits for verified merchants.
