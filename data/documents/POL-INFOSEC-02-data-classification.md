---
doc_id: POL-INFOSEC-02
title: Policy — Information Classification & Access
department: infosec
document_type: policy
access_level: internal
created_date: 2024-08-01
tags: [policy, access, classification]
---

# POL-INFOSEC-02 — Data Classification

## Levels
- **Public** — marketing materials
- **Internal** — runbooks, architecture overviews
- **Confidential** — incident RCA with customer impact figures
- **Restricted** — cryptographic keys, customer PII extracts (never in RAG corpus)

## Assistant Rules
Knowledge assistants MUST filter by employee clearance. Viewers see Internal and below. Analysts may access Confidential. Restricted content is out of scope for this assistant.
