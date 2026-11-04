---
doc_id: BOARD-2025-04-REG-FINDINGS
title: Board Risk Committee Paper — Regulator Supervisory Findings on Payments Incidents
department: risk
document_type: meeting_notes
access_level: restricted
created_date: 2025-04-08
tags: [board, regulator, supervisory, penalty, payments, restricted]
---

# Board Risk Committee — Supervisory Findings (RESTRICTED)

**Classification:** Restricted — Board and C-suite only. Not for analyst or departmental circulation.  
**Prepared by:** Chief Risk Officer  
**Date:** 2025-04-08

## Background
Following incidents INC-2025-0033 (duplicate NEFT credits) and INC-2025-0142 (UPI settlement
timeouts), the regulator's supervisory team conducted a thematic review of Apex Commercial Bank's
payments resilience between 2025-03-18 and 2025-04-02.

## Supervisory findings
1. **Finding SF-01 (High):** Idempotency controls for cross-region failover were not tested before
   production rollout; the bank relied on a single-region Redis store for duplicate detection.
2. **Finding SF-02 (Medium):** Connection-pool capacity changes bypassed the release gate; the same
   failure mode recurred across three incidents.
3. **Finding SF-03 (Low):** Customer remediation under POL-OPS-04 was completed, but regulatory
   notification exceeded the 6-hour window by 2 hours 40 minutes for INC-2025-0033.

## Provisional supervisory action
- The regulator proposed a monetary penalty of **INR 4.2 crore** (four point two crore), subject to the
  bank's written response due 2025-05-15.
- Direction to appoint an independent reviewer for the payments platform by Q3 2025.

## Board decisions
- Approve an accounting provision of INR 4.5 crore in the Q1 FY26 accounts, covering the proposed
  INR 4.2 crore penalty plus an estimated INR 0.3 crore in legal and response costs.
- Mandate quarterly failover drills for all Tier-1 payment rails, reported to the Risk Committee.
- Approve the independent reviewer budget of INR 1.1 crore; shortlist to be tabled in May.
- Communications are embargoed until the regulator's final order; no internal announcement before then.
