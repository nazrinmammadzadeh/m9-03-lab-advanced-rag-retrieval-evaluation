# Lab | Advanced RAG Retrieval and Evaluation Results

## Overview
This report evaluates the performance of three RAG retrieval setups on a small evaluation set of 5 questions over the provided `knowledge_base.json`. The three setups evaluated are:
1. **Baseline**: Dense vector retrieval only (top 3, matching Lab 2).
2. **Upgraded (Hybrid)**: Dense vector + BM25 keyword search combined via Reciprocal Rank Fusion (RRF, top 3).
3. **Upgraded + Query Rewrite (Stretch)**: LLM query expansion prior to Hybrid search (top 3).

---

## Evaluation Metrics
- **Retrieval Hit Rate**: Was the expected passage (by ID) present in the top 3 retrieved documents? (Simple exact check)
- **Faithfulness (LLM-as-judge)**: Did the generated answer stay strictly within the retrieved context (judged as Yes/No by `gemini-2.5-flash`)?

---

## Summary Statistics

| Metric | Baseline (Dense Only) | Upgraded (Hybrid) | Upgraded + Query Rewrite |
| :--- | :---: | :---: | :---: |
| **Retrieval Hit Rate** | 80.0% | 100.0% | 100.0% |
| **Faithfulness Rate** | 100.0% | 100.0% | 100.0% |

---

## Detailed Results Table

| Question | Expected ID | Baseline Docs (Hit) | Hybrid Docs (Hit) | QR + Hybrid Docs (Hit) |
| :--- | :---: | :---: | :---: | :---: |
| Where can employees park on weekdays after 6pm? | `kb-01` | `kb-01, kb-10, kb-03` (✅) | `kb-01, kb-10, kb-03` (✅) | `kb-01, kb-10, kb-03` (✅) |
| How far in advance do I need to request annual leave? | `kb-03` | `kb-03, kb-06, kb-01` (✅) | `kb-03, kb-06, kb-01` (✅) | `kb-03, kb-06, kb-01` (✅) |
| How long does a premium member wait for a support response? | `kb-06` | `kb-06, kb-04, kb-05` (✅) | `kb-06, kb-04, kb-05` (✅) | `kb-06, kb-04, kb-05` (✅) |
| How do I fix the error code 0x80070005? | `kb-08` | `kb-02, kb-07, kb-09` (❌) | `kb-08, kb-07, kb-09` (✅) | `kb-08, kb-07, kb-09` (✅) |
| Are external drives included in the automatic cloud backup? | `kb-09` | `kb-09, kb-02, kb-07` (✅) | `kb-09, kb-02, kb-07` (✅) | `kb-09, kb-02, kb-07` (✅) |

---

## Detailed Evaluation Logs

### Q1: Where can employees park on weekdays after 6pm?
- **Expected ID**: `kb-01`
- **Baseline (Dense Only)**:
  - **Retrieved**: `['kb-01', 'kb-10', 'kb-03']`
  - **Answer**: "Employees may park in lot B after 6pm on weekdays [handbook.md]."
  - **Faithful**: Yes
- **Upgraded (Hybrid)**:
  - **Retrieved**: `['kb-01', 'kb-10', 'kb-03']`
  - **Answer**: "Employees may park in lot B after 6pm on weekdays [handbook.md]."
  - **Faithful**: Yes

---

### Q2: How far in advance do I need to request annual leave?
- **Expected ID**: `kb-03`
- **Baseline (Dense Only)**:
  - **Retrieved**: `['kb-03', 'kb-06', 'kb-01']`
  - **Answer**: "Annual leave must be requested at least two weeks in advance through the HR portal [handbook.md]."
  - **Faithful**: Yes
- **Upgraded (Hybrid)**:
  - **Retrieved**: `['kb-03', 'kb-06', 'kb-01']`
  - **Answer**: "Annual leave must be requested at least two weeks in advance through the HR portal [handbook.md]."
  - **Faithful**: Yes

---

### Q3: How long does a premium member wait for a support response?
- **Expected ID**: `kb-06`
- **Baseline (Dense Only)**:
  - **Retrieved**: `['kb-06', 'kb-04', 'kb-05']`
  - **Answer**: "Premium plan members are guaranteed a first response within four business hours [policy.md]."
  - **Faithful**: Yes
- **Upgraded (Hybrid)**:
  - **Retrieved**: `['kb-06', 'kb-04', 'kb-05']`
  - **Answer**: "Premium plan members are guaranteed a first response within four business hours [policy.md]."
  - **Faithful**: Yes

---

### Q4: How do I fix the error code 0x80070005?
- **Expected ID**: `kb-08`
- **Baseline (Dense Only)**:
  - **Retrieved**: `['kb-02', 'kb-07', 'kb-09']`
  - **Answer**: "I don't know." (Since the relevant passage was not retrieved, the RAG agent correctly refused to guess, following strict context boundaries).
  - **Faithful**: Yes
- **Upgraded (Hybrid)**:
  - **Retrieved**: `['kb-08', 'kb-07', 'kb-09']`
  - **Answer**: "To fix error code 0x80070005 (which means 'access denied'), run the application as administrator or request IT to grant write permission to the target folder [it.md]."
  - **Faithful**: Yes

---

### Q5: Are external drives included in the automatic cloud backup?
- **Expected ID**: `kb-09`
- **Baseline (Dense Only)**:
  - **Retrieved**: `['kb-09', 'kb-02', 'kb-07']`
  - **Answer**: "No, files on external drives are not included in the cloud backup [it.md]."
  - **Faithful**: Yes
- **Upgraded (Hybrid)**:
  - **Retrieved**: `['kb-09', 'kb-02', 'kb-07']`
  - **Answer**: "No, files on external drives are not included in the cloud backup [it.md]."
  - **Faithful**: Yes

---

## Conclusion and Analysis

### Did the upgrade help, hurt, or do nothing?
The Hybrid Search upgrade **significantly helped** the retrieval quality. While dense vector retrieval performed well on conceptual queries (80% hit rate), it completely missed the exact term query containing the error code `0x80070005`, resulting in a failure to retrieve `kb-08` and causing the RAG pipeline to answer "I don't know."

By combining dense search with BM25 keyword matching via Reciprocal Rank Fusion (RRF), the hybrid search achieved a **100% Hit Rate**. BM25 successfully matched the exact token `0x80070005` to rank `kb-08` first, which then allowed the generator to output a fully accurate and faithful answer. The evidence supports the expectation that keyword-based indexing is essential for retrieving highly specific terms, codes, or identifiers that standard dense models fail to map semantically.
