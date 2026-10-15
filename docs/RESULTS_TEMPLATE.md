# Results write-up (fill in from YOUR real run; do not publish numbers you did not produce)

**Dataset:** PubMedQA PQA-L sample, n = ___, seed = ___, label mix = yes ___ / no ___ / maybe ___, version ___, sha256 ___
**Model:** ___ (temperature 0)   **Date:** ___   **Estimated cost of the run:** ___

## 1. Accuracy (deterministic)
| Prompt | correct | valid_json | rationale_grounded | no_personal_advice | fully passing |
|---|---|---|---|---|---|
| v1 | ___ | ___ | ___ | ___ | ___ |
| v2 | ___ | ___ | ___ | ___ | ___ |

Accuracy by gold label (yes / no / maybe): ___ / ___ / ___   (note: "maybe" is rare, so read it with caution)

## 2. Judge calibration (30 outputs scored by hand)
exact agreement ___ | within 1 point ___ | judge minus human mean ___ | kappa (pass at 4) ___
What the judge got wrong and why: ___
Decision: trust / do not trust the judge for ___ because ___

## 3. What the regression gate caught
Prompt change: ___ -> cases that flipped: ___ -> root cause found in the trace at step: ___

## 4. Failures I looked at (3 examples)
1. case ___: model said ___, gold ___, why: ___
2. ...

## 5. Limits
Sample size ___; one model; judge agreement measured on only ___ items; accuracy is not clinical safety.
