# Revision and reviewer concern map

This is a verified methods and diagnostic revision, not a completed scaled Q1 efficacy study. Missing recovery results are excluded. Raw retained scores remain unchanged; decimal formatting affects display only.

| Change | Reviewer concern | Revision | Evidence |
|---|---|---|---|
| Evidence provenance | Missing recovery package supports uncheckable results | Exclude every 1.5B gate/threshold and compact-refresh result; retain an unsupported-claim register | Verified manuscript Sections 5, 8 and Appendix B |
| Independent continuation measurement | Legacy prediction residuals are not continuation mismatch | Restore checkpoints and generate disjoint A/B outcomes under mu0 and legacy CRAVE | dev4b/branches.jsonl and summary.json |
| Variance gate | All actions may have identical reward | Record the exact zero-variance fraction and frozen checkpoint fallback outcome | dev4b/variance_gate.json |
| K stability | Noisy action ranks can look like policy shift | Evaluate K=3 and K=5 independently; retain failed thresholds without certification | dev4b/stability.json |
| Randomness audit | Matched seeds do not necessarily reduce variance | Compare seed-tape and independent contrasts at equal branch budgets | dev4b/checkpoint_metrics.json and utility_checkpoint_metrics.json |
| Estimand correction | Raw EM STOP is an endpoint and cannot strictly reverse sign | Separate primary EM from a declared bounded resource-priced sensitivity | dev4b/secondary_analysis_declaration.json |
| Theory | No finite-sample condition for reliable THINK STOP ranking | Add Theorem 3 with boundedness, independence, fixed-state assumptions and proof | Verified manuscript Section 4.4 |
| Negative findings | Accuracy-only reporting obscures additional computation | Retain all four policies, F1, output/input tokens, retrievals, verifications and latency | archival_reanalysis.json |
| Figures | Low-information cluster charts | Add continuation heatmaps, uncertainty plots, noise decomposition and coupling scatter | figures/ |
| Related methods | Simplified proxies could be mislabeled reproductions | Compare estimand, continuation and full cost conditions for closest papers | Verified manuscript Table 1 |
| Structure and precision | Repetition and protocol status displace results | Use requested nine-section structure; move status and archival proxy to appendices; preserve raw full precision | Verified manuscript |
| References | Unverified identifiers or mixed bibliographic formatting | Retain primary metadata and DOI registry responses; flag unresolved pages | artifacts/references/ |
| Completion scope | Scaled model and benchmark claims lack completed runs | Separate prospective intent from completed results; no fake final freeze or power calculation | prospective_scaled_plan.json |
| Declarations and archive | Unconfirmed declarations and absent permanent DOI | Add AI and ethics statements; leave author facts and Zenodo DOI pending | Verified manuscript Declarations |

## Completed development measurement

1080 branches across 9 checkpoints and 9 questions. Artifact: `D:\Study\Papers\p6\experiments\mive_revision\artifacts\dev4b\summary.json`.
Reward-variance gate: PASS; identical action means at 7/9 checkpoints. K selected by the complete A/B rule: None. If none passes, reported K=5 remains exploratory.
The theory now separates deterministic EM stopping from priced-value rank reversals and adds a bounded-return ranking condition. No continuation mechanism is asserted as the cause of the historical allocation failure.

## Unsupported claims and incomplete experiments

- **1.5B fixed-schedule baseline EM 0.25 / 0.275 and selected confidence threshold 0.5** — User confirms the recovery package exists nowhere; no raw outcomes or freeze retained. Honest replacement: No verified result for that run; exclude its numerical claims.
- **Six-episode CPU check with 859 generated tokens and all incorrect answers** — Only source manuscript assertions, no execution artifacts. Honest replacement: Do not report as a completed experiment.
- **Compact continuation-refresh training completed with 295 generation calls** — Missing branch, generation and training ledgers. Honest replacement: Continuation refresh is proposed, with no verified held-out benefit.
- **MIVE improves accuracy or efficiency versus legacy/confidence** — No new trained MIVE policy with a frozen held-out comparison. Honest replacement: A policy-conditioned estimand and a development audit are implemented.
- **7B or 1.5B held-out efficacy or model-size transfer** — No sufficiently powered frozen held-out comparison for the additional backbones. Honest replacement: Retain completed frozen development execution checks separately, without model ranking or transfer claims.
- **Three datasets with 300 held-out questions, three seeds and three caps** — No completed official-release manifests, training/dev finalization, or full inference grid. Honest replacement: Legacy 40-question one-cap evaluation and fresh small development audit.
- **Faithful Adaptive-RAG, Self-RAG or FLARE reproduction** — No native classifier/reflection/draft pipeline reproduced with charged costs. Honest replacement: Conceptual comparison with explicit reproduction conditions.
- **Hard mask, semantic penalty, direct/composite, refresh and label-coupling ablation benefits** — Full requested matched-budget ablation grid not completed. Honest replacement: Only measured coupling behavior is reported; no component-level causal benefit.
- **Noninferiority, savings, dev-powered sample size or a positive MIVE break-even count** — Missing paired MIVE dev and held-out outcomes; legacy token savings are negative. Honest replacement: No NI claim; no positive finite legacy token break-even count.
- **Official answer evaluator equivalence and full-Wikipedia RAG generalization** — Bounded remapped exports and supplied-context retrieval; alias equivalence not verified. Honest replacement: Normalized answer EM/F1 in the declared supplied-context environment.
- **Q1 acceptance readiness or reviewer-proof empirical conclusions** — Scale, baselines, ablations and permanent archive remain incomplete. Honest replacement: Evidence-led methodological draft; submission needs remaining experiments and author confirmation.
- **A deposited Zenodo DOI** — No Zenodo deposit exists. Honest replacement: Prepared archive only; insert a DOI after actual deposition.

## Remaining scaled study

The requested primary comparison requires 32,400 episodes before published baselines or ablations. The prospective plan is not a final freeze and permits no held-out inference. Complete official dataset/alias manifests, new MIVE training, development quality and K gates, paired development power, and final model-specific settings first. Additional backbone execution checks, when completed, are reported separately with their own frozen development configurations. No baseline proxy is presented as a published-method reproduction.

Zenodo is not connected and no deposit DOI exists. Funding and competing interests require explicit author confirmation. The archive contains sanitized numerical outcomes and code; raw benchmark text, model weights, binary runtimes and credentials are excluded.