# CRAVE submission readiness

## Scientific position

The manuscript is a methods and diagnostic study with a negative allocation finding. It is not a completed efficiency-superiority benchmark. Editing improves clarity and reproducibility; it cannot establish Q1 acceptance or replace missing evidence. No particular journal or ranking category has been selected.

## Verified evidence

- 642 retained evaluation records, including 162 pilot and 480 expanded episodes.
- 296 summary metrics with bootstrap intervals and 46 paired comparisons with raw and Holm-adjusted p values.
- 192 training and oracle branch payloads, scores, and matched seeds.
- 1080 reconstructed value and success predictions; maximum absolute Q discrepancy 2.384e-7 under a declared 2e-6 float32 tolerance.
- 90 diagnostic cells, 22 retrieval-support records, and zero recorded hard-budget violations.
- Seventeen core implementation tests passed.

These checks reconstruct retained outputs. They do not regenerate stochastic model responses, certify official evaluator equivalence, establish a representative full-benchmark sample, or independently validate physical invocation timing.

## Evidence needed for a broader journal claim

1. Acquire complete official, checksummed data releases and confirm official evaluator and alias equivalence. Build a frozen retrieval corpus suited to the claimed deployment rather than pooling evaluation contexts.
2. Expand controller training and reserve disjoint development tasks for selecting K, thresholds, loss weights, and architecture. Freeze these choices before the next test.
3. Compare matched and observational supervision at equal action coverage, state coverage, and label quantity. Remove duplicate myopic labels from confirmatory families. Measure ordered-interaction coverage before testing interaction benefits.
4. Implement the closest literature baselines faithfully, especially CoBRA, inference-time budget control, adaptive retrieval, and stopping methods. Match the backbone, executor, action caps, available evidence, and resource accounting.
5. Prespecify a primary paired estimand and meaningful accuracy margin. Determine the test-task count from development-estimated paired variance and the desired confidence width or detectable effect; do not count seeds as independent questions or claim retrospective observed power.
6. Evaluate multiple budgets, at least one additional backbone, latency under an explicit cold-run protocol, and input-token-inclusive costs. Measure labeling cost and the deployment volume needed to amortize it.
7. Retain negative outcomes and separate newly selected tests from the inspected post-pilot evaluation. New results require new frozen identities and an updated manuscript.

## Author and submission information

Supply human author names, affiliations, corresponding contact, contributions, funding, competing-interest declarations, and final human approval. No absence-of-funding or conflict statement has been inferred. Select a journal before adapting word limits, highlights, reference style, and supplementary files. Add an archival release identifier if appropriate, and choose a software license explicitly.

OpenAI Codex assisted code and manuscript authoring. The manuscript reports its role and the absence of pinned authoring-version identifiers. Human authors must verify and finalize the AI disclosure and all references. Data visualizations were computed by reproducible Matplotlib scripts from saved observations. No generated research image or invented result is included.

Publisher policies should be checked at submission time. [Elsevier's AI policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals) requires disclosure of substantive AI assistance and author oversight. [Research data statements](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-statement) describe available data and access limitations. These are publisher-wide guidance, not confirmation that any particular journal or Q1 category is suitable.
