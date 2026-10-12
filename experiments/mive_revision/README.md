# MIVE verified development revision

This revision retains the negative legacy allocation result and adds an independent continuation audit. It does not establish an efficiency gain for a newly trained MIVE controller or complete the requested scaled held-out study.

All commands run from the repository root. Install the repository requirements first. The original generator and encoder assets, upstream question exports, and local llama.cpp runtime are required for new inference and raw answer rescoring.

## Numerical reconstruction

```sh
python experiments/mive_revision/verify_statistics.py
```

The public archive contains sanitized branch outcomes, checkpoint identities, scalar value estimates, K stability results, and bootstrap statistics. This command reconstructs those numerical results, including Kendall tau-b, without benchmark text. Complete answer rescoring and state restoration require the retained local raw records and `verify_audit.py`.

## New development inference

```sh
python experiments/mive_revision/audit.py prepare
python experiments/mive_revision/audit.py run
```

`freeze.json` identifies the selected questions, model and controller hashes, checkpoints, budgets, seeds, K candidates, and gate thresholds. The resumed run uses `parallel_audit.py` with the separately recorded four-slot runtime amendment. Missing journal keys are appended; existing rows are preserved. A new scientific study must use a separate output directory and freeze, rather than overwrite the retained run.

```sh
python experiments/mive_revision/audit.py analyze
python experiments/mive_revision/correct_pairing.py
python experiments/mive_revision/analyze_utility.py
python experiments/mive_revision/verify_audit.py
python experiments/mive_revision/verify_statistics.py
python experiments/mive_revision/cost_analysis.py
```

The utility objective is a declared development sensitivity analysis, not part of the initial freeze. Its exact integer implementation avoids arithmetic-induced rank changes. Failed stability gates remain failed; the scripts do not authorize held-out tuning or inference.

## Additional backbone checks

```sh
python experiments/mive_revision/download_models.py
python experiments/mive_revision/backbone_smoke.py Qwen2.5-1.5B prepare
python experiments/mive_revision/backbone_smoke.py Qwen2.5-1.5B run
```

The 1.5B check uses a CPU-only server on port 8097. The corresponding 7B command uses a GPU server on port 8098. Both require complete official weight checksum verification before inference. These checks use three development questions, two fixed policies, and seed 17. They are execution checks, not a powered model comparison. Their frozen configurations and results identify what actually completed.

## Manuscript and archive

```sh
python experiments/mive_revision/figures.py
python experiments/mive_revision/package_revision.py --register
python experiments/mive_revision/build_manuscript.py
python experiments/mive_revision/package_revision.py
```

The Word builder requires the supplied revision as its equation source; set `SOURCE` to the local lawful copy before rebuilding. Render and inspect every page after editing. The archive excludes benchmark text, raw answer-bearing states, model weights, environments, binary runtimes, and credentials. Its checksums support numerical reconstruction; they do not replace the missing scaled experiments. No Zenodo DOI or software license grant is invented.
