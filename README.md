# AI-Assisted CI/CD Failure Triage

**Master's research implementation | PatchRail + TF-IDF + MiniLM + evidence-aware verification**

This repository contains a reproducible research prototype that investigates how AI-assisted log analysis can support failure diagnosis in Continuous Integration and Continuous Delivery (CI/CD) workflows. The implementation extends the existing open-source **PatchRail** CI failure classifier with semantic retrieval and an evidence-checking stage. It is a research extension, **not** a new CI/CD platform.

## Project objectives

- Run the original PatchRail rule-based failure diagnosis as the baseline.
- Retrieve similar incidents using **TF-IDF** or pretrained **MiniLM** embeddings when the baseline cannot give a named diagnosis.
- Check log evidence before presenting an AI-generated suggestion; reject unsupported specific diagnoses or provide a broader investigation hint.
- Compare baseline and extension behaviour on a small, exploratory set of CI failure logs.
- Demonstrate reproducibility with Python tests, evaluation scripts, saved evidence, and an optional GitHub Actions workflow.

**Safety boundary:** AI outputs are advisory and require human verification. The extension does not change the original CI job exit status or approve deployments.

## Technology

Python 3.11+, PatchRail v0.7.6, scikit-learn/TF-IDF, Sentence Transformers (`all-MiniLM-L6-v2`), pytest, and GitHub Actions.

## Repository contents

The project is currently provided as **`CICD implementation.zip`**. After extracting the archive, the main components are:

```text
S495 implementation/
├── .github/workflows/research-ci-triage.yml  # Optional GitHub Actions demo
├── implementation/PatchRail_AI_V3/           # Research extension and tests
├── upstream/patchrail-main/                  # Original PatchRail source
├── evidence/                                 # Local results, figures and audits
├── START_HERE.md                             # Detailed running instructions
├── RESULTS_INTERPRETATION.md                 # Results and limitations
├── ATTRIBUTION_AND_LICENSE.md                # Source attribution
├── RUN_WINDOWS_SETUP.cmd
├── RUN_WINDOWS_OFFLINE.cmd
└── RUN_WINDOWS_FULL.cmd
```

> **GitHub Actions note:** GitHub cannot run a workflow stored *inside* a ZIP. To enable the workflow, upload the **contents** of the extracted `S495 implementation` folder to the repository root, retaining the `.github/workflows/` path. Uploading only the ZIP is sufficient for sharing the archive, but not for executing the CI workflow on GitHub.

## Run locally (Windows 10/11)

1. Download `CICD implementation.zip` and extract it.
2. Open **Command Prompt** in the extracted `S495 implementation` folder.
3. Install the dependencies and run the full comparison:

```cmd
RUN_WINDOWS_SETUP.cmd
RUN_WINDOWS_FULL.cmd
```

`RUN_WINDOWS_FULL.cmd` runs the PatchRail benchmark, the V3 unit tests, the TF-IDF/MiniLM comparisons, and a sample single-log diagnosis. Outputs are written to `RUN_OUTPUT/`. The MiniLM model weights are **not included** in the ZIP and are downloaded on first use.

For the **TF-IDF-only** route, which does not require MiniLM model weights:

```cmd
python -m pip install -r implementation\PatchRail_AI_V3\requirements-v2.txt
RUN_WINDOWS_OFFLINE.cmd
```

The offline results are saved in `RUN_OUTPUT_OFFLINE/`. This route does **not** evaluate MiniLM.

## GitHub Actions demonstration

After uploading the **extracted** project files to the repository root:

1. Open the repository's **Actions** tab.
2. Choose **Research prototype CI failure triage demonstration**.
3. Select **Run workflow** and choose `tfidf` (recommended first) or `minilm`.
4. Review the job logs and download the `research-ci-failure-diagnosis` artifact.

This is a **manual** workflow (`workflow_dispatch`). The supplied package contains its configuration, but a successful hosted run must be verified on GitHub before being reported as completed.

## Evaluation evidence

The package documents the following local checks:

| Check | Documented outcome |
| --- | --- |
| Original PatchRail curated-fixture benchmark | 223/223 |
| Version 3 Python unit tests | 16 passed |
| Exploratory public-log evaluation | 22 logs |

For the **locally reproduced TF-IDF comparison** on the 22 previously examined logs (using provisional researcher-assigned labels):

| Method | Broad-label correct | Named suggestions | Incorrect named suggestions |
| --- | ---: | ---: | ---: |
| PatchRail baseline | 9/22 | 12/22 | 4 |
| TF-IDF V2 | 9/22 | 13/22 | 5 |
| TF-IDF V3 | 9/22 | 12/22 | 4 |

The package also includes **researcher-provided Windows MiniLM results**, preserved separately from locally reproduced outputs. Those observations are not independent external validation.

**Interpretation:** These exploratory, post-hoc results demonstrate an implemented evidence-checking approach; they do **not** establish generalisable accuracy gains, faster software delivery, improved DORA metrics, or proven root-cause correctness. Independent labels and untouched evaluation logs are required for stronger claims.

Relevant files: `evidence/reproduced_here/`, `evidence/user_windows_v3_results/`, `RESULTS_INTERPRETATION.md`, and `SUBMISSION_CHECKLIST.md`.

## Attribution and licensing

- **PatchRail** is a third-party open-source project: https://github.com/patchrail/patchrail. Its original licence (Apache 2.0) is retained under `upstream/patchrail-main/LICENSE`.
- The AI failure-triage implementation is a separate academic research extension of PatchRail.
- MiniLM is an externally provided pretrained model; its weights are not included in this repository archive.

See `ATTRIBUTION_AND_LICENSE.md` for provenance, authorship, and reuse information.

