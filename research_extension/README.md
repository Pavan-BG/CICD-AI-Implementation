# PatchRail research extension

Original PatchRail files and 223 labelled CI fixtures are preserved. This extension adds TF-IDF + KMeans clustering, nearest-neighbour retrieval, and an optional real pretrained LLM diagnosis/verification stage.

## Run offline experiment

From the extracted project root, with Python 3.11+:

```powershell
python -m pip install -r research_extension/requirements.txt
python -m research_extension.run --backend none
python -m pytest -q research_extension/tests
```

Results are created in `results/research_extension/`, including `clusters.csv`, `cluster_report.json`, `silhouette_scores.json`, `reference_audit_50.csv`, `real_world_22_baselines.csv`, `case_details.json`, `summary.json` and three PNG charts. Source data are the project's `examples/ci-triage` and 22 explicitly enumerated logs in `research_extension/real_world_reference.json`; no raw log is copied or changed. The 223 fixture records and the 22 real-world records never overlap.

The 50-row audit compares cluster membership against *existing maintained fixture expected labels*, mapped into broader categories; this is not independently hand-annotated ground truth. A human should inspect and correct those 50 category decisions before claiming a human label agreement rate. Low cosine silhouette scores and the heterogeneous cluster are substantive limitations, not hidden.

## Run with a pretrained LLM

Use a local Ollama installation (download and run models on your own machine):

```powershell
ollama pull qwen2.5:0.5b
ollama serve
python -m research_extension.run --backend ollama --model qwen2.5:0.5b
```

Alternatively, set `OPENAI_API_KEY` and use:

```powershell
python -m research_extension.run --backend openai --model gpt-4.1-mini
```

`--suggest` adds a recommendation for each cluster. Neither the Ollama weights nor API credentials are included. Calls to external APIs can incur charges. Logs sent to APIs may contain sensitive repository information; audit/redact before external transmission. Local Ollama is preferable for private logs. The model outputs must be checked for accuracy and safety. The extension is **read-only** and does not deploy, repair, or approve software.

If model loading or a call fails, `summary.json` reports `llm_status=incomplete` and the command returns a non-zero exit code. Never treat offline TF-IDF votes as measured LLM output. `verified` means the model category matched the majority category of the three retrieved logs; it does not prove the root cause is correct. Diagnostic correctness is additionally compared against the provisional broad categories in `real_world_reference.json`, mapped from the PatchRail benchmark documentation. For independent accuracy claims, obtain expert labels and a fresh held-out dataset.

## Method details

Cleaning removes timestamps, hashes, URLs, file paths and numeric tokens. TF-IDF uses English stop words, unigram/bigram features and `min_df=2`. KMeans tries k=4 through k=8 with random seed 42 and 20 initialisations. Best k is selected by cosine silhouette. For each cluster the script emits size, eight top terms from its centre and five example filenames. New logs retrieve three fixture logs by TF-IDF cosine similarity without including the queried real-world log in the index.

Cluster names are derived deterministically from dominant fixture labels. Some clusters merge unrelated failure classes; those are explicitly named "Mixed" and must not be presented as single coherent root causes. `tfidf_vote_broad` is a separate, non-LLM nearest-neighbour baseline using the three known fixture failure labels. All 22 held-out real-world cases are evaluated using the *same* source-derived category definitions.

## Source provenance

Original PatchRail is authored by its contributors and distributed under Apache-2.0 (`LICENSE`). See `docs/real-world-benchmark.md` and `examples/real-world/README.md` for the original 22 failure narratives and source links. The categorisation in `real_world_reference.json` is a researcher mapping of these published narratives; unknown cases are preserved rather than assigned invented causes. The source benchmark describes PatchRail code fixes, **not** a verified record of the fixes made by each external repository. Consequently, automatic comparison to the external repository's actual fixes is deliberately not claimed.

For Windows, `research_extension/run_offline.bat` installs extension dependencies and runs the full offline pipeline and new unit tests. After installing Ollama and pulling the model, `research_extension/run_agent_local.bat` runs the genuine small-model agent and optional suggestions. See `research_extension/EXECUTION_REPORT.md` for the precisely recorded findings and limitations.
