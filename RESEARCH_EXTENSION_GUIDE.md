# CI/CD diagnostic research extension

This is the **original PatchRail 0.7.6 project plus additive research code**, not a replacement. All original source files, public sample logs, fixtures, tests, documentation and Apache-2.0 licence are preserved.

## Fast start on Windows (VS Code terminal)

1. Extract the ZIP and open the project directory in VS Code.
2. Run `research_extension\run_offline.bat` in a terminal, or execute the equivalent commands from `research_extension/README.md`.
3. Inspect `results/research_extension/summary.json`, `clusters.csv`, `real_world_22_baselines.csv`, the complete per-case `case_details.json`, and the four figures.
4. For actual generative LLM inference, separately install [Ollama](https://ollama.com/), run `ollama pull qwen2.5:0.5b`, and launch `research_extension\run_agent_local.bat` while the Ollama service is running.

The offline model was fitted and evaluated successfully, and 198 focused automated tests passed (plus 695 subtests). **LLM inference was not performed in the archive-creation environment**, which had no model weights or model service. The agent code is included and its transport was unit-tested with a mock response; this is not evidence of actual LLM diagnosis.

See `research_extension/EXECUTION_REPORT.md` for exact outcomes, datasets, weaknesses, fixed denominators and reproducibility details. No fabricated model outputs or undocumented external-repository fixes are included.
