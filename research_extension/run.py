from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .core import (ROOT, category_of_fixture, cluster_rows, major_cluster,
                   read_fixtures, read_real_world, save_model, train_cluster_model)
from .providers import diagnose, suggest_improvement

sys.path.insert(0, str(ROOT / "src"))
from patchrail.ci.classify import classify_ci_log, redact_ci_log


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def derive_names(model) -> dict[int, str]:
    names_file = Path(__file__).with_name("cluster_names_reviewed.json")
    assigned = json.loads(names_file.read_text(encoding="utf-8"))
    if model.kmeans.n_clusters != 8:
        raise ValueError("Cluster names must be manually reviewed again when selected k changes")
    return {i: assigned[str(i)] for i in range(8)}


def evaluate_reference_subset(model, names: dict[int,str], n: int = 50) -> list[dict]:
    # Deterministic stratified audit uses existing fixture classifications, not new human annotation.
    selected = []
    by_cluster = {i: [j for j, c in enumerate(model.labels()) if c == i]
                  for i in range(model.kmeans.n_clusters)}
    for c, indices in by_cluster.items():
        take = min(len(indices), n // model.kmeans.n_clusters)
        selected.extend(indices[:take])
    remaining = [i for i in range(len(model.fixtures)) if i not in set(selected)]
    selected += remaining[:n - len(selected)]
    results = []
    for i in selected:
        fixture = model.fixtures[i]
        actual = category_of_fixture(fixture.expected)
        cluster = int(model.labels()[i])
        in_cluster = [category_of_fixture(model.fixtures[j].expected)
                      for j, c in enumerate(model.labels()) if c == cluster]
        dominant, support = Counter(in_cluster).most_common(1)[0]
        results.append({"file": fixture.name, "fixture_expected": fixture.expected,
                        "reference_category": actual, "cluster": cluster,
                        "cluster_name": names[cluster],
                        "cluster_majority_category": dominant,
                        "majority_match": int(actual == dominant),
                        "majority_support": support})
    return results


def plot_outputs(out: Path, scores: dict[int,float], rows: list[dict], summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(list(scores), list(scores.values()), marker="o")
    ax.set(xlabel="Number of clusters (k)", ylabel="Cosine silhouette score",
           title="KMeans model selection: 223 fixture logs")
    ax.set_xticks(list(scores));ax.grid(axis="y", alpha=.25)
    fig.tight_layout();fig.savefig(out / "figure_4_32_silhouette.png", dpi=180);plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.barh([str(r["cluster"]) for r in rows], [r["size"] for r in rows])
    ax.invert_yaxis();ax.set(xlabel="Fixture logs", ylabel="Cluster ID",
           title="KMeans cluster sizes: 223 fixture logs")
    fig.tight_layout();fig.savefig(out / "figure_4_33_clusters.png",dpi=180);plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    labels = ["PatchRail baseline", "TF-IDF 3-neighbour vote"]
    heights = [summary["patchrail_correct_broad"], summary["retrieval_correct_broad"]]
    ax.bar(labels, heights)
    ax.set(ylabel="Correct broad-category labels",ylim=(0,summary["real_world_n"]),
           title=f"Offline comparison on the same {summary['real_world_n']} archived runs")
    for idx,v in enumerate(heights):ax.text(idx,v+.15,f"{v}/{summary['real_world_n']}",ha="center")
    fig.tight_layout();fig.savefig(out / "figure_4_34_offline_comparison.png",dpi=180);plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="PatchRail research extension: clustering and LLM diagnosis")
    parser.add_argument("--output",type=Path,default=ROOT / "results" / "research_extension")
    parser.add_argument("--backend",choices=["none","ollama","openai"],default="none")
    parser.add_argument("--model",default=None)
    parser.add_argument("--base-url",default=None)
    parser.add_argument("--suggest",action="store_true")
    args=parser.parse_args(argv)
    if args.backend == "none" and args.suggest:
        parser.error("--suggest requires a real LLM backend")
    model_name=args.model or ("qwen2.5:0.5b" if args.backend=="ollama" else "gpt-4.1-mini")
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    fixtures=read_fixtures();heldout=read_real_world()
    model=train_cluster_model(fixtures)
    names=derive_names(model)
    rows=cluster_rows(model,names)
    with (out / "cluster_report.json").open("w",encoding="utf-8") as f:
        json.dump(rows,f,indent=2)
    with (out / "silhouette_scores.json").open("w",encoding="utf-8") as f:
        json.dump(model.silhouettes,f,indent=2)
    write_csv(out / "clusters.csv",[{"cluster":r["cluster"],"name":r["name"],
               "size":r["size"],"top_8_terms":"; ".join(r["top_terms"]),
               "example_file":r["examples"][0]} for r in rows])
    save_model(model,out)
    audit=evaluate_reference_subset(model,names,50)
    write_csv(out / "reference_audit_50.csv",audit)
    cases=[]
    for log in heldout:
        start=time.perf_counter();baseline=classify_ci_log(log.text);time_baseline=time.perf_counter()-start
        refs=model.match(log.text,3)
        winner,votes=major_cluster(refs)
        source_category_votes=Counter(category_of_fixture(r["fixture_label"]) for r in refs)
        broad_category=source_category_votes.most_common(1)[0][0]
        baseline_raw=baseline["failure_class"]
        baseline_broad=category_of_fixture(baseline_raw)
        cases.append({"file":log.name,"reference_category":log.expected,
                      "patchrail_label":baseline_raw,"patchrail_broad":baseline_broad,
                      "patchrail_correct":int(baseline_broad==log.expected),
                      "patchrail_seconds":round(time_baseline,6),
                      "tfidf_vote_broad":broad_category,
                      "tfidf_vote_correct":int(broad_category==log.expected),
                      "retrieved_cluster_majority":names[winner] if winner is not None else "no majority",
                      "cluster_majority_votes":votes,
                      "neighbors":refs,
                      "llm_diagnosis":None,"llm_cluster":None,"verified":None})
    summary={"fixture_n":len(fixtures),"real_world_n":len(heldout),
             "k_selected":model.kmeans.n_clusters,
             "silhouette":round(model.silhouettes[model.kmeans.n_clusters],6),
             "audited_fixture_n":len(audit),
             "fixture_cluster_majority_match":sum(r["majority_match"] for r in audit),
             "patchrail_correct_broad":sum(c["patchrail_correct"] for c in cases),
             "retrieval_correct_broad":sum(c["tfidf_vote_correct"] for c in cases),
             "cluster_vote_exists":sum(c["cluster_majority_votes"]>=2 for c in cases),
             "llm_backend":args.backend,"llm_model":model_name if args.backend!="none" else None,
             "llm_status":"not_executed" if args.backend=="none" else "pending"}
    if args.backend!="none":
        try:
            for idx,c in enumerate(cases):
                refs=[{**n,"cluster_name":names[n["cluster"]]} for n in c["neighbors"]]
                refs=[{**n,"example":redact_ci_log(n["example"])["text"]} for n in refs]
                safe_new_text=redact_ci_log(heldout[idx].text)["text"]
                model_answer=diagnose(safe_new_text,refs,list(names.values()),backend=args.backend,
                                      model=model_name,base_url=args.base_url)
                majority=c["retrieved_cluster_majority"]
                c["llm_diagnosis"]=model_answer["diagnosis"]
                c["llm_cluster"]=model_answer["cluster"]
                c["verified"]=int(majority!="no majority" and model_answer["cluster"]==majority)
                selected_cluster=next(k for k,label in names.items() if label==model_answer["cluster"])
                source_broad=Counter(category_of_fixture(f.expected)
                                     for f,cluster in zip(model.fixtures,model.labels())
                                     if cluster==selected_cluster).most_common(1)[0][0]
                c["llm_correct_broad"]=int(source_broad==heldout[idx].expected)
        except Exception as exc:
            summary["llm_status"]="incomplete"
            summary["llm_error"] = f"{type(exc).__name__}: {exc}"
        else:
            summary["llm_status"]="completed"
            summary["llm_verified"] = sum(c["verified"] for c in cases)
            summary["llm_correct_broad"] = sum(c["llm_correct_broad"] for c in cases)
            if args.suggest:
                suggestions=[]
                for r in rows:
                    examples=[{"example":redact_ci_log(model.fixtures[i].text)["text"][:1200]} for i,m in enumerate(model.labels())
                              if m==r["cluster"]][:3]
                    suggestion=suggest_improvement(r["name"],examples,backend=args.backend,
                                                   model=model_name,base_url=args.base_url)
                    suggestions.append({"cluster":r["cluster"],"cluster_name":r["name"],
                                        "suggestion":suggestion,
                                        "actual_repository_fix_comparison":"not_assessed"})
                write_csv(out / "llm_improvement_suggestions.csv",suggestions)
    write_csv(out / "real_world_22_baselines.csv",[
        {k:v for k,v in c.items() if k!="neighbors"} for c in cases])
    with (out / "case_details.json").open("w",encoding="utf-8") as f:
        json.dump(cases,f,indent=2)
    with (out / "summary.json").open("w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    plot_outputs(out,model.silhouettes,rows,summary)
    print(json.dumps(summary,indent=2))
    return 0 if summary["llm_status"] in ("not_executed","completed") else 2


if __name__=="__main__":
    raise SystemExit(main())
