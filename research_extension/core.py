from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples" / "ci-triage"
REAL_WORLD = ROOT / "examples" / "real-world"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
TIMESTAMP_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b")
HASH_RE = re.compile(r"\b(?:[0-9a-fA-F]{10,}|[0-9a-fA-F]{6,}-[0-9a-fA-F-]{8,})\b")
PATH_RE = re.compile(r"(?<!\w)(?:[A-Za-z]:)?(?:[/\\][\w.\-]+){2,}")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
URL_RE = re.compile(r"https?://\S+")

CATEGORY_MAP = {
    "python_dependency_resolution": "dependency", "node_dependency_install": "dependency",
    "ruby_bundle_failure": "dependency", "php_composer_failure": "dependency",
    "database_migration_failure": "database", "python_test_failure": "test",
    "node_test_failure": "test", "go_test_failure": "test", "rust_test_failure": "test",
    "browser_test_failure": "test", "elixir_mix_failure": "test",
    "typescript_typecheck": "code_quality", "javascript_lint": "code_quality",
    "go_lint": "code_quality", "python_lint": "code_quality",
    "python_type_check": "code_quality", "shell_lint": "code_quality",
    "code_coverage_threshold": "code_quality",
    "java_build_failure": "build", "dotnet_build_failure": "build",
    "docker_build_failure": "build", "cpp_build_failure": "build",
    "xcode_build_failure": "build", "docs_build_failure": "build",
    "github_actions_workflow": "configuration", "helm_chart_failure": "configuration",
    "terraform_iac_failure": "configuration", "kubernetes_deploy_failure": "deployment",
    "release_publish_failure": "deployment", "network_transient_failure": "infrastructure",
    "runner_resource_exhaustion": "infrastructure", "ci_job_timeout": "infrastructure",
    "artifact_or_cache_failure": "infrastructure", "git_checkout_failure": "infrastructure",
    "git_merge_conflict": "configuration", "security_scan_failure": "security",
    "secrets_or_permissions_failure": "security", "unknown": "unknown",
}


def clean_log(text: str) -> str:
    text = ANSI_RE.sub(" ", text)
    text = TIMESTAMP_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = HASH_RE.sub(" ", text)
    text = PATH_RE.sub(" ", text)
    text = NUMBER_RE.sub(" ", text)
    text = re.sub(r"[^a-zA-Z\s_-]", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Record:
    name: str
    path: Path
    text: str
    expected: str
    source: str


def read_fixtures() -> list[Record]:
    result = []
    for file in sorted(FIXTURES.glob("*.log")):
        label_file = file.with_suffix(".expected.json")
        if not label_file.is_file():
            continue
        expected = json.loads(label_file.read_text(encoding="utf-8"))["failure_class"]
        result.append(Record(file.name, file, file.read_text(encoding="utf-8", errors="replace"), expected, "fixture"))
    return result


def reference_records() -> list[dict]:
    path = Path(__file__).with_name("real_world_reference.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        if not (REAL_WORLD / item["file"]).is_file():
            raise FileNotFoundError(REAL_WORLD / item["file"])
    return data


def read_real_world() -> list[Record]:
    return [Record(item["file"], REAL_WORLD / item["file"],
                   (REAL_WORLD / item["file"]).read_text(encoding="utf-8", errors="replace"),
                   item["category"], "real_world") for item in reference_records()]


@dataclass
class ClusterModel:
    vectorizer: TfidfVectorizer
    kmeans: KMeans
    fixtures: list[Record]
    texts: list[str]
    vectors: object
    silhouettes: dict[int, float]

    def labels(self) -> np.ndarray:
        return self.kmeans.labels_

    def match(self, text: str, count: int = 3) -> list[dict]:
        query = self.vectorizer.transform([clean_log(text)])
        similarity = cosine_similarity(query, self.vectors)[0]
        order = np.argsort(-similarity, kind="stable")[:count]
        return [{"file": self.fixtures[i].name, "score": round(float(similarity[i]), 6),
                 "cluster": int(self.kmeans.labels_[i]),
                 "fixture_label": self.fixtures[i].expected,
                 "example": self.fixtures[i].text[:1000]} for i in order]


def train_cluster_model(fixtures: list[Record]) -> ClusterModel:
    texts = [clean_log(item.text) for item in fixtures]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True, max_features=18000,
                                 token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z_-]{1,}\b")
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[0] < 9:
        raise ValueError("At least nine fixture logs are needed to compare k=4..8")
    scores = {}
    candidates = {}
    for k in range(4, 9):
        fitted = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300).fit(matrix)
        value = float(silhouette_score(matrix, fitted.labels_, metric="cosine"))
        scores[k] = value
        candidates[k] = fitted
    best = max(scores, key=lambda k: (scores[k], -k))
    return ClusterModel(vectorizer, candidates[best], fixtures, texts, matrix, scores)


def cluster_rows(model: ClusterModel, names: dict[int, str]) -> list[dict]:
    terms = np.asarray(model.vectorizer.get_feature_names_out())
    rows = []
    for c in range(model.kmeans.n_clusters):
        positions = np.flatnonzero(model.labels() == c).tolist()
        top = np.argsort(model.kmeans.cluster_centers_[c])[::-1][:8]
        counts = Counter(model.fixtures[i].expected for i in positions)
        rows.append({"cluster": c, "name": names[c], "size": len(positions),
                     "top_terms": terms[top].tolist(),
                     "examples": [model.fixtures[i].name for i in positions[:5]],
                     "dominant_reference_classes": counts.most_common(5)})
    return rows


def category_of_fixture(label: str) -> str:
    if label not in CATEGORY_MAP:
        raise KeyError(f"Unmapped fixture category: {label}")
    return CATEGORY_MAP[label]


def major_cluster(neighbors: list[dict]) -> tuple[int | None, int]:
    vote = Counter(n["cluster"] for n in neighbors)
    winner, count = vote.most_common(1)[0]
    return (winner if count >= 2 else None), count


def save_model(model: ClusterModel, out_dir: Path) -> None:
    joblib.dump({"vectorizer": model.vectorizer, "kmeans": model.kmeans,
                 "filenames": [f.name for f in model.fixtures], "vectors": model.vectors},
                out_dir / "cluster_model.joblib")
