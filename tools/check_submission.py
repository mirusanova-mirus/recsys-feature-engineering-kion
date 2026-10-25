"""Check the compact submission using only the Python standard library."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CHECKS = []


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(message):
    CHECKS.append(message)


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, message, tolerance=1e-7):
    require(math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) < tolerance, message)


def markdown():
    excluded = {"EXPERIMENT_PLAN.md", "DIAGNOSTIC_PLAN.md"}
    files = list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md")) + [ROOT / "report/README.md"]
    count = 0
    for path in files:
        if path.name in excluded:
            continue
        text = path.read_text(encoding="utf-8")
        require(text.count("```") % 2 == 0, "Unclosed code fence: " + str(path))
        require(text.count("<details>") == text.count("</details>"), "Unclosed details: " + str(path))
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        formulas = re.findall(r"^```math[^\n]*\n(.*?)^```[ \t]*$", text, flags=re.M | re.S)
        formulas += re.findall(r"(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)", prose)
        require(all("<" not in formula for formula in formulas), "Use \\lt instead of a raw < in Markdown math: " + str(path))
        require(not any(c in prose for c in [";", "—", "–"]), "Document punctuation: " + str(path))
        require(not any(p.rstrip().endswith(".") for p in re.split(r"\n\s*\n", prose)), "Paragraph final dot: " + str(path))
        for _, target in re.findall(r"!?\[([^\]]*)\]\(([^\s)]+)\)", text):
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            require((path.parent / target.split("#")[0]).exists(), "Missing local link: " + str(path) + " -> " + target)
            count += 1
        for block in re.findall(r"(?:^\|.*\n)+", prose, flags=re.M):
            rows = [line.split("|")[1:-1] for line in block.splitlines()]
            require(len({len(row) for row in rows}) == 1, "Uneven table: " + str(path))
    record("Markdown structure, punctuation and " + str(count) + " local links")


def latex():
    report = ROOT / "report"
    body = (report / "sections/body.tex").read_text(encoding="utf-8")
    for name in re.findall(r"\\input\{([^{}]+)\}", body):
        path = report / (name + ".tex")
        require(path.exists(), "Missing LaTeX section: " + name)
        text = re.sub(r"^\s*%.*$", "", path.read_text(encoding="utf-8"), flags=re.M)
        require(not any(c in text for c in [";", "—", "–"]), "LaTeX punctuation: " + str(path))
        require(not any(p.rstrip().endswith(".") for p in re.split(r"\n\s*\n", text)), "LaTeX paragraph final dot: " + str(path))
        for image in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^{}]+)\}", text):
            require((report / image).exists(), "Missing LaTeX figure: " + image)
    record("Active LaTeX sections, figures and writing style")


def kion():
    lock = load("artifacts/main/selection.lock.json")
    require(sha(ROOT / "artifacts/main/audit.json") == lock["audit_sha256"], "KION audit seal changed")
    require(lock["config"] == load("configs/main.json"), "KION configuration changed")
    checked = 0
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    for folder in ["main", "additional", "replication", "independent"]:
        directory = ROOT / "artifacts" / folder
        rows = list(csv.DictReader((directory / "metrics.csv").open(encoding="utf-8")))
        for row in rows:
            if row["seed"] != "ensemble" or row["regime"] not in {"mixed", "uniform"}:
                continue
            path = directory / ("queries_" + row["regime"] + "_" + row["variant"] + ".csv")
            if not path.exists():
                continue
            queries = list(csv.DictReader(path.open(encoding="utf-8")))
            for metric in ["ndcg@10", "recall@10"]:
                close(math.fsum(float(q[metric]) for q in queries) / len(queries), float(row[metric]), "KION per-query mean: " + str(path))
            checked += 1
        require(load("artifacts/" + folder + "/validation.json")["passed"], "KION saved audit failed")
    for artifact, variant in [("main", "manual"), ("main", "automatic"), ("additional", "all_automatic")]:
        rows = list(csv.DictReader((ROOT / "artifacts" / artifact / "metrics.csv").open(encoding="utf-8")))
        row = next(r for r in rows if r["variant"] == variant and r["seed"] == "ensemble" and r["regime"] == "mixed")
        for metric in ["ndcg@10", "recall@10"]:
            require(format(float(row[metric]), ".5f") in report, "KION report metric missing")
    require(checked >= 12, "Too few KION per-query files checked")
    for relative in ["artifacts/additional/plan.lock.json", "artifacts/independent/plan.lock.json"]:
        plan = load(relative)
        inputs = plan.get("fingerprint", {}).get("inputs", {})
        for name, digest in inputs.items():
            path = ROOT / name
            if path.exists():
                require(sha(path) == digest, "KION follow-up input changed: " + name)
    record("KION seals, four saved audits and " + str(checked) + " recomputed ensemble tables")


def yambda():
    lock = load("artifacts/main/selection.lock.json")
    require(lock["config"] == load("configs/main.json"), "RL configuration changed")
    require(not lock["test_metrics_computed_before_seal"], "RL test preceded selection")
    require(sha(ROOT / "EXPERIMENT_PLAN.md") == lock["protocol_plan_sha256"], "RL protocol plan changed")
    require(sha(ROOT / "DIAGNOSTIC_PLAN.md") == lock["diagnostic_plan_sha256"], "RL diagnostic plan changed")
    for name, digest in lock["evaluation_source_sha256"].items():
        require(sha(ROOT / name) == digest, "RL sealed source changed: " + name)
    require(sha(ROOT / "artifacts/main/evaluation/confirmation/summary.json") == lock["confirmation_summary_sha256"], "RL confirmation changed")
    summary = load("artifacts/main/evaluation/test/summary.json")
    expected_ids = None
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    for name, stats in summary.items():
        with gzip.open(ROOT / ("artifacts/main/evaluation/test/" + name + "_mean.csv.gz"), "rt", encoding="utf-8") as stream:
            queries = list(csv.DictReader(stream))
        ids = [(q["qid"], q["uid"]) for q in queries]
        require(len(ids) == len(set(ids)), "RL duplicate query IDs")
        if expected_ids is None:
            expected_ids = ids
        require(ids == expected_ids, "RL candidate queries differ by method")
        for metric in ["ndcg@10", "recall@10", "map@10", "expected_completion", "expected_audit_completion"]:
            close(math.fsum(float(q[metric]) for q in queries) / len(queries), stats["mean"][metric], "RL per-query mean: " + name + "/" + metric)
        for metric, digits in [("ndcg@10", 5), ("recall@10", 5), ("expected_completion", 4), ("expected_audit_completion", 4)]:
            require(format(stats["mean"][metric], "." + str(digits) + "f") in report, "RL report metric missing: " + name)
    require(len(summary) == 10, "RL comparison is incomplete")
    record("RL source and plan seals, confirmation checksum and ten recomputed test summaries")


def main():
    markdown()
    latex()
    if (ROOT / "recsys").exists():
        kion()
    else:
        yambda()
    result = {"passed": True, "scope": "Compact evidence and document checks, full data/model audit is a separate command", "checks": CHECKS}
    destination = ROOT / "provenance/submission_check.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
