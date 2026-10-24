"""Package source and compact research evidence; never redistribute downloaded data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from .build import load_config


def main(config):
    out = Path(config["output_dir"])
    validation = json.loads((out / "validation.json").read_text())
    assert validation["passed"]
    extra = Path("artifacts/additional")
    replica = Path("artifacts/replication")
    independent = Path("artifacts/independent")
    for folder in [extra, replica, independent]:
        assert json.loads((folder / "validation.json").read_text())["passed"]
    assert json.loads((independent / "cohort_validation.json").read_text())["passed"]
    paths = [Path(n) for n in ["README.md", "REPORT.md", "ADDITIONAL.md", "requirements.txt", "Makefile", ".gitignore", "data/SOURCE_MANIFEST.json", "output/pdf/kion_feature_research.pdf"]]
    for pattern in ["configs/*.json", "recsys/*.py", "tests/*.py"]:
        paths.extend(Path(".").glob(pattern))
    for folder in [out, extra, replica, independent]:
        paths.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in [".json", ".csv", ".log"])
        paths.extend((folder / "figures").glob("*.png"))
    paths = sorted(set(paths))
    manifest = {"description": "Code, report and measured results; raw data, feature caches and models are reproduced by make reproduce.",
                "files": {str(p): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in paths}}
    destination = Path("submission")
    destination.mkdir(exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(destination / "recsys_kion.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, "recsys_kion/" + p.as_posix())
        z.write(destination / "manifest.json", "recsys_kion/SUBMISSION_MANIFEST.json")
    print(f"Packaged {len(paths)} files: {destination / 'recsys_kion.zip'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/main.json")
    main(load_config(p.parse_args().config))
