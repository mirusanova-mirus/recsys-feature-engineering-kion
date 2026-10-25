"""Package current sources, LaTeX and compact evidence without data or fitted weights."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def payload_files(root):
    files = set(root.glob("*.md"))
    files.update(root / name for name in ["Makefile", "requirements.txt", ".gitignore", "data/SOURCE_MANIFEST.json"])
    forbidden = {".git", "__pycache__", ".venv", "cache", "models", "predictions", "archive", "build", "tmp"}
    for name in ["recsys", "seqrl", "configs", "tests", "tools", "docs", "report", "artifacts", "provenance", "output/figures"]:
        folder = root / name
        if folder.exists():
            for path in folder.rglob("*"):
                relative = path.relative_to(root)
                if any(part in forbidden or part.startswith("reference_") for part in relative.parts):
                    continue
                if path.is_file() and path.suffix not in {".pdf", ".pyc", ".xdv", ".aux", ".log"} and path.name != ".DS_Store":
                    files.add(path)
    files = sorted(p for p in files if p.is_file())
    forbidden_suffixes = {".pt", ".cbm", ".joblib", ".parquet"}
    if any(p.suffix in forbidden_suffixes for p in files):
        raise ValueError("A fitted model or data cache entered the submission")
    return files


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names):
            raise ValueError("Unsafe or duplicate archive entries")
        manifest_name = next(n for n in names if n.endswith("/MANIFEST.sha256.json"))
        manifest = json.loads(archive.read(manifest_name))
        if set(names) != set(manifest["sha256"]) | {manifest_name}:
            raise ValueError("Archive entries differ from the manifest")
        for name, expected in manifest["sha256"].items():
            if sha(archive.read(name)) != expected:
                raise ValueError("Checksum mismatch: " + name)
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
    return {"archive": str(path.resolve()), "files": len(manifest["sha256"]), "bytes": path.stat().st_size, "sha256": sha(path.read_bytes()), "verified": True}


def build(root, destination):
    payload = {root.name + "/" + str(p.relative_to(root)): p.read_bytes() for p in payload_files(root)}
    manifest = {
        "description": "Current source snapshot, editable LaTeX and measured results",
        "omitted": "Raw data, fitted weights, large caches, prior PDF exports and editor build files",
        "sha256": {name: sha(data) for name, data in payload.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr(root.name + "/MANIFEST.sha256.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return verify(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="submission/project.zip")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = verify(args.verify) if args.verify else build(root, root / args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
