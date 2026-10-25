"""Run a report generator while preserving edited documents in the repository root."""
import argparse
from pathlib import Path
import re
import subprocess
import sys


def rebase_links(text):
    def replace(match):
        url = match[2]
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith(("/", "#")):
            return match[0]
        return match[1] + "../../" + url + match[3]
    return re.sub(r"(!?\[[^\]]*\]\()([^\s)]+)(\))", replace, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", action="append", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Specify the generator command after --")
    root = Path(__file__).resolve().parents[1]
    targets = [Path(name) for name in args.output]
    if any(p.is_absolute() or ".." in p.parts or len(p.parts) != 1 for p in targets):
        parser.error("Outputs must be filenames in the repository root")
    before = {}
    for relative in targets:
        path = root / relative
        before[relative] = (path.read_bytes(), path.stat().st_mode) if path.exists() else None
    try:
        result = subprocess.run(command, cwd=root)
        if result.returncode == 0:
            destination = root / "output/generated"
            destination.mkdir(parents=True, exist_ok=True)
            for relative, original in before.items():
                path = root / relative
                if path.exists() and (original is None or path.read_bytes() != original[0]):
                    text = rebase_links(path.read_text(encoding="utf-8"))
                    (destination / relative).write_text(text, encoding="utf-8")
                    print("Generated:", "output/generated/" + str(relative))
        return result.returncode
    finally:
        for relative, original in before.items():
            path = root / relative
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original[0])
                path.chmod(original[1])


if __name__ == "__main__":
    sys.exit(main())
