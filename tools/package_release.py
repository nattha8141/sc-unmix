"""Build sc-unmix.zip without modifying source, notebook or configuration."""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "VERIFICATION.md",
    "requirements.txt",
    "pyproject.toml",
    "SC_Unmix.ipynb",
)
TREES = {
    "sc_unmix": {".py"},
    "configs": {".yaml", ".json"},
    "tests": {".py"},
    "tools": {".py"},
}


def main():
    paths = [ROOT / name for name in FILES]
    paths.append(ROOT / "checkpoints/vocals_best.pt")
    for folder, suffixes in TREES.items():
        paths.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
            and p.suffix in suffixes
            and not p.is_symlink()
            and not any(
                part.startswith(".") or part == "__pycache__"
                for part in p.relative_to(ROOT).parts
            )
        )
    archive = ROOT.parent / "sc-unmix.zip"
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(paths):
            bundle.write(path, "sc-unmix/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
    print(f"{archive}: {len(paths)} files, {archive.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
