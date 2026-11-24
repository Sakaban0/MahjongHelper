"""Build the Windows source distribution from its public files."""
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from mahjong_helper import __version__

ROOT = Path(__file__).resolve().parents[1]


ZIP_NAME = f"MahjongHelper-Windows-{__version__}.zip"


def add_registered_tiles(archive):
    profile = ROOT.parent / "json"
    settings = json.loads((profile / "settings.json").read_text(encoding="utf-8-sig"))
    entries = []
    written = set()
    for tile in settings["templates"]:
        path = (profile / tile["path"]).resolve()
        if not path.is_relative_to((profile / "templates").resolve()):
            raise ValueError(f"見本の保存先を確認してください: {path}")
        data = path.read_bytes()
        name = hashlib.sha256(data).hexdigest() + path.suffix.lower()
        if name not in written:
            archive.writestr("app/config/registered/" + name, data)
            archive.writestr("json/templates/" + name, data)
            written.add(name)
        tile["path"] = "templates/" + name
        entry = dict(id=tile["id"], group=tile.get("group",tile["id"]), file=name)
        if "reviewed_theme" in tile:
            entry["reviewed_theme"] = tile["reviewed_theme"]
        entries.append(entry)
    settings["imported_bundle_entries"] = [json.dumps([e["id"], e["group"], e["file"]], ensure_ascii=False)
                                           for e in entries]
    archive.writestr("json/settings.json", json.dumps(settings, ensure_ascii=False, indent=2).encode("utf-8"))
    archive.writestr("app/config/registered/index.json",
                     json.dumps(entries, ensure_ascii=False, indent=2).encode("utf-8"))


def main():
    files = [ROOT / name for name in (
        "setup.cmd", "start.ps1",
        "docs/GUIDE.md", "docs/VALIDATION.md", "config/requirements-tested.txt",
        "config/requirements.txt", "config/settings.example.json",
        "mahjong_helper/visible_refs/north.png", "mahjong_helper/visible_refs/plum.png",
        "mahjong_helper/visible_refs/tiles.png", "mahjong_helper/visible_refs/tiles.json",
    )]
    for pattern in ("mahjong_helper/*.py", "mahjong_helper/*_patterns.npz"):
        files.extend(sorted(ROOT.glob(pattern)))
    for name in ("board_0.png", "board_1.png", "board_2.png", "board_3.png",
                 "board_4.png", "overlap_ambiguous.png", "settings.json",
                 "settings-layered.json", "templates/bars.png", "templates/circle.png",
                 "templates/cross.png"):
        files.append(ROOT / "examples/demo" / name)
    for case in ("south", "five"):
        for suffix in ("board.png", "settings.json", "example.png"):
            files.append(ROOT / "examples/demo-auto" / f"{case}-{suffix}")
    output = ROOT / "dist" / ZIP_NAME
    output.parent.mkdir(exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            archive.write(ROOT.parent / 'README.md', 'README.md')
            archive.write(ROOT.parent / 'Start-Desktop.vbs', 'Start-Desktop.vbs')
            archive.write(ROOT.parent / 'Start-Desktop.cmd', 'Start-Desktop.cmd')
            archive.writestr('json/', b'')
            archive.writestr('debug/', b'')
            add_registered_tiles(archive)
            for file in files:
                archive.write(file, Path("app") / file.relative_to(ROOT))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
