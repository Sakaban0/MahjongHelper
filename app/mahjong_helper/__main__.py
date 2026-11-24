import argparse
import json
from pathlib import Path
import sys


def import_bundled_tiles(profile):
    from hashlib import sha256
    from .core import Settings

    bundle = Path(__file__).resolve().parents[1] / 'config' / 'registered'
    index = bundle / 'index.json'
    if not index.is_file():
        return
    settings = Settings.load(profile)
    entries = json.loads(index.read_text(encoding='utf-8'))
    imported = set(settings.imported_bundle_entries)
    pending = [(entry, json.dumps([entry['id'], entry['group'], entry['file']], ensure_ascii=False))
               for entry in entries]
    pending = [(entry, key) for entry, key in pending if key not in imported]
    if not pending:
        return
    dest = profile.parent / 'templates'
    dest.mkdir(parents=True, exist_ok=True)
    existing = set()
    for tile in settings.templates:
        path = profile.parent / tile['path']
        if path.is_file():
            existing.add((tile['id'], sha256(path.read_bytes()).hexdigest()))
    for entry, key in pending:
        name = entry['file']
        if Path(name).name != name:
            raise ValueError('同梱した牌のファイル名が不正です')
        data = (bundle / name).read_bytes()
        digest = sha256(data).hexdigest()
        if digest != Path(name).stem:
            raise ValueError('同梱した牌の画像が破損しています')
        if (entry['id'], digest) not in existing:
            (dest / name).write_bytes(data)
            tile = dict(id=entry['id'], group=entry['group'], path='templates/'+name)
            if 'reviewed_theme' in entry:
                tile['reviewed_theme'] = entry['reviewed_theme']
            settings.templates.append(tile)
            existing.add((entry['id'], digest))
        settings.imported_bundle_entries.append(key)
    settings.save(profile)


def main():
    parser = argparse.ArgumentParser(description='画面画像だけを使う上海ペア補助アプリ')
    default_profile = Path(__file__).resolve().parents[2] / 'json' / 'settings.json'
    parser.add_argument('--config', default=default_profile)
    parser.add_argument('--image', help='起動時に開く静止画像')
    parser.add_argument('--analyze', help='GUIなしで画像を解析してJSON出力')
    parser.add_argument('--output', default='result.json')
    args = parser.parse_args()
    if Path(args.config).resolve() == default_profile:
        import_bundled_tiles(default_profile)
    from .core import Settings, Recognizer, read_image
    if args.analyze:
        result = Recognizer(Settings.load(args.config), Path(args.config).resolve().parent).analyze(read_image(args.analyze))
        Path(args.output).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{len(result.tiles)} tiles, {len(result.pairs)} pairs, {result.elapsed_ms:.1f} ms')
        return
    from .capture import enable_dpi_awareness
    enable_dpi_awareness()
    from PySide6.QtWidgets import QApplication
    from .ui import MainWindow
    app = QApplication(sys.argv)
    try:
        window = MainWindow(args.config)
        if args.image:
            window.image = read_image(args.image)
            window.canvas.set_image(window.image)
            window.recognize()
        window.show()
        sys.exit(app.exec())
    except Exception as exc:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, '起動エラー', str(exc))
        raise


if __name__ == '__main__':
    main()
