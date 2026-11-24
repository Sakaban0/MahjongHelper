from __future__ import annotations

from pathlib import Path
import copy
import json
import os
import time
import uuid

import cv2
from PySide6.QtCore import Qt, QRectF, Signal, QObject, QRunnable, QThreadPool, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QShortcut, QKeySequence, QFontDatabase, QFontMetrics, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QMenu, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QSplitter, QScrollArea, QComboBox, QCheckBox, QDialog,
    QDialogButtonBox, QFormLayout, QDoubleSpinBox, QSpinBox, QLineEdit)

from . import __version__
from .core import (Settings, Recognizer, Tile, Result, StabilityGate, read_image,
                   write_image, infer_size, evaluate, project, pair_status, selection_highlight, same_face_detail)
from .capture import PrintWindowCapture, list_windows, active, client_geometry, roi_pixels


def pixmap(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy())


ZONE_TYPES = (('ignored_regions', '無視領域'), ('stock_regions', '保有マス'))


def zone_entries(settings):
    return [(key, index, rect) for key, _ in ZONE_TYPES
            for index, rect in enumerate(getattr(settings, key))]


def paint_zones(painter, entries, width, height, selected=-1):
    painter.save()
    for number, (_, _, (x, y, w, h)) in enumerate(entries):
        rect = QRectF(x*width, y*height, w*width, h*height)
        pen = QPen(QColor('black'), 4 if number == selected else 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRect(rect)
        if number == selected:
            pen.setColor(QColor('white'))
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(rect)
    painter.restore()


def paint_hints(painter, result, selected, all_tiles, labels=False, pair_count=1):
    if not result:
        return
    start = selected % len(result.pairs) if result.pairs else 0
    pairs = result.pairs[start:start+pair_count]
    colors = ['#00e5ff', '#ff78c8', '#ffc04d', '#a991ff', '#64e6ab']
    palette = getattr(result, 'preview_colors', {})
    chosen = {}
    for offset, pair in enumerate(pairs):
        for index in pair:
            chosen.setdefault(index, colors[palette.get(pair, offset) % len(colors)])
    for tile in result.tiles:
        held = tile.visual.get("stock", False)
        if not held and not labels and tile.index not in chosen and not (all_tiles and tile.state == '取れる'):
            continue
        color = chosen[tile.index] if tile.index in chosen else {'取れる':'#58e68b', '取れない':'#f97373'}.get(tile.state, '#ffd166')
        painter.setPen(QPen(QColor("#58e68b" if held else color), 2.5))
        painter.drawRect(QRectF(*tile.rect))
        if labels:
            x, y, _, _ = tile.rect
            label = f'#{tile.index}'
            if tile.auto_reason == '部分照合のため空き判定を保留':
                label += f' {tile.pattern}（部分）'
            painter.drawText(int(x), max(12, int(y)-3), label)
    lookup = {t.index: t for t in result.tiles}
    for offset, pair in enumerate(pairs):
        a, b = (lookup[i].rect for i in pair)
        painter.setPen(QPen(QColor(colors[palette.get(pair, offset) % len(colors)]), 3))
        painter.drawLine(int(a[0]+a[2]/2), int(a[1]+a[3]/2), int(b[0]+b[2]/2), int(b[1]+b[3]/2))


class PairNotice:
    def __init__(self, settings):
        self.phase = 'idle'
        self.count = 0
        self.configure(settings)

    @property
    def text(self):
        if self.phase is None and self.count:
            return self.found_text.replace('{件数}', str(self.count))
        return self.messages[self.phase]

    def configure(self, settings):
        self.found_text = settings.found_pairs_text
        self.messages = {'idle': settings.idle_text, None: settings.no_pair_text, 'recognizing': settings.recognizing_text,
                         'searching': settings.searching_text}
        self.font = QFont(QApplication.font())
        self.font.setPixelSize(settings.notice_font_size)
        self.position = list(settings.no_pair_position)

    def layout(self, width, height, metrics):
        metrics = QFontMetrics(self.font)
        text = metrics.elidedText(self.text, Qt.TextElideMode.ElideRight, max(0, int(width)-18))
        w = min(width, metrics.horizontalAdvance(text)+18)
        h = min(height, metrics.height()+10)
        return QRectF((width-w)*self.position[0], (height-h)*self.position[1], w, h), text

    def paint(self, painter, width, height):
        painter.save()
        painter.setFont(self.font)
        rect, text = self.layout(width, height, painter.fontMetrics())
        painter.setPen(QPen(QColor('red'), 2))
        painter.setBrush(QColor('white'))
        painter.drawRect(rect.adjusted(1, 1, -1, -1))
        painter.setPen(QColor('black'))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


class BoardScrollArea(QScrollArea):
    def wheelEvent(self, event):
        canvas = self.widget()
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier and canvas.image is not None and delta):
            return super().wheelEvent(event)
        pos = event.position()
        bars = (self.horizontalScrollBar(), self.verticalScrollBar())
        anchor = [(bar.value()+point)/canvas.scale for bar,point in zip(bars,(pos.x(),pos.y()))]
        canvas.zoom = max(.25, min(5., canvas.zoom * 1.2**(max(-8, min(8, delta/120)))))
        canvas.set_image(canvas.image)
        for bar, coord, point in zip(bars, anchor, (pos.x(),pos.y())):
            bar.setValue(round(coord*canvas.scale-point))
        event.accept()


class Canvas(QWidget):
    notice_moved = Signal()
    selected = Signal(tuple)
    zone_picked = Signal(int)
    tile_picked = Signal(int)
    tile_activated = Signal(int)

    def __init__(self):
        super().__init__()
        self.notice = None
        self.notice_offset = None
        self.image = self.result = None
        self.pair_count = 1
        self.pair = 0
        self.all_tiles = False
        self.debug = True
        self.zoom = 1.
        self.scale = 1.
        self.zones = []
        self.selected_zone = -1
        self.region_drag = False
        self.drag_start = self.drag_end = None
        self.selected_tile = None
        self.setMinimumSize(500, 300)

    def set_image(self, image):
        self.image = image
        self.scale = self.zoom * min(1., 1000/image.shape[1], 650/image.shape[0])
        self.setFixedSize(round(image.shape[1]*self.scale), round(image.shape[0]*self.scale))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#17232a'))
        if self.image is None:
            painter.setPen(QColor('white'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '対象ウィンドウを選択してください')
            if self.notice and self.notice.text and self.debug:
                self.notice.paint(painter, self.width(), self.height())
            return
        painter.scale(self.scale, self.scale)
        painter.drawPixmap(0, 0, pixmap(self.image))
        paint_hints(painter, self.result, self.pair, self.all_tiles, self.debug, self.pair_count)
        if self.result and self.selected_tile is not None:
            tile = next((t for t in self.result.tiles if t.index == self.selected_tile), None)
            if tile:
                painter.setPen(QPen(QColor('#ff5cdd'), 3))
                painter.drawRect(QRectF(*tile.rect))
        if self.debug:
            paint_zones(painter, self.zones, self.image.shape[1], self.image.shape[0], self.selected_zone)
        if self.region_drag and self.drag_start and self.drag_end:
            painter.setPen(QPen(QColor('magenta'), 2))
            painter.drawRect(QRectF(self.drag_start[0], self.drag_start[1],
                                    self.drag_end[0]-self.drag_start[0], self.drag_end[1]-self.drag_start[1]).normalized())

        if self.notice and self.notice.text and self.debug and not self.region_drag:
            self.notice.paint(painter, self.image.shape[1], self.image.shape[0])

    def mousePressEvent(self, event):
        if self.image is not None and event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = (event.position().x()/self.scale, event.position().y()/self.scale)
            if self.notice and self.notice.text and self.debug and not self.region_drag:
                h, w = self.image.shape[:2]
                rect, _ = self.notice.layout(w, h, self.fontMetrics())
                x, y = self.drag_start
                if rect.contains(x, y):
                    self.notice_offset = (x-rect.x(), y-rect.y())
                    self.drag_start = self.drag_end = None
                    return
            self.drag_end = self.drag_start

    def move_notice(self, pos):
        h, w = self.image.shape[:2]
        rect, _ = self.notice.layout(w, h, self.fontMetrics())
        self.notice.position = [min(1., max(0., (value/self.scale-offset)/max(1., span)))
                                for value, offset, span in zip((pos.x(), pos.y()),
                                    self.notice_offset, (w-rect.width(), h-rect.height()))]
        self.update()

    def mouseMoveEvent(self, event):
        if self.notice_offset is not None:
            self.move_notice(event.position())
            return
        if self.drag_start:
            self.drag_end = (event.position().x()/self.scale, event.position().y()/self.scale)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.notice_offset is not None:
            self.move_notice(event.position())
            self.notice_offset = None
            self.notice_moved.emit()
            return
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.position().x()/self.scale, event.position().y()/self.scale
        h, w = self.image.shape[:2]
        x, y = max(0, round(min(x0, x1))), max(0, round(min(y0, y1)))
        right, bottom = min(w, round(max(x0, x1))), min(h, round(max(y0, y1)))
        self.drag_start = self.drag_end = None
        if max(abs(x1-x0), abs(y1-y0))*self.scale < 5:
            if not self.region_drag and self.debug:
                hits = [(rw*rh, i) for i, (_, _, (rx, ry, rw, rh)) in enumerate(self.zones)
                        if rx <= x1/w <= rx+rw and ry <= y1/h <= ry+rh]
                if hits:
                    self.zone_picked.emit(min(hits)[1])
            self.pick_at(x1, y1)
            self.update()
            return
        if self.region_drag and right-x >= 8 and bottom-y >= 8:
            self.selected.emit((x, y, right-x, bottom-y))
        self.update()

    def pick_at(self, x, y):
        candidates = [t for t in self.result.tiles if QRectF(*t.rect).contains(x, y)] if self.result else []
        if candidates:
            tile = min(candidates, key=lambda t: t.rect[2]*t.rect[3])
            self.selected_tile = tile.index
            self.tile_picked.emit(tile.index)
            return tile.index
        return None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            picked = self.pick_at(event.position().x()/self.scale, event.position().y()/self.scale)
            self.drag_start = self.drag_end = None
            if picked is not None:
                self.tile_activated.emit(picked)


class Overlay(QWidget):
    def __init__(self):
        flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                 | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput
                 | Qt.WindowType.WindowDoesNotAcceptFocus)
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.result = None
        self.pair_count = 1
        self.pair = 0
        self.all_tiles = False
        self.zones = []
        self.notice = None
        self.pixel_size = (1, 1)
        self.affinity = False

    def place(self, rect):
        if not (self.notice and self.notice.phase and self.notice.text) and (not self.result or not (self.result.pairs or (self.notice and self.notice.text) or any(t.visual.get('stock') for t in self.result.tiles))):
            self.hide()
            return
        x, y, w, h = rect
        self.pixel_size = (w, h)
        self.show()
        if os.name == 'nt':
            from .capture import user
            # Extra protection; PrintWindow already captures only the target.
            self.affinity = bool(user.SetWindowDisplayAffinity(int(self.winId()), 0x11))
            user.SetWindowPos(int(self.winId()), -1, x, y, w, h, 0x0010 | 0x0040)  # NOACTIVATE
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(self.width()/self.pixel_size[0], self.height()/self.pixel_size[1])
        paint_hints(painter, self.result, self.pair, self.all_tiles, pair_count=self.pair_count)
        if self.notice and self.notice.text and (self.notice.phase or self.result is not None):
            self.notice.paint(painter, *self.pixel_size)


class JobSignals(QObject):
    searching = Signal(int)
    done = Signal(int, object, object)
    failed = Signal(int, str)


class CaptureJob(QRunnable):
    def __init__(self, token, provider, hwnd, settings):
        super().__init__()
        self.token, self.provider, self.hwnd, self.settings = token, provider, hwnd, settings
        self.signals = JobSignals()

    def run(self):
        try:
            frame = self.provider.capture(self.hwnd, self.settings.roi)
            self.signals.done.emit(self.token, frame, None)
        except Exception as exc:
            self.signals.failed.emit(self.token, str(exc))


class AnalyzeJob(QRunnable):
    def __init__(self, token, image, settings, base):
        super().__init__()
        self.token, self.image, self.settings, self.base = token, image, settings, base
        self.signals = JobSignals()

    def run(self):
        try:
            result = Recognizer(self.settings, self.base).analyze(
                self.image, lambda: self.signals.searching.emit(self.token))
            self.signals.done.emit(self.token, self.image, result)
        except Exception as exc:
            self.signals.failed.emit(self.token, str(exc))


class MainWindow(QMainWindow):
    HEADERS = ['ID', '画像x', '画像y', '幅', '高さ', '柄ID (?=不明)', '信頼度', '盤面x', '盤面y', '段数 (?=不明)', '状態', '理由 / 上の牌', '試験・相対段（基準=0）']

    def __init__(self, config_path):
        super().__init__()
        if os.name == 'nt' and not getattr(QApplication.instance(), '_helper_font_set', False):
            font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'meiryo.ttc'
            if font_path.exists():
                font_id = QFontDatabase.addApplicationFont(str(font_path))
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    QApplication.instance().setFont(QFont(families[0], 9))
            QApplication.instance()._helper_font_set = True
        self.config_path = Path(config_path).resolve()
        self.base = self.config_path.parent
        self.settings = Settings.load(self.config_path)
        self.gate = StabilityGate(self.settings)
        self.provider = PrintWindowCapture()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.generation = 0
        self.busy = False
        self._running = False
        self.hwnd = None
        self.image = self.result = self.frame = None
        self.last_debug = None
        self.source = 'file'
        self.last_geometry = None
        self.preview_deadline = 0.0
        self.validated_frame = False
        self.overlay = Overlay()
        self.setWindowTitle('上海 画面認識ヘルパー — 画像のみ・クリック操作なし')
        size = [self.settings.initial_width, self.settings.initial_height]
        if self.settings.remember_size and len(self.settings.last_size) == 2:
            size = self.settings.last_size
        self.resize(*size)
        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.windows = QComboBox()
        self.windows.setFixedWidth(230)
        self.windows.setAccessibleName('対象ウィンドウ')
        row.addWidget(self.windows)
        refresh = self.button(row, '↻', self.refresh_windows)
        self.button(row, '取得', self.select_window)
        regions = QPushButton('領域設定')
        row.addWidget(regions)
        menu = QMenu(regions)
        regions.setMenu(menu)
        menu.addAction('盤面の範囲を設定', lambda: self.safe(self.reset_region))
        for key, label in [('ignored_regions', '無視領域'), ('stock_regions', '保有マス')]:
            menu.addAction(label+'を追加', lambda checked=False, k=key: self.safe(lambda: self.edit_zone(k)))
        self.play_button = self.button(row, '▶', self.toggle_playback)
        next_button = self.button(row, '→', self.next_pair)
        self.raw_box = QCheckBox('元画像のみ')
        self.raw_box.toggled.connect(self.display_options)
        row.addWidget(self.raw_box)
        row.addStretch()
        self.quick_folder = None
        self.quick_number = 1
        self.quick_button = self.button(row, 'クイックデバッグ', self.quick_debug)
        self.quick_button.setVisible(self.settings.debug_mode)
        table_button = QPushButton('表を表示')
        table_button.setCheckable(True)
        row.addWidget(table_button)
        gear = self.button(row, '⚙', self.edit_settings)
        for button, name in ((refresh, '一覧更新'), (gear, '設定'),
                             (self.play_button, '開始'), (next_button, '次のペア')):
            button.setAccessibleName(name)
            button.setToolTip(name)
            button.setFixedSize(36, 32)
            button.setStyleSheet(
                "QPushButton { color: black; background: #f5f5f5; border: 1px solid #999;"
                " border-radius: 4px; font: 20px 'Segoe UI Symbol'; }"
                "QPushButton:hover { background: #e1e7eb; }"
                "QPushButton:focus { border: 2px solid #2879ac; }")
        for position in range(row.count()):
            control = row.itemAt(position).widget()
            if control is not None:
                control.setFixedHeight(32)
        self.selecting_region = False
        self.original_image = None
        self.pair_banner = QLabel('ペア線：画像を読み込み、認識してください。')
        self.pair_banner.setWordWrap(True)
        self.pair_banner.setStyleSheet('QLabel { background: #fff0c2; color: #3b3218; padding: 8px; border: 1px solid #dab854; }')
        layout.addWidget(self.pair_banner)
        split = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(split, 1)
        self.canvas = Canvas()
        self.canvas.notice = self.overlay.notice = PairNotice(self.settings)
        self.canvas.notice_moved.connect(self.save_notice_position)
        self.canvas.selected.connect(self.selection)
        self.canvas.tile_picked.connect(self.pick_tile)
        self.canvas.tile_activated.connect(lambda index: self.safe(self.register_selected_template))
        scroll = BoardScrollArea()
        scroll.setWidget(self.canvas)
        split.addWidget(scroll)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.itemChanged.connect(self.table_changed)
        self.table.itemSelectionChanged.connect(self.table_selection_changed)
        split.addWidget(self.table)
        split.setSizes([560, 220])
        self.table.hide()
        table_button.toggled.connect(self.table.setVisible)
        table_button.toggled.connect(lambda shown: table_button.setText('表を隠す' if shown else '表を表示'))
        selection_row = QHBoxLayout()
        layout.addLayout(selection_row)
        self.tile_preview = QLabel('牌をクリックして選択')
        self.tile_preview.setFixedSize(60, 80)
        self.tile_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tile_preview.setWordWrap(True)
        selection_row.addWidget(self.tile_preview)
        self.selection_info = QLabel('画像の牌または表の行をクリックして選択。\n牌をダブルクリックすると登録画面を開けます。')
        self.selection_info.setWordWrap(True)
        selection_row.addWidget(self.selection_info, 1)
        zone_row = QHBoxLayout()
        layout.addLayout(zone_row)
        zone_row.addWidget(QLabel('登録領域'))
        self.zone_list = QComboBox()
        self.zone_list.setAccessibleName('登録領域')
        self.zone_list.setPlaceholderText('領域を選択')
        zone_row.addWidget(self.zone_list)
        self.delete_zone_button = self.button(zone_row, '選択領域を削除', self.delete_zone)
        zone_row.addStretch()
        self.zone_list.currentIndexChanged.connect(self.select_zone)
        self.canvas.zone_picked.connect(self.zone_list.setCurrentIndex)
        self.reload_zones()
        self.status = QLabel('画像と見本を登録してください。未確認の配置はペア表示しません。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.refresh_windows()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(self.settings.interval_ms)
        self.follow = QTimer(self)
        self.follow.timeout.connect(self.follow_target)
        self.follow.start(80)
        QShortcut(QKeySequence('Ctrl+Right'), self, activated=self.next_pair)
        QShortcut(QKeySequence('Ctrl+Space'), self, activated=self.pause)

    def button(self, row, label, callback):
        button = QPushButton(label)
        button.clicked.connect(lambda checked=False: self.safe(callback))
        row.addWidget(button)
        return button

    def safe(self, callback):
        try:
            callback()
        except Exception as exc:
            QMessageBox.warning(self, '処理できません', str(exc))

    def invalidate(self, clear=False):
        self.canvas.notice.phase = 'idle'
        self.canvas.update()
        self.clear_pair_preview()
        self.generation += 1
        self.overlay.hide()
        self.overlay.result = None
        self.validated_frame = False
        self.gate.reset()
        if clear:
            self.canvas.notice.count = 0
            self.result = self.canvas.result = None
            self.canvas.selected_tile = None
            self.tile_preview.clear()
            self.table.setRowCount(0)
            self.canvas.update()
            self.pair_banner.setText('ペア線：' + pair_status(None))

    def refresh_windows(self):
        self.windows.clear()
        for hwnd, title in list_windows():
            if hwnd != int(self.winId()) and hwnd != int(self.overlay.winId()):
                self.windows.addItem(title, hwnd)
        index = self.windows.findText(self.settings.window_title)
        if index >= 0:
            self.windows.setCurrentIndex(index)

    def select_window(self):
        self.selecting_region = self.canvas.region_drag = False
        self.pause()
        self.invalidate(True)
        self.hwnd = self.windows.currentData()
        if not self.hwnd:
            raise ValueError('対象ウィンドウを選択してください')
        self.settings.window_title = self.windows.currentText()
        self.settings.save(self.config_path)
        self.source = 'live'
        self.frame = None
        self.submit_capture()

    def edit_zone(self, key):
        if self.image is None or self.busy:
            raise ValueError('盤面を取得し、認識が完了してから設定してください')
        self.pause()
        self.selecting_region = key
        self.canvas.region_drag = True
        label = '無視する範囲' if key == 'ignored_regions' else '保有マス'
        self.status.setText(label + 'をドラッグして囲んでください。')

    def reload_zones(self):
        entries = zone_entries(self.settings)
        self.canvas.zones = entries
        self.zone_list.clear()
        for key, index, _ in entries:
            self.zone_list.addItem(f'{dict(ZONE_TYPES)[key]} {index+1}')
        self.zone_list.setCurrentIndex(-1)
        self.select_zone(-1)
        self.overlay.update()

    def select_zone(self, index):
        self.canvas.selected_zone = index
        self.delete_zone_button.setEnabled(index >= 0)
        self.canvas.update()

    def delete_zone(self):
        index = self.zone_list.currentIndex()
        if index < 0:
            return
        if self.busy:
            raise ValueError('画像処理の完了後に操作してください')
        key, number, _ = self.canvas.zones[index]
        candidate = copy.deepcopy(self.settings)
        del getattr(candidate, key)[number]
        candidate.save(self.config_path)
        self.settings = candidate
        self.gate = StabilityGate(candidate)
        self.selecting_region = self.canvas.region_drag = False
        self.reload_zones()
        self.recognize()

    def reset_region(self):
        if self.source == 'file' and self.image is None:
            raise ValueError('先に画像を開くか、対象を取得してください')
        candidate = copy.deepcopy(self.settings)
        candidate.ignored_regions.clear()
        candidate.stock_regions.clear()
        if self.source == 'live':
            candidate.roi = [0., 0., 1., 1.]
        candidate.save(self.config_path)
        self.settings = candidate
        self.gate = StabilityGate(candidate)
        self.reload_zones()
        self.pause()
        self.invalidate(True)
        self.frame = None
        self.selecting_region = self.canvas.region_drag = True
        if self.source == 'live':
            self.image = self.canvas.image = None
            self.canvas.update()
            self.capture_reset_region(self.generation)
        else:
            if self.original_image is None:
                self.original_image = self.image
            self.image = self.original_image
            self.canvas.set_image(self.image)
        self.status.setText('画像全体から盤面の範囲をドラッグしてください。')

    def capture_reset_region(self, token):
        if token != self.generation:
            return
        if self.busy:
            QTimer.singleShot(50, lambda: self.capture_reset_region(token))
        else:
            self.submit_capture()

    def save_notice_position(self):
        self.settings.no_pair_position = list(self.canvas.notice.position)
        self.settings.save(self.config_path)
        self.overlay.update()

    def edit_settings(self):
        self.pause()
        dialog = QDialog(self)
        dialog.setWindowTitle('認識と盤面の設定')
        dialog.resize(650, 650)
        box = QVBoxLayout(dialog)
        box.addWidget(QLabel('牌の表面の大きさと、1段ごとの描画ずれを指定します。\n画像上の大きさは実ピクセル、盤面範囲は対象内の0〜1の比率です。'))
        form = QFormLayout()
        form.addRow("アプリのバージョン", QLabel(__version__))
        box.addLayout(form)
        fields = {}
        specs = [('pair_count', '同時表示するペア数（組）', 1, 99, 0),
                 ('initial_width', '起動時の幅（px）', 760, 16384, 0),
                 ('initial_height', '起動時の高さ（px）', 360, 16384, 0),
                 ('tile_width', '牌の表面の幅', 8, 4000, 2),
                 ('tile_height', '牌の表面の高さ', 8, 4000, 2),
                 ('layer_dx', '1段上がるときの横のずれ', -1000, 1000, 2),
                 ('layer_dy', '1段上がるときの縦のずれ', -1000, 1000, 2),
                 ('threshold', '柄の認識しきい値', 0, 1, 3),
                 ('ambiguity_margin', '柄候補1位・2位に必要な差', 0, 1, 3),
                 ('interval_ms', '画面の確認間隔（ms）', 100, 60000, 0),
                 ('stable_frames', '安定を確認するフレーム数', 2, 100, 0),
                 ('change_threshold', '画面の変化を許容する平均差', 0, 255, 2),
                 ('surface_brightness', '牌の表面の明るさの下限', 0, 255, 0)]
        for key, label, low, high, decimals in specs:
            field = QDoubleSpinBox() if decimals else QSpinBox()
            field.setRange(low, high)
            if decimals:
                field.setDecimals(decimals)
                field.setSingleStep(.01 if high == 1 else 1)
            field.setValue(getattr(self.settings, key))
            fields[key] = field
            form.addRow(label, field)
        notice_text = QLineEdit(self.settings.no_pair_text)
        form.addRow('ペアがないときの表示文', notice_text)
        idle_text = QLineEdit(self.settings.idle_text)
        form.addRow('待機中の表示文', idle_text)
        found_text = QLineEdit(self.settings.found_pairs_text)
        form.addRow('ペア発見時の表示文（{件数}）', found_text)
        recognizing = QLineEdit(self.settings.recognizing_text)
        searching = QLineEdit(self.settings.searching_text)
        font_size = QSpinBox()
        font_size.setRange(8, 96)
        font_size.setSuffix(' px')
        font_size.setValue(self.settings.notice_font_size)
        form.addRow('画像認識中の表示文', recognizing)
        form.addRow('ペア検索中の表示文', searching)
        form.addRow('案内の文字サイズ', font_size)
        stock = QCheckBox('保有マスの牌を優先')
        height = QCheckBox('段数の高い牌を優先')
        stock.setChecked(self.settings.prioritize_stock)
        height.setChecked(self.settings.prioritize_height)
        form.addRow(stock)
        form.addRow(height)
        priority = QComboBox()
        def update_priority_order():
            priority.setEnabled(stock.isChecked() and height.isChecked())
        stock.toggled.connect(update_priority_order)
        height.toggled.connect(update_priority_order)
        update_priority_order()
        priority.addItem('保有マス → 高い段', 'stock')
        priority.addItem('高い段 → 保有マス', 'height')
        priority.setCurrentIndex(max(0,priority.findData(self.settings.pair_priority)))
        form.addRow('ペアの優先順位', priority)
        debug_mode = QCheckBox('デバッグモード')
        debug_mode.setChecked(self.settings.debug_mode)
        form.addRow(debug_mode)
        remember = QCheckBox('次回は終了時のウィンドウサイズを使う')
        remember.setChecked(self.settings.remember_size)
        form.addRow(remember)
        flat = QCheckBox('このゲームは重なりのない単層盤面と確認しました')
        flat.setChecked(self.settings.flat_board)
        form.addRow('単層の描画規則', flat)
        auto_pairs = QCheckBox('画像から取れる牌とペアを自動判定する')
        auto_pairs.setChecked(self.settings.auto_pairs)
        form.addRow('ペアの自動認識', auto_pairs)
        learn = QCheckBox('一致した未知の柄を見本として保存')
        learn.setChecked(self.settings.auto_register)
        form.addRow('柄の自動登録', learn)
        shade = QCheckBox('この盤面の明るい牌面を空き判定に使用')
        shade.setChecked(self.settings.use_bright_face_cue)
        form.addRow('テーマ固有の色判定', shade)
        roi_fields = []
        roi_row = QHBoxLayout()
        for label, value in zip(('左', '上', '幅', '高さ'), self.settings.roi):
            roi_row.addWidget(QLabel(label))
            field = QDoubleSpinBox()
            field.setRange(0, 1)
            field.setDecimals(6)
            field.setSingleStep(.01)
            field.setValue(value)
            roi_fields.append(field)
            roi_row.addWidget(field)
        box.addLayout(roi_row)
        action_row = QHBoxLayout()
        box.addLayout(action_row)
        self.button(action_row, '登録した見本の一覧・編集', self.manage_templates)
        box.addWidget(QLabel(f'設定の保存先: {self.config_path}'))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        box.addWidget(buttons)
        if dialog.exec():
            candidate = copy.deepcopy(self.settings)
            for key, field in fields.items():
                setattr(candidate, key, field.value())
            candidate.idle_text = idle_text.text().strip()
            candidate.found_pairs_text = found_text.text().strip()
            candidate.recognizing_text = recognizing.text().strip()
            candidate.searching_text = searching.text().strip()
            candidate.notice_font_size = font_size.value()
            candidate.no_pair_text = notice_text.text().strip()
            candidate.prioritize_stock = stock.isChecked()
            candidate.prioritize_height = height.isChecked()
            candidate.pair_priority = priority.currentData()
            candidate.debug_mode = debug_mode.isChecked()
            candidate.remember_size = remember.isChecked()
            candidate.flat_board = flat.isChecked()
            candidate.auto_pairs = auto_pairs.isChecked()
            candidate.auto_register = learn.isChecked()
            candidate.use_bright_face_cue = shade.isChecked()
            candidate.roi = [f.value() for f in roi_fields]
            candidate.validate()
            Recognizer(candidate, self.base)
            candidate.save(self.config_path)
            previous = copy.deepcopy(self.settings)
            previous.no_pair_text = candidate.no_pair_text
            previous.idle_text = candidate.idle_text
            previous.found_pairs_text = candidate.found_pairs_text
            previous.recognizing_text = candidate.recognizing_text
            previous.searching_text = candidate.searching_text
            previous.notice_font_size = candidate.notice_font_size
            self.canvas.notice.configure(candidate)
            self.canvas.update()
            self.overlay.update()
            previous.prioritize_stock = candidate.prioritize_stock
            previous.prioritize_height = candidate.prioritize_height
            previous.pair_priority = candidate.pair_priority
            previous.pair_count = candidate.pair_count
            previous.debug_mode = candidate.debug_mode
            self.quick_button.setVisible(candidate.debug_mode)
            self.settings = candidate
            self.gate.settings = candidate
            if previous == candidate and self.result is not None:
                from .core import prioritize_pairs
                prioritize_pairs(self.result, candidate)
                self.canvas.pair = 0
                self.show_result()
                self.display_options()
                return
            self.gate = StabilityGate(candidate)
            self.timer.setInterval(candidate.interval_ms)
            self.invalidate(True)
            self.status.setText('設定を保存しました。対象の再取得または画像の再認識を行ってください。')

    def manage_templates(self):
        self.pause()
        dialog = QDialog(self)
        dialog.setWindowTitle('登録した柄の見本')
        dialog.resize(850, 500)
        box = QVBoxLayout(dialog)
        box.addWidget(QLabel('柄IDとペアグループを修正できます。同じ柄の追加見本は同じID・グループに揃えてください。'))
        table = QTableWidget(len(self.settings.templates), 3)
        table.setHorizontalHeaderLabels(['柄ID', 'ペア判定グループ', '見本画像'])
        for row, entry in enumerate(self.settings.templates):
            for col, value in enumerate((entry['id'], entry.get('group') or entry['id'], entry['path'])):
                table.setItem(row, col, QTableWidgetItem(value))
        table.setColumnWidth(2, 450)
        box.addWidget(table)
        remove = QPushButton('選択した見本を登録から外す（画像ファイルは残します）')
        remove.clicked.connect(lambda: [table.removeRow(row) for row in sorted({i.row() for i in table.selectedIndexes()}, reverse=True)])
        box.addWidget(remove)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        box.addWidget(buttons)
        if dialog.exec():
            candidate = copy.deepcopy(self.settings)
            candidate.templates = []
            for row in range(table.rowCount()):
                pattern, group, path = (table.item(row, c).text().strip() for c in range(3))
                if not pattern or not path:
                    raise ValueError('柄IDと画像ファイル名は必須です')
                candidate.templates.append({'id': pattern, 'group': group or pattern, 'path': path})
            Recognizer(candidate, self.base)
            candidate.save(self.config_path)
            self.settings = candidate
            self.gate = StabilityGate(candidate)
            self.invalidate(True)

    @property
    def running(self):
        return self._running

    @running.setter
    def running(self, value):
        self._running = value
        label = '一時停止' if value else '開始'
        self.play_button.setText('⏸︎' if value else '▶')
        self.play_button.setToolTip(label)
        self.play_button.setAccessibleName(label)

    def toggle_playback(self):
        if self.running:
            self.pause()
        else:
            self.start()

    def start(self):
        self.selecting_region = self.canvas.region_drag = False
        if self.source != 'live' or not self.hwnd:
            raise ValueError('先に対象ウィンドウを設定してください')
        # Resume reviewed geometry/accessibility only if the next captured
        # frame still matches; never display the saved hints before validation.
        if self.result is not None and self.frame is not None:
            self.generation += 1
            self.overlay.hide()
            self.validated_frame = False
            self.gate.reset()
            for _ in range(self.settings.stable_frames):
                self.gate.push(self.image)
            self.canvas.notice.phase = 'idle'
            self.overlay.result = None
            self.canvas.update()
        else:
            self.invalidate(True)
        self.running = True
        self.status.setText('ライブ開始。ゲームウィンドウへ切り替えてください。')

    def pause(self):
        self.running = False
        self.invalidate()
        self.status.setText('一時停止。補助表示を隠しました。')

    def reset_board(self):
        self.invalidate(True)
        self.frame = None
        self.status.setText('過去の配置・修正・候補を破棄しました。')
        if self.source == 'file' and self.image is not None:
            self.recognize()

    def tick(self):
        if self.running and self.hwnd and active(self.hwnd) and not self.busy:
            self.submit_capture()

    def clear_pair_preview(self):
        if self.preview_deadline:
            self.preview_deadline = 0.0
            self.overlay.result = self.canvas.result = None
            self.overlay.hide()
            self.canvas.update()

    def update_pair_preview(self):
        # Display-only snapshot: provisional lines never become playable results.
        if self.result is not None and self.last_debug and self.last_debug[1] is self.result:
            first = self.canvas.pair
            pairs = self.result.pairs[first:first+self.settings.pair_count]
            previous = self.result
            self.preview_deadline = time.monotonic() + 1.5
        elif self.preview_deadline and self.overlay.result:
            previous = self.overlay.result
            pairs = previous.pairs
        else:
            return
        reference = self.last_debug[0]
        if time.monotonic() >= self.preview_deadline or reference.shape != self.image.shape:
            self.clear_pair_preview()
            return
        present = {}
        needed = {i for pair in pairs for i in pair}
        for tile in previous.tiles:
            if tile.index not in needed:
                continue
            x,y,w,h = map(round, tile.rect)
            old = reference[y:y+h,x:x+w]
            new = self.image[y:y+h,x:x+w]
            if same_face_detail(old, new):
                present[tile.index] = tile
        remaining = [pair for pair in pairs if all(i in present for i in pair)]
        if not remaining:
            self.clear_pair_preview()
            return
        indices = {i for pair in remaining for i in pair}
        preview = Result([present[i] for i in sorted(indices)], remaining)
        preview.preview_colors = getattr(previous, "preview_colors", {pair: i for i, pair in enumerate(pairs)})
        self.overlay.result = self.canvas.result = preview
        self.overlay.pair = self.canvas.pair = 0
        self.overlay.update()
        self.canvas.update()

    def follow_target(self):
        if self.preview_deadline and time.monotonic() >= self.preview_deadline:
            self.clear_pair_preview()
            self.pair_banner.setText('ペア線：盤面を確認中。')
        if self.source != 'live' or not self.hwnd or not active(self.hwnd):
            self.overlay.hide()
            self.validated_frame = False
            return
        try:
            x, y, w, h = client_geometry(self.hwnd)
            if self.frame and (w, h) != self.frame.client_size:
                self.invalidate(True)
                self.frame = None
                self.status.setText('表示サイズが変わりました。配置を再構築します。')
                return
            if self.running and (self.canvas.notice.phase or
                    (self.frame and self.overlay.result and (self.validated_frame or self.preview_deadline))):
                rx, ry, rw, rh = roi_pixels(w, h, self.settings.roi)
                self.overlay.place((x+rx, y+ry, rw, rh))
        except Exception:
            self.overlay.hide()

    def submit_capture(self):
        if self.busy:
            return
        self.busy = True
        job = CaptureJob(self.generation, self.provider, self.hwnd, copy.deepcopy(self.settings))
        job.signals.done.connect(self.captured)
        job.signals.failed.connect(self.failed)
        self.pool.start(job)

    def captured(self, token, frame, unused):
        self.busy = False
        if token != self.generation:
            return
        if self.frame is not None and (self.frame.client_size != frame.client_size
                or self.image is None or self.image.shape != frame.image.shape):
            # A resized board may look identical after thumbnail normalisation,
            # but saved rectangles/accessibility cannot be reused at a new size.
            self.invalidate(True)
        self.frame = frame
        self.image = frame.image
        if self.selecting_region:
            self.canvas.set_image(self.image)
            return
        if not self.running and not self.settings.templates:
            guessed = infer_size(self.image)
            if guessed:
                self.settings.tile_width, self.settings.tile_height = guessed
        gate_image = self.image
        if self.running and self.result and self.last_debug and self.last_debug[1] is self.result:
            reference = self.last_debug[0]
            if reference is not None and selection_highlight(reference, self.image, self.result.tiles,
                                                             self.settings.ignored_regions):
                gate_image = reference
        status = self.gate.push(gate_image)
        if not self.running:
            self.canvas.set_image(self.image)
            self.submit_analysis()
        elif status == 'moving':
            self.canvas.notice.phase = 'idle'
            self.validated_frame = False
            self.update_pair_preview()
            if not self.preview_deadline:
                self.overlay.result = self.canvas.result = None
            self.result = None
            self.table.setRowCount(0)
            self.canvas.set_image(self.image)
            self.status.setText('盤面の安定を待っています。')
            self.pair_banner.setText('ペア線：両方の牌が残っている直前の候補を一時表示中。' if self.preview_deadline else 'ペア線：盤面を確認中。')
        elif status == 'ready':
            if self.preview_deadline:
                self.update_pair_preview()
            self.validated_frame = True
            self.canvas.set_image(self.image)
            self.submit_analysis()
        elif status == 'unchanged':
            if self.result is not None:
                self.canvas.notice.phase = None
                self.overlay.result = self.result
            self.validated_frame = True
            self.canvas.set_image(self.image)

    def recognize(self):
        self.invalidate(True)
        if self.source == 'live':
            if self.hwnd:
                self.submit_capture()
        elif self.image is not None:
            self.submit_analysis()

    def submit_analysis(self):
        if self.busy or self.image is None:
            return
        self.busy = True
        job = AnalyzeJob(self.generation, self.image.copy(), copy.deepcopy(self.settings), self.base)
        self.set_notice_phase(self.generation, 'recognizing')
        job.signals.searching.connect(self.pair_search_started)
        job.signals.done.connect(self.analyzed)
        job.signals.failed.connect(self.failed)
        self.pool.start(job)

    def pair_search_started(self, token):
        self.set_notice_phase(token, 'searching')

    def set_notice_phase(self, token, phase):
        if token == self.generation:
            self.canvas.notice.phase = phase
            self.canvas.update()
            self.overlay.update()
            if self.running:
                self.follow_target()

    def analyzed(self, token, image, result):
        self.busy = False
        if token != self.generation:
            return
        self.canvas.notice.phase = None
        self.preview_deadline = 0.0
        self.result = result
        if self.settings.auto_register:
            registered = {entry['id'] for entry in self.settings.templates}
            for tile in result.tiles:
                if not tile.pattern or not tile.pattern.startswith('自動-') or tile.pattern in registered:
                    continue
                x, y, w, h = tile.rect
                path = self.base/'templates'/(tile.pattern+'.png')
                path.parent.mkdir(exist_ok=True)
                write_image(path, image[y:y+h, x:x+w])
                self.settings.templates.append(dict(id=tile.pattern, group=tile.pattern, path='templates/'+path.name))
                registered.add(tile.pattern)
                self.settings.save(self.config_path)
        if result.face_size:
            self.settings.tile_width, self.settings.tile_height = result.face_size
        self.canvas.pair = 0
        self.show_result()

    def failed(self, token, message):
        self.busy = False
        if token == self.generation:
            self.invalidate(True)
            self.running = False
            self.status.setText('停止: ' + message)

    def show_result(self):
        if not self.result:
            return
        self.canvas.notice.phase = None
        self.canvas.notice.count = len(self.result.pairs)
        self.canvas.pair_count = self.overlay.pair_count = self.settings.pair_count
        self.overlay.pair = self.canvas.pair
        self.canvas.result = self.result
        self.last_debug = (self.canvas.image, self.result, copy.deepcopy(self.settings))
        self.pair_banner.setText('ペア線：' + pair_status(self.result))
        self.overlay.result = self.result if self.running else None
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.result.tiles))
        readonly = {0, 6, 10, 11, 12}
        for row, tile in enumerate(self.result.tiles):
            values = [tile.index, *tile.rect, tile.pattern or '?', f'{tile.confidence:.3f}',
                      f'{tile.board_x:.4f}', f'{tile.board_y:.4f}', tile.level if tile.level is not None else '?', tile.state, tile.reason, tile.visual.get('estimated_layer') if tile.visual.get('estimated_layer') is not None else '?']
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                if col in readonly:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        self.canvas.update()
        self.display_options()
        recognized = sum(t.pattern is not None for t in self.result.tiles)
        patterns = len({t['id'] for t in self.settings.templates})
        if self.result.certified:
            geometry_notice = '配置確認済み。候補の順番は最善手を意味しません。'
        elif any(t.access_confirmed for t in self.result.tiles):
            geometry_notice = '個別の空き確認を使用中。未確認の牌は保留し、画面変化で確認を解除します。'
        elif any(t.auto_reason for t in self.result.tiles):
            geometry_notice = '画像からペアを自動判定しています。縁・側面が曖昧な牌は保留し、絶対段数は断定しません。'
        else:
            geometry_notice = '取れる牌を個別に確認するか、段数・配置全体を修正して確認してください。'
        paired_tiles = len({index for pair in self.result.pairs for index in pair})
        self.status.setText(f'位置候補 {len(self.result.tiles)}件 / 柄を認識 {recognized}件 / 見本 {patterns}柄・{len(self.settings.templates)}枚 / 自由 {paired_tiles}枚（{len(self.result.pairs)}組のペア） / 認識 {self.result.elapsed_ms:.1f} ms。'
                            + geometry_notice)
        if not self.settings.templates:
            self.status.setText(self.status.text() + ' 柄の見本が未登録です。牌を選んで登録してください。')
        if self.canvas.selected_tile is not None:
            self.pick_tile(self.canvas.selected_tile)

    def pick_tile(self, index):
        if not self.result:
            return
        row = next((row for row, tile in enumerate(self.result.tiles) if tile.index == index), None)
        if row is None:
            return
        self.table.selectRow(row)
        self.table.scrollToItem(self.table.item(row, 0))
        self.table_selection_changed()

    def table_selection_changed(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not self.result or not rows or rows[0] >= len(self.result.tiles):
            self.canvas.selected_tile = None
            self.tile_preview.clear()
            self.canvas.update()
            return
        tile = self.result.tiles[rows[0]]
        self.canvas.selected_tile = tile.index
        x, y, w, h = (round(v) for v in tile.rect)
        crop = self.image[y:y+h, x:x+w]
        if crop.size:
            self.tile_preview.setPixmap(pixmap(crop).scaled(60, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        if not self.settings.templates:
            reason = '柄の見本が未登録です'
        elif tile.pattern is None:
            reason = f'柄不明: 最良スコア {tile.confidence:.3f} / 次点 {tile.runner_up:.3f}（しきい値 {self.settings.threshold:.3f}、必要差 {self.settings.ambiguity_margin:.3f}）'
        else:
            reason = f'柄 {tile.pattern} / スコア {tile.confidence:.3f}'
        if tile.access_confirmed:
            reason += ' / 空きを手動確認済み'
        self.selection_info.setText(f'選択: #{tile.index}　表面 {w}×{h}px\n{reason}\n隣の牌や隠れた部分が入っていないか、登録画面で確認してください。')
        self.canvas.update()

    def register_selected_template(self):
        if self.busy:
            raise ValueError('画像処理の完了後に登録してください')
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows or not self.result:
            raise ValueError('画像の牌をクリック、または表の行を選択してください')
        tiles = self.read_table()
        tile = tiles[rows[0]]
        x, y, w, h = (round(v) for v in tile.rect)
        self.result.tiles = tiles
        self.register_template(self.image[y:y+h, x:x+w], tile.pattern or '')

    def display_options(self):
        self.canvas.debug = not self.raw_box.isChecked()
        self.overlay.pair = self.canvas.pair
        self.canvas.update()
        self.overlay.update()

    def next_pair(self):
        if self.result and self.result.pairs:
            next_start = self.canvas.pair + self.settings.pair_count
            self.canvas.pair = next_start if next_start < len(self.result.pairs) else 0
            self.display_options()
            self.status.setText(f'ペア候補 {self.canvas.pair+1}〜{min(len(self.result.pairs), self.canvas.pair+self.settings.pair_count)} / {len(self.result.pairs)}組')

    def read_table(self):
        if not self.result:
            raise ValueError('認識結果がありません')
        tiles = []
        for row in range(self.table.rowCount()):
            value = lambda col: self.table.item(row, col).text().strip()
            rect = tuple(float(value(c)) for c in (1, 2, 3, 4))
            x, y, w, h = rect
            if min(x, y) < 0 or min(w, h) < 8 or x+w > self.image.shape[1]+1 or y+h > self.image.shape[0]+1:
                raise ValueError(f'{row+1}行: 牌の矩形が画像範囲外です')
            level = None if value(9) in ('', '?') else int(value(9))
            if level is not None and not 0 <= level <= 30:
                raise ValueError('段数は0〜30、または ? を指定してください')
            pattern = value(5)
            tile = Tile(int(value(0)), rect, None if pattern in ('', '?') else pattern,
                        float(value(6)), board_x=float(value(7)), board_y=float(value(8)), level=level)
            previous = next((t for t in self.result.tiles if t.index == tile.index), None)
            if previous:
                tile.runner_up = previous.runner_up
                tile.access_confirmed = previous.access_confirmed
                tile.auto_access = previous.auto_access
                tile.auto_reason = previous.auto_reason
                tile.visual = copy.deepcopy(previous.visual)
            if not all(__import__('math').isfinite(v) for v in (*rect, tile.board_x, tile.board_y)):
                raise ValueError('座標は有限の数値を指定してください')
            tiles.append(tile)
        return tiles

    def reproject(self):
        tiles = self.read_table()
        for tile in tiles:
            project(tile, self.settings)
        self.result.tiles = tiles
        evaluate(self.result, self.settings)
        self.overlay.hide()
        self.show_result()

    def certify(self):
        tiles = self.read_table()
        self.generation += 1  # reject any earlier recognition job
        self.result.tiles = tiles
        groups = Recognizer(self.settings, self.base).groups
        evaluate(self.result, self.settings, groups, certified=True)
        if self.source == 'live' and self.frame:
            self.running = True
            self.validated_frame = False
            self.gate.reset()
            for _ in range(self.settings.stable_frames):
                self.gate.push(self.image)
        self.show_result()

    def confirm_accessibility(self):
        self.set_accessibility(True)

    def clear_accessibility(self):
        self.set_accessibility(False)

    def set_accessibility(self, confirmed):
        if self.busy:
            raise ValueError('画像処理の完了後に操作してください')
        if not self.result:
            raise ValueError('認識結果がありません')
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            raise ValueError('画像の牌をクリック、または表の行を選択してください')
        self.generation += 1
        tiles = self.read_table()
        for row in rows:
            tiles[row].access_confirmed = confirmed
        self.result.tiles = tiles
        groups = Recognizer(self.settings, self.base).groups
        evaluate(self.result, self.settings, groups, certified=False)
        self.show_result()
        if self.source == 'live' and self.frame:
            self.start()

    def table_changed(self, item):
        self.canvas.notice.phase = 'idle'
        # A table edit invalidates published hints immediately, before Apply.
        self.running = False
        self.generation += 1
        self.overlay.hide()
        self.overlay.result = None
        self.validated_frame = False
        if self.result:
            for tile in self.result.tiles:
                tile.access_confirmed = False
                tile.auto_access, tile.auto_reason, tile.visual = 'unknown', '', {}
            self.result.pairs = []
            self.result.certified = False
            self.canvas.update()
            self.pair_banner.setText('ペア線：表を編集したため、前の空き確認を解除しました。修正後に再確認してください。')

    def delete_tile(self):
        if not self.result:
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
        self.result.tiles = self.read_table()
        evaluate(self.result, self.settings)
        self.overlay.hide()
        self.show_result()

    def selection(self, rect):
        self.safe(lambda: self.handle_selection(rect))

    def handle_selection(self, rect):
        if self.busy:
            raise ValueError('画像処理の完了後に操作してください')
        x, y, w, h = rect
        if not self.selecting_region:
            return
        self.canvas.region_drag = False
        if self.selecting_region in ('ignored_regions', 'stock_regions'):
            height, width = self.image.shape[:2]
            getattr(self.settings, self.selecting_region).append([x/width,y/height,w/width,h/height])
            self.selecting_region = False
            self.settings.save(self.config_path)
            self.reload_zones()
            self.recognize()
        else:
            self.pause()
            self.selecting_region = False
            height, width = self.image.shape[:2]
            if self.source == 'live':
                old_x, old_y, old_w, old_h = self.settings.roi
                self.settings.roi = [old_x+x/width*old_w, old_y+y/height*old_h,
                                     w/width*old_w, h/height*old_h]
                self.settings.save(self.config_path)
            self.image = self.image[y:y+h, x:x+w].copy()
            self.frame = None
            self.canvas.set_image(self.image)
            self.recognize()

    def add_template_file(self):
        path, _ = QFileDialog.getOpenFileName(self, '牌表面全体の見本', '', '画像 (*.png *.jpg *.jpeg *.bmp *.webp)')
        if path:
            self.register_template(read_image(path))

    def register_template(self, image, suggested_id=''):
        if self.busy:
            raise ValueError('画像処理の完了後に登録してください')
        self.pause()
        image = image.copy()
        dialog = QDialog(self)
        dialog.setWindowTitle('選んだ牌の柄を見本に登録')
        dialog.resize(520, 440)
        layout = QVBoxLayout(dialog)
        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setPixmap(pixmap(image).scaled(220, 240, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(preview)
        explanation = QLabel(f'切り出し範囲: {image.shape[1]}×{image.shape[0]}px\n牌の表面全体が入り、隣の牌や隠れた柄が混ざっていないことを確認してください。\n既存の柄IDを選ぶと、同じ柄の追加見本として登録します。')
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QFormLayout()
        layout.addLayout(form)
        ids = QComboBox()
        ids.setObjectName('template_pattern')
        ids.setEditable(True)
        ids.addItems(sorted({t['id'] for t in self.settings.templates}))
        ids.setCurrentIndex(-1)
        ids.setEditText(suggested_id)
        ids.lineEdit().setPlaceholderText('例: 南、二索、tile_01')
        group_field = QLineEdit()
        group_field.setObjectName('template_group')
        group_field.setPlaceholderText('通常は柄IDと同じ。花牌は flower など')
        form.addRow('柄名 / ID', ids)
        form.addRow('ペア判定グループ', group_field)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        save_button.setText('登録して柄を再認識')
        def update_group(text):
            pattern = text.strip()
            existing = next((t.get('group') or pattern for t in self.settings.templates if t['id'] == pattern), pattern)
            group_field.setText(existing)
            save_button.setEnabled(bool(pattern))
        ids.currentTextChanged.connect(update_group)
        update_group(suggested_id)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if not dialog.exec():
            return
        pattern = ids.currentText().strip()
        group = group_field.text().strip() or pattern
        if not pattern:
            return
        directory = self.base / 'templates'
        directory.mkdir(exist_ok=True)
        path = directory / (uuid.uuid4().hex + '.png')
        write_image(path, image)
        candidate = copy.deepcopy(self.settings)
        candidate.templates.append({'id': pattern, 'group': group, 'path': 'templates/'+path.name})
        recognizer = Recognizer(candidate, self.base)
        candidate.save(self.config_path)
        self.settings = candidate
        self.gate = StabilityGate(candidate)
        if self.result is not None and self.image is not None:
            # Keep all detected rectangles and manual geometry so users can
            # register several tiles in succession without losing their place.
            started = time.perf_counter()
            for tile in self.result.tiles:
                x, y, w, h = (round(v) for v in tile.rect)
                tile.pattern, tile.confidence, tile.runner_up = recognizer.classify(self.image[y:y+h, x:x+w])
            if candidate.auto_pairs and not candidate.flat_board:
                from .visual_access import infer_accessibility
                infer_accessibility(self.image, self.result.tiles, candidate)
            evaluate(self.result, candidate, recognizer.groups, certified=candidate.flat_board)
            self.result.elapsed_ms = (time.perf_counter()-started)*1000
            self.show_result()
            self.status.setText(f'見本「{pattern}」を保存し、柄を再認識しました。 ' + self.status.text())
        elif self.image is not None:
            self.submit_analysis()
        else:
            self.status.setText(f'見本「{pattern}」を保存しました。ゲーム画像を開いてください。')

    def guess_size(self):
        if self.image is None:
            return
        size = infer_size(self.image)
        if size is None:
            raise ValueError('明るい矩形を検出できません。設定でサイズを指定してください')
        self.settings.tile_width, self.settings.tile_height = size
        self.settings.save(self.config_path)
        self.status.setText(f'牌サイズの推定: {size[0]:.1f} × {size[1]:.1f}px。設定から修正できます。再認識してください。')
        self.invalidate(True)

    def debug_directory(self):
        root = Path(__file__).resolve().parents[2]
        return (root if self.base == root / 'json' else self.base) / 'debug'

    def quick_debug(self):
        from tempfile import TemporaryDirectory
        image = self.canvas.image
        if image is None:
            self.status.setText('保存する盤面画像がありません。先に画像を取得してください。')
            return
        if self.quick_folder is None:
            parent = self.debug_directory()
            parent.mkdir(parents=True, exist_ok=True)
            number = 1 + max((int(p.name) for p in parent.iterdir()
                              if p.name.isascii() and p.name.isdigit()), default=0)
            while True:
                folder = parent / str(number)
                try:
                    folder.mkdir()
                except FileExistsError:
                    number += 1
                else:
                    self.quick_folder = folder
                    break
        folder = self.quick_folder
        result = self.canvas.result
        preview = pixmap(image)
        painter = QPainter(preview)
        try:
            paint_hints(painter, result, self.canvas.pair, True,
                        labels=True, pair_count=self.canvas.pair_count)
        finally:
            painter.end()
        with TemporaryDirectory(dir=folder) as temporary:
            board = Path(temporary) / 'board.png'
            annotated = Path(temporary) / 'annotated.png'
            write_image(board, image)
            if not preview.save(str(annotated)):
                raise OSError('注釈画像を保存できませんでした')
            record = Path(temporary) / 'result.json'
            record.write_text(json.dumps(result.to_dict() if result is not None else None,
                                         ensure_ascii=False, indent=2), encoding='utf-8')
            sources = (board, annotated, record)
            while True:
                n = self.quick_number
                self.quick_number += 1
                targets = [folder / f'{src.stem}{n}{src.suffix}' for src in sources]
                if any(path.exists() for path in targets):
                    continue
                published = []
                try:
                    for src, target in zip(sources, targets):
                        src.rename(target)
                        published.append(target)
                except OSError:
                    for path in published:
                        path.unlink()
                    raise
                break
        names = '・'.join(path.name for path in targets)
        self.status.setText(f'保存しました：{folder}（{names}）')

    def save_debug(self):
        from dataclasses import asdict
        from datetime import datetime
        import shutil

        if self.last_debug is None:
            return
        image, result, settings = self.last_debug
        if image is None:
            return
        folder = self.debug_directory() / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        folder.mkdir(parents=True)
        write_image(folder/'board.png', image)
        preview = pixmap(image)
        painter = QPainter(preview)
        selected = self.canvas.pair if result is self.result else 0
        paint_hints(painter, result, selected, True, labels=True, pair_count=settings.pair_count)
        painter.end()
        if not preview.save(str(folder/'annotated.png')):
            raise OSError('判定後の盤面画像を保存できませんでした')
        data = asdict(settings)
        for index, entry in enumerate(data['templates']):
            source = self.base / entry['path']
            if source.is_file():
                entry['path'] = f'templates/{index}{source.suffix}'
                target = folder / entry['path']
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, target)
        display = dict(mode=self.source, running=self.running, selected_pair=selected,
                       overlay_visible=self.overlay.isVisible(), validated=self.validated_frame)
        for name, value in [('result', result.to_dict()), ('settings', data), ('display', display)]:
            (folder/(name+'.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def closeEvent(self, event):
        try:
            self.save_debug()
        except Exception as exc:
            QMessageBox.warning(self, 'デバッグ保存に失敗しました', str(exc))
        if self.settings.remember_size:
            size = self.normalGeometry().size() if self.isMinimized() else self.size()
            self.settings.last_size = [size.width(), size.height()]
            try:
                self.settings.save(self.config_path)
            except OSError as exc:
                QMessageBox.warning(self, 'ウィンドウサイズの保存に失敗しました', str(exc))
        self.running = False
        self.overlay.close()
        self.timer.stop()
        self.follow.stop()
        self.pool.waitForDone()
        super().closeEvent(event)
