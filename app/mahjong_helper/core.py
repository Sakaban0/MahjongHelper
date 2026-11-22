from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
import json
import math
import time
from itertools import combinations

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像を読み込めません: {path}")
    return image


def write_image(path, image):
    ok, data = cv2.imencode(Path(path).suffix or '.png', image)
    if not ok:
        raise ValueError('画像の保存に失敗しました')
    data.tofile(str(path))


@dataclass
class Settings:
    tile_width: float = 60
    tile_height: float = 80
    layer_dx: float = -6
    layer_dy: float = -6
    threshold: float = .78
    ambiguity_margin: float = .07
    interval_ms: int = 400
    stable_frames: int = 3
    change_threshold: float = 1.5
    surface_brightness: int = 155
    flat_board: bool = False
    auto_pairs: bool = True
    auto_register: bool = False
    use_bright_face_cue: bool = False
    roi: list = field(default_factory=lambda: [0., 0., 1., 1.])
    window_title: str = ''
    templates: list = field(default_factory=list)

    ignored_regions: list = field(default_factory=list)
    stock_regions: list = field(default_factory=list)
    imported_bundle_entries: list = field(default_factory=list)
    pair_count: int = 1
    initial_width: int = 1250
    initial_height: int = 900
    remember_size: bool = False
    last_size: list = field(default_factory=list)

    @classmethod
    def load(cls, path):
        if not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        obj = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        obj.validate()
        return obj

    def validate(self):
        for region in self.ignored_regions + self.stock_regions:
            if len(region) != 4 or not all(math.isfinite(n) for n in region):
                raise ValueError('領域の座標が不正です')
            x,y,w,h = region
            if min(x,y)<0 or min(w,h)<=0 or max(x+w,y+h)>1.00001:
                raise ValueError('領域は盤面画像内で指定してください')
        if type(self.pair_count) is not int or not 1 <= self.pair_count <= 99:
            raise ValueError("同時表示するペア数は1〜99組で指定してください")
        if not all(math.isfinite(v) for v in (self.tile_width, self.tile_height, self.layer_dx,
                   self.layer_dy, self.change_threshold, *self.roi)):
            raise ValueError('設定には有限の数値を指定してください')
        if self.tile_width < 8 or self.tile_height < 8:
            raise ValueError('牌の幅・高さは8px以上にしてください')
        if not (0 <= self.threshold <= 1 and 0 <= self.ambiguity_margin <= 1):
            raise ValueError('照合しきい値と候補差は0〜1です')
        if self.interval_ms < 100 or self.stable_frames < 2:
            raise ValueError('確認間隔100ms以上、安定フレーム数2以上が必要です')
        if not 0 <= self.surface_brightness <= 255 or self.change_threshold < 0:
            raise ValueError('表面の明るさは0〜255、変化しきい値は0以上です')
        if len(self.roi) != 4:
            raise ValueError('盤面領域は [左, 上, 幅, 高さ] です')
        x, y, w, h = self.roi
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.00001 or y + h > 1.00001:
            raise ValueError('盤面領域はウィンドウ内の0〜1の比率で指定してください')

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding='utf-8')


@dataclass
class Tile:
    index: int
    rect: tuple
    pattern: str | None = None
    confidence: float = 0
    runner_up: float = 0
    board_x: float = 0
    board_y: float = 0
    level: int | None = None
    state: str = '判定不明'
    reason: str = '上下関係未確認'
    above: list = field(default_factory=list)
    access_confirmed: bool = False
    auto_access: str = 'unknown'
    auto_reason: str = ''
    visual: dict = field(default_factory=dict)


@dataclass
class Result:
    tiles: list[Tile]
    pairs: list[tuple[int, int]]
    elapsed_ms: float = 0
    certified: bool = False
    face_size: tuple = ()

    def to_dict(self):
        return asdict(self)


def intersection(a, b):
    x, y, w, h = a
    u, v, s, t = b
    return max(0., min(x + w, u + s) - max(x, u)) * max(0., min(y + h, v + t) - max(y, v))


def suppress(rects, expected_size=None):
    """Prefer full surfaces, discard nested/near-identical contour candidates."""
    kept = []
    key = (lambda r: abs(r[2]/expected_size[0]-1)+abs(r[3]/expected_size[1]-1)) if expected_size else (lambda r: -r[2]*r[3])
    for rect in sorted(rects, key=key):
        if all(intersection(rect, r) / min(rect[2] * rect[3], r[2] * r[3]) < .55 for r in kept):
            kept.append(rect)
    return sorted(kept, key=lambda r: (round(r[1] / 4), r[0]))


def surfaces(image, settings):
    """Stage 1: bright face segmentation, plus closed edge contours.

    Thickness/partly occluded faces cannot be reliably reconstructed here.
    Partial plausible rectangles remain unknown; no hidden tiles are invented.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, settings.surface_brightness, 255)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    candidates = []
    # A single low threshold joins neighbouring light faces through bevels.
    # Keep additional unclosed high-threshold masks: thin seams must survive.
    masks = [bright, edges]
    masks.extend(cv2.inRange(gray, cutoff, 255) for cutoff in (180, 200, 220, 235)
                 if cutoff > settings.surface_brightness)
    if np.mean(gray >= 250) > .035:
        masks.append(cv2.inRange(gray, 248, 255))
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / h
            expected = settings.tile_width / settings.tile_height
            if not (.48 * settings.tile_width <= w <= 1.6 * settings.tile_width
                    and .48 * settings.tile_height <= h <= 1.6 * settings.tile_height):
                continue
            if not .65 * expected <= ratio <= 1.4 * expected:
                continue
            fill = cv2.contourArea(contour) / (w * h)
            if fill >= .72:
                candidates.append((x, y, w, h))
    return suppress(candidates, (settings.tile_width, settings.tile_height))


def infer_size(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = [cv2.inRange(gray, cutoff, 255) for cutoff in (160, 180, 200, 220, 235)]
    edges = cv2.Canny(gray, 60, 160)
    masks.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))
    boxes = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            r = cv2.boundingRect(contour)
            if (r[2] >= 15 and r[3] >= 20 and .45 < r[2]/r[3] < 1.1
                    and r[2]*r[3] < image.shape[0]*image.shape[1]*.12
                    and cv2.contourArea(contour)/(r[2]*r[3]) >= .78):
                boxes.append(r)
    if not boxes:
        return None
    # Cluster by relative size, avoiding rounding-bin boundaries. Count distinct
    # locations so repeated thresholds don't outvote many real tile faces.
    clusters = [[r for r in boxes if abs(r[2]/seed[2]-1)<.12 and abs(r[3]/seed[3]-1)<.12] for seed in boxes]
    cluster = max(clusters, key=lambda group: len({(round(r[0]/5), round(r[1]/5)) for r in group}))
    return tuple(float(np.median([r[i] for r in cluster])) for i in (2, 3))


def feature(image):
    # Discard border/selection tint. Gradients retain pattern strokes while
    # reducing uniform illumination and face-colour changes.
    h, w = image.shape[:2]
    scale = min(72/w, 96/h)
    resized = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_LINEAR)
    padded = np.full((96, 72, 3), np.median(image.reshape(-1, 3), axis=0), dtype=np.uint8)
    rh, rw = resized.shape[:2]
    padded[(96-rh)//2:(96-rh)//2+rh, (72-rw)//2:(72-rw)//2+rw] = resized
    image = padded
    h, w = image.shape[:2]
    image = image[max(1, int(h * .12)):int(h * .88), max(1, int(w * .12)):int(w * .88)]
    if image.size == 0:
        raise ValueError('見本画像が小さすぎます')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(cv2.GaussianBlur(gray, (5, 5), 1.3), cv2.CV_32F)


def ink_layout(face):
    # Record strong red/green/blue ink by position. Small translations and
    # antialiasing should not change the dominant colour in a large cell.
    hsv = cv2.cvtColor(cv2.resize(face, (60, 72)), cv2.COLOR_BGR2HSV)
    hue, sat, value = cv2.split(hsv[6:-6, 6:-6])
    masks = [(hue < 13) | (hue > 167), (hue >= 34) & (hue <= 90),
             (hue >= 95) & (hue <= 138)]
    return np.stack([(mask & (sat > 75) & (value > 45)).reshape(3,20,3,16).mean(axis=(1,3))
                     for mask in masks])


def compatible_ink(a, b):
    # Veto a substantial coloured mark missing from the corresponding cell.
    return not np.any((np.maximum(a,b) > .16) & (np.minimum(a,b) < .025))


def plain_face(face):
    h, w = face.shape[:2]
    if min(w, h) < 12:
        return False
    # Ignore the bevel only. Even a small printed mark rules out a blank.
    dx, dy = max(1, round(w*.04)), max(1, round(h*.04))
    inside = face[dy:-dy, dx:-dx].astype(np.int16)
    tone = np.median(inside, axis=(0, 1))
    difference = np.abs(inside-tone).max(axis=2)
    return bool(tone.min() >= 165 and np.ptp(tone) <= 32
                and np.percentile(difference, 98) <= 7 and difference.max() <= 20)


def ink_signature(face):
    # Align by printed ink rather than by the (theme-dependent) bevel.
    h,w = face.shape[:2]
    centre = face[round(.10*h):round(.90*h), round(.09*w):round(.91*w)]
    paper = np.percentile(centre, 90, axis=(0,1))
    pigment = np.maximum(0, paper-centre)
    mask = pigment.max(axis=2) > 62
    ys,xs = np.where(mask)
    if len(xs)<30 or np.ptp(xs)<.25*w or np.ptp(ys)<.25*h:
        return None
    pigment[~mask] = 0
    ink = pigment[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(np.float32)
    ih,iw = ink.shape[:2]
    scale = min(34/iw,44/ih)
    ink = cv2.resize(ink,(max(1,round(iw*scale)),max(1,round(ih*scale))),interpolation=cv2.INTER_AREA)
    ih,iw = ink.shape[:2]
    patch = np.zeros((48,38,3),np.float32)
    y,x = (48-ih)//2,(38-iw)//2
    patch[y:y+ih,x:x+iw] = ink
    patch = cv2.GaussianBlur(patch,(3,3),.75)
    return patch / max(1.,np.linalg.norm(patch))


def atlas_face(face):
    h, w = face.shape[:2]
    scale = min(64/w, 88/h)
    sw, sh = max(1, round(w*scale)), max(1, round(h*scale))
    canvas = np.full((88,64,3), np.median(face.reshape(-1,3), axis=0), np.uint8)
    x, y = (64-sw)//2, (88-sh)//2
    canvas[y:y+sh,x:x+sw] = cv2.resize(face,(sw,sh),interpolation=cv2.INTER_AREA)
    return canvas


@lru_cache(maxsize=1)
def bundled_patterns():
    folder = Path(__file__).resolve().parent / 'visible_refs'
    names = json.loads((folder/'tiles.json').read_text(encoding='utf8'))
    sheet = read_image(folder/'tiles.png')
    entries = []
    for index, name in enumerate(names):
        signature = ink_signature(sheet[88*index:88*(index+1)])
        if signature is not None:
            entries.append((name, signature.ravel()))
    return [name for name, _ in entries], np.stack([v for _, v in entries])


class Recognizer:
    def __init__(self, settings, base_dir):
        self.settings = settings
        self.examples = []
        self.blank_ids = set()
        self.groups = {}
        for item in settings.templates:
            path = Path(item['path'])
            if not path.is_absolute():
                path = Path(base_dir) / path
            sample = read_image(path)
            if plain_face(sample):
                self.blank_ids.add(item['id'])
            self.examples.append((item['id'], feature(sample), ink_layout(sample)))
            group = item.get('group') or item['id']
            if item['id'] in self.groups and self.groups[item['id']] != group:
                raise ValueError(f"柄ID {item['id']} のグループ指定が一致していません")
            self.groups[item['id']] = group
        self.has_manual_patterns = any(not key.startswith('自動-') for key in self.groups)

    def named_pattern(self, crop):
        signature = ink_signature(atlas_face(crop))
        if signature is None:
            return None
        names, matrix = bundled_patterns()
        similarities = np.einsum('ij,j->i', matrix, signature.ravel())
        best = int(similarities.argmax())
        name, value = names[best], float(similarities[best])
        rival = max(float(v) for n, v in zip(names, similarities) if n != name)
        if value >= max(.948, self.settings.threshold) and value-rival >= max(.023, self.settings.ambiguity_margin*.4):
            return name, min(value, 1.), rival
        return None

    def classify(self, crop):
        named_match = self.named_pattern(crop) if self.settings.auto_register else None
        if named_match and not self.has_manual_patterns:
            return named_match
        target = feature(crop)
        # Face bounds vary by a few pixels with bevels/highlights. Search small
        # translations instead of requiring the glyph to land on identical pixels.
        search = cv2.copyMakeBorder(target, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
        scores = {}
        layout = ink_layout(crop)
        for pattern, template, sample_layout in self.examples:
            if not compatible_ink(layout, sample_layout):
                continue
            best = -1.
            # Preserve example aspect ratio; compare at several nearby sizes.
            scale = min(target.shape[1] / template.shape[1], target.shape[0] / template.shape[0])
            for relative in (.88, .94, 1., 1.06, 1.12):
                size = (max(2, round(template.shape[1] * scale * relative)),
                        max(2, round(template.shape[0] * scale * relative)))
                if size[0] > search.shape[1] or size[1] > search.shape[0]:
                    continue
                sample = cv2.resize(template, size, interpolation=cv2.INTER_AREA)
                if sample.std() < .5 or target.std() < .5:
                    continue
                score = float(cv2.minMaxLoc(cv2.matchTemplate(search, sample, cv2.TM_CCOEFF_NORMED))[1])
                if np.isfinite(score):
                    best = max(best, score)
            scores[pattern] = max(scores.get(pattern, -1), best)
        ranked = sorted(scores.items(), key=lambda p: (-p[1], p[0]))
        if not ranked:
            return named_match or (None, 0., 0.)
        pattern, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.
        if score < self.settings.threshold or score - second < self.settings.ambiguity_margin:
            pattern = None
        if named_match and (pattern is None or pattern.startswith('自動-')):
            return named_match
        return pattern, max(0., score), max(0., second)

    def analyze(self, image):
        settings = self.settings
        if not (settings.ignored_regions or settings.stock_regions):
            return self._analyze(image)
        started = time.perf_counter()
        height, width = image.shape[:2]

        def region_mask(regions):
            mask = np.zeros((height, width), dtype=bool)
            for area in region_slices(regions, width, height):
                mask[area] = True
            return mask

        ignored = region_mask(settings.ignored_regions)
        stock = region_mask(settings.stock_regions) & ~ignored
        board = image.copy()
        board[ignored] = 0

        def inside(tile, mask):
            x,y,w,h = tile.rect
            return mask[min(height-1,int(y+h/2)),min(width-1,int(x+w/2))]

        held = []
        if stock.any():
            # Keep surrounding artwork available to the theme recognizer.
            held = [t for t in self._analyze(board).tiles if inside(t, stock)]
            # Stock slots need not share the board grid. Resolve anonymous
            # labels from the face itself before comparing them to board tiles.
            from .alternate import identify, match, set_group
            theme = identify(board)
            for tile in held:
                if tile.pattern and not tile.pattern.startswith('自動-'):
                    continue
                matched = None
                if theme == 'jade':
                    from .jade import classify
                    x, y, w, h = tile.rect
                    matched, _ = classify(board[y:y+h, x:x+w])
                elif theme in ('green', 'blue', 'teal'):
                    matched = match(board, tile.rect, theme)
                if matched:
                    tile.pattern, tile.confidence, tile.runner_up = matched
                    set_group(tile)
            board[stock] = 0
        result = self._analyze(board)
        excluded = ignored | stock
        result.tiles = [t for t in result.tiles if not inside(t, excluded)]
        for tile in held:
            tile.visual['stock'] = True
        result.tiles.extend(held)
        for index, tile in enumerate(result.tiles, 1):
            tile.index = index
        evaluate(result, settings, self.groups, certified=result.certified)
        result.elapsed_ms = (time.perf_counter()-started)*1000
        return result

    def _analyze(self, image):
        start = time.perf_counter()
        tiles = []
        settings = self.settings
        rects = surfaces(image, settings)
        if settings.auto_pairs and not settings.flat_board:
            def full_faces(rectangles, width, height):
                return sum(.91*width < w < 1.09*width and .91*height < h < 1.09*height
                           for _, _, w, h in rectangles)

            support = full_faces(rects, settings.tile_width, settings.tile_height)
            if support < max(4, len(rects)*.2):
                estimate = infer_size(image)
                if estimate:
                    from dataclasses import replace
                    candidate = replace(settings, tile_width=estimate[0], tile_height=estimate[1])
                    alternatives = surfaces(image, candidate)
                    if full_faces(alternatives, *estimate) >= max(10, 3*support, len(alternatives)*.65):
                        settings, rects = candidate, alternatives
        # A browser zoom change scales both axes together. Require several
        # independently detected faces before overriding the saved calibration.
        if settings.auto_pairs and not settings.flat_board and len(rects) >= 5:
            widths, heights = zip(*[(w, h) for _, _, w, h in rects])
            w, h = float(np.median(widths)), float(np.median(heights))
            sx, sy = w/settings.tile_width, h/settings.tile_height
            agreement = sum(abs(a/w-1)<.1 and abs(b/h-1)<.1 for a,b in zip(widths, heights))
            # New artwork can change the aspect ratio as well as the scale.
            # Require independent contour agreement for a change of proportions.
            changed_shape = False
            if max(abs(sx-1),abs(sy-1)) > .10:
                estimate = infer_size(image)
                changed_shape = estimate is not None and abs(estimate[0]/w-1)<.06 and abs(estimate[1]/h-1)<.06
                if estimate and len(rects)>=25:
                    def supporting(size):
                        return sum(abs(a/size[0]-1)<.10 and abs(b/size[1]-1)<.10 for a,b in zip(widths,heights))
                    votes = supporting(estimate)
                    if votes >= max(10, 2*supporting((settings.tile_width,settings.tile_height))):
                        w,h = estimate
                        changed_shape = True
                        agreement = len(rects)
            if (changed_shape or (abs(sx/sy-1)<.12 and abs(sx-1)>.15)) and agreement >= max(5, len(rects)*.65):
                from dataclasses import replace
                settings = replace(settings, tile_width=w, tile_height=h)
                rects = surfaces(image, settings)
        from .alternate import identify, blue_size, read_board
        theme = identify(image) if settings.auto_pairs and settings.auto_register and not settings.flat_board else None
        if theme:
            size = blue_size(image) if theme == 'blue' else infer_size(image)
            if size and theme in ('green', 'teal') and not .69 < size[0]/size[1] < .89:
                size = None
            if size and max(abs(size[0]/settings.tile_width-1), abs(size[1]/settings.tile_height-1))>.08:
                from dataclasses import replace
                settings = replace(settings, tile_width=size[0], tile_height=size[1])
                rects = surfaces(image, settings)
        for index, rect in enumerate(rects, 1):
            x, y, w, h = rect
            pattern, score, second = self.classify(image[y:y+h, x:x+w])
            tile = Tile(index, rect, pattern, score, second,
                        x / settings.tile_width, y / settings.tile_height)
            # Local surface/side evidence does not determine an absolute level.
            tile.level = 0 if self.settings.flat_board else None
            tiles.append(tile)
        for a, b in combinations(tiles, 2):
            if intersection(a.rect, b.rect) > min(a.rect[2]*a.rect[3], b.rect[2]*b.rect[3]) * .035:
                a.level = b.level = None
        if self.settings.flat_board:
            # A user-declared rendering rule, never inferred from rectangle overlap.
            for tile in tiles:
                tile.level = 0
        for tile in tiles:
            w, h = tile.rect[2:]
            if not (.85 <= w/self.settings.tile_width <= 1.15 and .85 <= h/self.settings.tile_height <= 1.15):
                tile.level = None  # merged/partial faces need manual separation
        result = Result(tiles, [], face_size=(settings.tile_width, settings.tile_height))
        if self.settings.auto_pairs and not self.settings.flat_board:
            from .visual_access import infer_accessibility
            infer_accessibility(image, tiles, settings)
        for tile in tiles:
            x, y, w, h = tile.rect
            if not (.94 <= w/settings.tile_width <= 1.07 and .94 <= h/settings.tile_height <= 1.07):
                continue
            if not plain_face(image[y:y+h, x:x+w]):
                continue
            if any(intersection(tile.rect, other.rect) > .025*w*h for other in tiles if other is not tile):
                continue
            # Uniform pixels alone are not a tile. Require contrast against the
            # immediate exterior on at least three sides of this detected face.
            if x < 2 or y < 2 or x+w+2 > image.shape[1] or y+h+2 > image.shape[0]:
                continue
            tone = np.median(image[y+2:y+h-2,x+2:x+w-2], axis=(0,1))
            borders = (image[y-2,x+2:x+w-2], image[y+h+1,x+2:x+w-2],
                       image[y+2:y+h-2,x-2], image[y+2:y+h-2,x+w+1])
            if sum(np.median(np.abs(edge-tone).max(axis=1)) > 10 for edge in borders) >= 3:
                if len(self.blank_ids) <= 1:
                    tile.pattern = next(iter(self.blank_ids), '白')
                    tile.confidence = float(np.mean(np.abs(image[y+2:y+h-2,x+2:x+w-2]-tone).max(axis=2)<=7))
                    tile.visual['blank_face'] = True
        if settings.auto_register:
            self.discover_patterns(image, tiles, settings)
        ivory = False
        if settings.auto_pairs and not settings.flat_board:
            from .ivory import apply_board
            ivory = read_board(image, result, settings, theme) if theme else apply_board(image, result, settings)
        evaluate(result, self.settings, self.groups, certified=self.settings.flat_board)
        if not settings.flat_board:
            self.find_visible_fragments(image, result, settings)
        if not settings.flat_board and not ivory:
            from .layers import estimate_layers
            estimate_layers(result.tiles, settings)
            prioritize_pairs(result)
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result

    def find_visible_fragments(self, image, result, settings):
        # A clipped corner can drop contour fill below the full-face threshold.
        # Recover only near-full-size contours with a distinctive visible mark.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(cv2.inRange(gray, 220, 255), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        refs = Path(__file__).resolve().parent/'visible_refs'
        marks = [(name, cv2.cvtColor(read_image(refs/file), cv2.COLOR_BGR2GRAY))
                 for name, file in [('北','north.png'), ('梅','plum.png')] if (refs/file).exists()]
        for contour in contours:
            rect = cv2.boundingRect(contour)
            x,y,w,h = rect
            if not (.85 < w/settings.tile_width < 1.2 and .85 < h/settings.tile_height < 1.2):
                continue
            if cv2.contourArea(contour)/(w*h) < .48:
                continue
            if any(intersection(rect,t.rect) > .65*min(w*h,t.rect[2]*t.rect[3]) for t in result.tiles):
                continue
            crop = cv2.resize(gray[y:y+h,x:x+w], (69,101))
            scores = []
            for name, mark in marks:
                search = cv2.copyMakeBorder(crop, 3,3,3,3, cv2.BORDER_REPLICATE)
                score = max(float(cv2.matchTemplate(search[:, :68], cv2.resize(mark,None,fx=k,fy=k),
                            cv2.TM_CCOEFF_NORMED).max()) for k in (.94,1.,1.04))
                scores.append((score,name))
            scores.sort(reverse=True)
            if not scores or scores[0][0]<.86 or (len(scores)>1 and scores[0][0]-scores[1][0]<.10):
                continue
            score,name = scores[0]
            result.tiles.append(Tile(len(result.tiles)+1,rect,name,score,
                board_x=x/settings.tile_width,board_y=y/settings.tile_height,
                reason='見えている柄の部分照合。側面・重なりは未確認',
                auto_reason='部分照合のため空き判定を保留'))

    def discover_patterns(self, image, tiles, settings):
        import hashlib
        samples = []
        for tile in tiles:
            x, y, w, h = tile.rect
            if tile.pattern or tile.confidence >= settings.threshold-settings.ambiguity_margin:
                continue
            if not (.80 <= w/settings.tile_width <= 1.15 and .80 <= h/settings.tile_height <= 1.15):
                continue
            crop = image[y:y+h, x:x+w]
            feature_map = feature(crop)
            if feature_map.std() < 2:
                continue
            samples.append((tile, feature_map, crop, ink_signature(crop)))
        for i, (tile, target, crop, ink) in enumerate(samples):
            for other, template, other_crop, other_ink in samples[:i]:
                if intersection(tile.rect, other.rect) or tile.pattern:
                    continue
                # Match most of the glyph, allowing only small translations.
                search = cv2.copyMakeBorder(target, 8, 8, 8, 8, cv2.BORDER_CONSTANT)
                score = max(float(cv2.matchTemplate(search, cv2.resize(template,
                    (round(template.shape[1]*k), round(template.shape[0]*k)),
                    interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED).max()) for k in (.88,.94,1.,1.06,1.12))
                score = max(score, float(cv2.matchTemplate(target, template[3:-3,3:-3], cv2.TM_CCOEFF_NORMED).max()))
                if ink is not None and other_ink is not None:
                    similarity = float(np.sum(ink*other_ink))
                    if similarity >= .966:
                        score = max(score,similarity)
                if score < .92 or not compatible_ink(ink_layout(crop), ink_layout(other_crop)):
                    continue
                # Check colour as well as stroke shape (red/green variants).
                def ink_hist(face):
                    h, w = face.shape[:2]
                    face = face[h//9:h-h//9, w//9:w-w//9]
                    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
                    return cv2.normalize(cv2.calcHist([hsv], [0], ((hsv[:,:,1]>85)*255).astype(np.uint8), [18], [0,180]), None).ravel()
                if np.linalg.norm(ink_hist(crop)-ink_hist(other_crop)) > .35:
                    continue
                name = other.pattern or '自動-' + hashlib.sha256(other_crop.tobytes()).hexdigest()[:10]
                other.pattern = tile.pattern = name
                other.confidence = tile.confidence = score


def evaluate(result, settings, groups=None, certified=False):
    """Board units use nominal face width/height; half-row offsets work naturally.

    Certification means the user checked face bounds, levels, and all visible
    blockers in this snapshot. A missing blocker can never be disproved from
    bright contour detection alone.
    """
    groups = groups or {}
    result.certified = certified
    result.pairs = []
    tiles = result.tiles
    for tile in tiles:
        tile.state, tile.reason, tile.above = '判定不明', '配置全体が未確認', []
        # Direct human observation of accessibility is independent of numeric
        # layer reconstruction, and is valid only for this unchanged snapshot.
        if tile.visual.get('stock'):
            tile.state, tile.reason = '取れる', '保有マスの牌'
            continue
        if tile.access_confirmed and not certified:
            if tile.pattern is None:
                tile.reason = '空きは手動確認済みですが、柄不明です'
            else:
                tile.state, tile.reason = '取れる', '空きを手動確認済み（この画像のみ）'
            continue
        if settings.auto_pairs and not certified and tile.auto_reason:
            if tile.auto_access == 'free' and tile.pattern is not None:
                tile.state, tile.reason = '取れる', '自動判定: ' + tile.auto_reason
            elif tile.auto_access == 'free':
                tile.reason = '自動で空きを検出しましたが、柄不明です'
            else:
                tile.reason = tile.auto_reason
            continue
        if tile.level is None:
            tile.reason = '段数・上下関係が不明'
            continue
        if not certified:
            continue
        x, y = tile.board_x, tile.board_y
        left = right = False
        ambiguous = False
        for other in tiles:
            if other is tile or other.visual.get('stock'):
                continue
            # An unknown-height face near this tile can be a blocker after
            # projection at an unknown level. Conservatively withhold globally.
            if other.level is None:
                ambiguous = True
                continue
            dx, dy = other.board_x - x, other.board_y - y
            overlap_x = min(x+1, other.board_x+1) - max(x, other.board_x)
            overlap_y = min(y+1, other.board_y+1) - max(y, other.board_y)
            if other.level > tile.level and overlap_x > .025 and overlap_y > .025:
                tile.above.append(other.index)
            if other.level == tile.level:
                if overlap_x > .08 and overlap_y > .08:
                    ambiguous = True  # impossible/conflicting same-level faces
                if overlap_y > .025:
                    left |= -1.08 <= dx <= -.85
                    right |= .85 <= dx <= 1.08
                    if .08 < abs(dx) < .85:
                        ambiguous = True
        if tile.above:
            tile.state, tile.reason = '取れない', f'上の牌: {tile.above}'
        elif left and right:
            tile.state, tile.reason = '取れない', '同じ段の左右が塞がっています'
        elif ambiguous:
            tile.reason = '不明段数または配置の矛盾があります'
        elif tile.pattern is None:
            tile.reason = '柄不明（しきい値未満または候補が拮抗）'
        else:
            tile.state, tile.reason = '取れる', '上が空き、左または右が空いています'
    free = sorted((t for t in tiles if t.state == '取れる' and t.pattern is not None), key=lambda t: (t.board_y, t.board_x, t.index))
    for a, b in combinations(free, 2):
        if groups.get(a.pattern, a.visual.get('match_group', a.pattern)) == groups.get(b.pattern, b.visual.get('match_group', b.pattern)):
            result.pairs.append((a.index, b.index))

    prioritize_pairs(result)


def prioritize_pairs(result):
    """Prefer the highest removable tile, then its partner. Keep ties stable."""
    if len(result.pairs) < 2:
        return
    stock_ids = {t.index for t in result.tiles if t.visual.get("stock")}
    levels = {}
    for tile in result.tiles:
        level = tile.level if tile.level is not None else tile.visual.get('estimated_layer')
        levels[tile.index] = level if level is not None else -1

    def priority(pair):
        first, second = (levels[index] for index in pair)
        return sum(i in stock_ids for i in pair), max(first, second), min(first, second)

    result.pairs.sort(key=priority, reverse=True)


def pair_status(result):
    """Explain absence of lines without presenting withheld hints as no moves."""
    if result is None:
        return '画像を読み込み、認識してください。'
    if not result.tiles:
        return '牌の位置候補がありません。盤面範囲と牌サイズを確認してください。'
    if result.pairs:
        source = ('確認した配置' if result.certified else
                  ('画像から空きを自動検出した牌' if any(t.auto_access == 'free' and not t.access_confirmed for t in result.tiles if t.state == '取れる') else '空きを手動確認した牌'))
        return f'{source}から {len(result.pairs)} 組を表示できます。「次のペア」で切り替えます。'
    if not any(t.pattern is not None for t in result.tiles):
        return '線を保留：柄を認識できていません。牌を選んで見本を登録してください。'
    checked = sum(t.access_confirmed for t in result.tiles)
    if any(t.auto_reason for t in result.tiles) and not result.certified and not checked:
        count = sum(t.state == '取れる' for t in result.tiles)
        return f'自動判定中：取れる牌 {count} 枚。同グループの2枚が揃っていないか、表面・側面が曖昧なため保留しています。'
    if not result.certified and not checked:
        return '線を保留：柄は認識していますが、取れるか未確認です。同柄の2枚をそれぞれ選び「取れると確認」を押してください。'
    if not result.certified:
        return f'空きを手動確認した {checked} 枚の中に、柄が確定した同グループの2枚がありません。他の未確認牌は判定を保留しています。'
    if any(t.state == '判定不明' for t in result.tiles):
        return '線を保留：段数・配置の矛盾・柄不明があります。表の「理由」を確認するか、取れる牌を個別に確認してください。'
    return '確認した配置には、同グループの取れる2枚がありません。'


def project(tile, settings):
    if tile.level is not None:
        tile.board_x = (tile.rect[0] - tile.level * settings.layer_dx) / settings.tile_width
        tile.board_y = (tile.rect[1] - tile.level * settings.layer_dy) / settings.tile_height


def region_slices(regions, width, height):
    for x, y, w, h in regions:
        yield (slice(round(y*height), round((y+h)*height)),
               slice(round(x*width), round((x+w)*width)))


def same_face_detail(old, new):
    """Compare interior ink without counting the selection border."""
    if not old.size or old.shape != new.shape:
        return False
    inset = max(2, round(min(old.shape[:2])*.12))
    a, b = old[inset:-inset,inset:-inset], new[inset:-inset,inset:-inset]
    if not a.size:
        return False
    if np.max(cv2.absdiff(a,b)) <= 5:
        return True
    a, b = (cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) for face in (a,b))
    return min(a.std(), b.std()) >= 10 and cv2.matchTemplate(b,a,cv2.TM_CCOEFF_NORMED)[0,0] >= .985


def selection_highlight(reference, image, tiles, ignored_regions):
    """Accept a local tint or outline only while the original face detail remains."""
    if reference.shape != image.shape:
        return False
    height, width = image.shape[:2]
    delta = cv2.absdiff(reference, image).max(axis=2)
    for area in region_slices(ignored_regions, width, height):
        delta[area] = 0
    bounds = cv2.boundingRect((delta > 6).astype(np.uint8))
    left, top, dw, dh = bounds
    if not dw or not dh:
        return False
    for tile in tiles:
        x, y, w, h = map(round, tile.rect)
        pad = max(2, round(min(w,h)*.06))
        if left < x-pad or top < y-pad or left+dw > x+w+pad or top+dh > y+h+pad:
            continue
        roi = np.s_[max(0,y):y+h, max(0,x):x+w]
        if same_face_detail(reference[roi], image[roi]):
            return True
    return False


class StabilityGate:
    """No history of hidden tiles. Changed frames immediately invalidate hints."""
    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.anchor = self.accepted = None
        self.count = 0

    def push(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        for area in region_slices(self.settings.ignored_regions, width, height):
            gray[area] = 0
        small = cv2.resize(gray, (192, 128))
        def same(a, b):
            if a is None:
                return False
            delta = cv2.absdiff(a, b)
            return (float(delta.mean()) <= self.settings.change_threshold
                    and float(np.mean(delta > 12)) < .0004)
        if same(self.accepted, small):
            self.anchor, self.count = small, 0
            return 'unchanged'
        self.accepted = None
        if not same(self.anchor, small):
            self.anchor, self.count = small, 1
            return 'moving'
        self.count += 1
        if self.count < self.settings.stable_frames:
            return 'moving'
        self.accepted = small.copy()
        return 'ready'
