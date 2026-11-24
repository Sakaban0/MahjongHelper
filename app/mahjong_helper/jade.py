"""Recognize patterned white dragons and separated upper rows on green felt."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .alternate import fit_grid


def glyph(crop):
    # Only the central ink is used; bevels, debug outlines and drop shadows
    # should not become part of the label.
    patch = cv2.resize(crop, (40, 50), interpolation=cv2.INTER_AREA)[5:-5, 4:-4]
    color = patch.astype(np.float32)
    chroma = color.max(axis=2) - color.min(axis=2)
    pigment = (np.maximum(0, 250-color) * (chroma > 32)[:,:,None])
    # Exclude the amber and lime used by the helper's optional debug outline.
    b,g,r = color.transpose(2,0,1)
    pigment[((r>235)&(g>180)&(b<146)) | ((g>207)&(r<115)&(b>107)&(b<163))] = 0
    pigment = cv2.GaussianBlur(pigment,(3,3),.8).ravel()
    return pigment / max(1., np.linalg.norm(pigment))


@lru_cache(maxsize=1)
def references():
    with np.load(Path(__file__).with_name('jade_patterns.npz'),allow_pickle=False) as refs:
        return refs['names'], refs['samples']


def white_dragon(crop):
    b,g,r = cv2.resize(crop,(40,50),interpolation=cv2.INTER_AREA).astype(float).transpose(2,0,1)
    # Four blue ornaments on an otherwise unprinted field. A cast shadow may
    # darken the field or hide one corner. A uniform white region is insufficient.
    blue = (b>r+42)&(b>g+40)
    corners = [blue[y:y+11,x:x+10].sum() for x in (2,28) for y in (2,37)]
    center = np.stack((b,g,r),axis=2)[13:37,10:30]
    is_frame = (sum(count>=3 for count in corners)>=3 or min(corners[0],corners[2])>=8 or min(corners[1],corners[3])>=8)
    horizontal = max(blue[:14,8:32].sum(axis=1).max(), blue[-14:,8:32].sum(axis=1).max())
    return (horizontal >= 6 and is_frame and
            (np.ptp(center,axis=2)>35).mean()<.035 and
            np.median(center)>175 and np.mean(blue[12:38,10:30])<.015)


def circle_count(crop):
    """Red and green rows are shared; blue centers distinguish five from six."""
    b,g,r = cv2.resize(crop,(48,60),interpolation=cv2.INTER_AREA).astype(float).transpose(2,0,1)
    red = (r>g+28)&(r>b+28)
    green = (g>r+18)&(g>b+14)
    blue = (b>r+30)&(b>g+26)
    def centers(mask, lo, hi):
        band = mask[lo:hi,3:45].astype(np.uint8)
        _, _, stats, points = cv2.connectedComponentsWithStats(band)
        return sorted(x+3 for (_,_,w,h,area),(x,y) in zip(stats[1:],points[1:])
                      if 5<=w<=16 and 5<=h<=17 and area>=19)
    def two(xs):
        return len(xs)==2 and 7<=xs[0]<=20 and 28<=xs[1]<=42
    if two(centers(red,5,27)) and two(centers(green,34,55)):
        middle = centers(blue,19,42)
        if two(middle):
            return '六筒'
        if len(middle)==1 and 17<=middle[0]<=30:
            return '五筒'


def expand_stock_face(image, tile, stock_mask):
    """A circular pip can supply the detection rectangle instead of the paper."""
    x, y, w, h = tile.rect
    if w < .91*h:
        return
    pad = round(.48 * w)
    left, top = max(0, x-pad), max(0, y-pad)
    patch = image[top:y+h+pad, left:x+w+pad]
    mask = (patch.min(axis=2) > 230).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        u, v, width, height = cv2.boundingRect(contour)
        a, b = left+u, top+v
        if (.70 < width/height < .88 and w <= width <= w*1.5 and h < height <= h*1.65
                and abs(a+width/2-x-w/2) < .1*w and abs(b+height/2-y-h/2) < .12*h
                and stock_mask[b:b+height, a:a+width].all()):
            tile.rect = (a, b, width, height)
            return


def identify_reviewed_faces(image, tiles, samples):
    for tile in tiles:
        if tile.pattern and not tile.pattern.startswith('自動-'):
            continue
        x, y, w, h = tile.rect
        sample = glyph(image[y:y+h, x:x+w])
        scores = {}
        for name, reference in samples:
            score = float(sample @ reference)
            scores[name] = max(scores.get(name, 0.), score)
        ranked = sorted(scores, key=scores.get, reverse=True)
        name = ranked[0]
        rival = scores[ranked[1]] if len(ranked) > 1 else 0.
        if scores[name] >= .97 and scores[name]-rival >= .065:
            tile.pattern, tile.confidence, tile.runner_up = name, scores[name], rival


def recover_shaded_faces(image, result):
    """Recover partly shaded paper only with a matching face on this board."""
    from .core import Tile
    w, h = map(round, result.face_size)
    refs = [(t.pattern, glyph(image[y:y+b, x:x+a])) for t in result.tiles
            for x, y, a, b in [t.rect] if t.pattern and .9*w < a < 1.1*w
            and .9*h < b < 1.1*h]
    if not refs:
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(cv2.inRange(gray, 200, 255), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, a, b = cv2.boundingRect(contour)
        if not (.74*w < a <= w and .77*h < b <= h and x+w <= image.shape[1]
                and y+h <= image.shape[0] and cv2.contourArea(contour) > .54*a*b):
            continue
        if any(abs(t.rect[0]-x) < .45*w and abs(t.rect[1]-y) < .45*h for t in result.tiles):
            continue
        sample = glyph(image[y:y+h, x:x+w])
        scores = sorted(((float(sample @ ref), name) for name, ref in refs), reverse=True)
        score, name = scores[0]
        rival = max((v for v, label in scores if label != name), default=0.)
        if score < .97 or score-rival < .07:
            continue
        tile = Tile(len(result.tiles)+1, (x,y,w,h), name, score, rival, x/w, y/h)
        tile.auto_access = 'unknown'
        tile.auto_reason = '影のある牌面を同柄から復元。上の牌との重なりを確認できず保留'
        result.tiles.append(tile)


def exposed_front_face(image, tile, tiles):
    """A front tile can hide the bottom bevel of the tile behind its top edge."""
    v = tile.visual
    if (tile.auto_access != 'unknown' or not tile.pattern or v.get('higher_blocker')
            or min(v.get('left_background', 0), v.get('right_background', 0)) < .90
            or min(v.get('rim') or [0]) < .95):
        return False
    x, y, w, h = tile.rect
    neighbors = []
    for other in tiles:
        if other is tile:
            continue
        a, b, c, d = other.rect
        if min(x+w, a+c) > max(x, a) and min(y+h, b+d) > max(y, b):
            neighbors.append((a, b, c, d))
    if len(neighbors) != 1:
        return False
    a, b, c, d = neighbors[0]
    if not (.035*w < a-x < .15*w and b < y and 0 < b+d-y <= .12*h
            and .88*w < c < 1.12*w and .85*h < d < 1.12*h):
        return False
    # Across the overlap the rear paper ends in shadow, followed by the
    # continuous bright top rim of the foreground face.
    left, right = x+round(.18*w), x+round(.82*w)
    top = image[y:y+round(.12*h), left:right].min(axis=2)
    margin = max(2, round(.025*h))
    return bool(len(top) >= 2*margin and
                np.mean(top[:margin] < 225) > .85 and
                np.mean(top[-margin:] > 235) > .90)


def verify_faces(image, tiles):
    for tile in tiles:
        x,y,w,h = map(round,tile.rect)
        count = circle_count(image[y:y+h,x:x+w])
        if count or tile.pattern in ('五筒','六筒'):
            tile.pattern = count
        if exposed_front_face(image, tile, tiles):
            tile.auto_access = 'free'
            tile.auto_reason = '左右の背景と、奥の牌の手前に続く明るい上縁を確認'
        if tile.auto_access != 'free':
            continue
        # Half-row neighbours can block either side even without a fitted slot.
        blocked = set()
        for other in tiles:
            if other is tile:
                continue
            a,b,c,d = other.rect
            rows = abs(b-y)/h
            if min(abs(rows),abs(rows-.5))>.04 or not .88*h<d<1.12*h:
                continue
            if abs(a-x-w)<.055*w:
                blocked.add(1)
            if abs(x-a-c)<.055*w:
                blocked.add(-1)
        if (1 in blocked or tile.visual.get('right_blocked')) and (-1 in blocked or tile.visual.get('left_blocked')):
            tile.auto_access = 'unknown'
            tile.auto_reason = '左右の牌面が接しているため保留'


def classify(crop):
    names, samples = references()
    scores = samples @ glyph(crop)
    best = int(scores.argmax())
    strength = float(scores[best])
    rival = float(scores[names != names[best]].max())
    if white_dragon(crop):
        corner_score = max(0., float(scores[names == '白'].max()))
        if corner_score < .59:
            return None, True
        return ('白', .941 + .058 * corner_score, 0.), True
    if names[best] != '白' and strength > .88 and strength - rival > .04:
        return (str(names[best]), min(1., strength), rival), True
    return None, strength > .65


def read_jade(image,result,settings):
    from .core import Tile
    # Inner ornamental frames are smaller than full faces and vote for a
    # different grid phase. First fit only the full-size contours.
    # The score tray has tile-shaped outlines but exposes the dark felt.
    faces = []
    for tile in result.tiles:
        x,y,w,h = tile.rect
        crop = image[y:y+h,x:x+w]
        if crop.size and (crop.min(axis=2)>150).mean() >= .4:
            tile.index = len(faces)+1
            faces.append(tile)
    result.tiles[:] = faces
    w,h=result.face_size
    full=[t for t in result.tiles if .88*w<t.rect[2]<1.14*w and .88*h<t.rect[3]<1.14*h]
    fitted=fit_grid(full,'jade',result.face_size)
    if fitted is None:
        return False
    slots,origin,geometry=fitted
    pitch,offset=geometry[:2]/2,geometry[2:]
    width,height=np.rint(geometry[:2]*.98).astype(int)
    planes=range(max(z for z,_,_ in slots)+1)
    grid=np.array([(x,y) for _,x,y in slots])
    left,top=grid.min(axis=0)-2
    right,bottom=grid.max(axis=0)+2
    candidates=[]
    for z in planes:
        for x in range(left,right+1):
            for y in range(top,bottom+1):
                u,v=np.rint(origin+pitch*[x,y]+z*offset).astype(int)
                if u<0 or v<0 or u+width>image.shape[1] or v+height>image.shape[0]:
                    continue
                crop=image[v:v+height,u:u+width]
                # Large patches of dark table or an adjacent tile cannot be
                # accepted merely because the remaining pigment looks similar.
                if (crop.min(axis=2)>150).mean()<.45:
                    continue
                # Contours include one or two bevel pixels. Locate the printed
                # face locally without changing its estimated board cell.
                matched, plausible = classify(crop)
                if plausible and (matched is None or matched[1] < .98 or matched[0] == '白'):
                    step = max(1, round(width / 58))
                    for dx,dy in ((-1,-1),(0,-2),(-1,-2),(-2,-2),(-2,-3),(1,-2),(1,0),(0,1)):
                        a,b=u+dx*step,v+dy*step
                        if a<0 or b<0 or a+width>image.shape[1] or b+height>image.shape[0]:
                            continue
                        shifted, _ = classify(image[b:b+height,a:a+width])
                        if shifted:
                            shifted=(shifted[0],shifted[1]-.001*(abs(dx)+abs(dy)),shifted[2])
                        if shifted and (matched is None or shifted[1]>matched[1]):
                            matched=shifted
                if matched and (matched[1]>.91 or (z,x,y) in slots):
                    candidates.append((matched,(z,x,y),(int(u),int(v),int(width),int(height))))
    used=set()
    manual={entry['id'] for entry in settings.templates if not entry['id'].startswith('自動-')}
    for (name,score,rival),key,rect in sorted(candidates,key=lambda item:-item[0][1]):
        u,v,w,h=rect
        t=slots.get(key)
        if any(n.index in used and abs(n.rect[0]-u)<.35*w and abs(n.rect[1]-v)<.32*h for n in result.tiles):
            continue
        if t is None:
            nearby=[o for o in result.tiles if abs(o.rect[0]+o.rect[2]/2-u-w/2)<w*.28
                    and abs(o.rect[1]+o.rect[3]/2-v-h/2)<h*.28]
            if len(nearby)>1 or (nearby and nearby[0].index in used):
                continue
            t=nearby[0] if nearby else Tile(len(result.tiles)+1,rect,board_x=u/w,board_y=v/h)
            if not nearby:
                result.tiles.append(t)
            # A recovered frame supersedes its own contour-based placement.
            for old in list(slots):
                if slots[old] is t:
                    del slots[old]
            slots[key]=t
        if t.index in used:
            continue
        used.add(t.index)
        if t.pattern not in manual:
            t.pattern,t.confidence,t.runner_up=name,score,rival
        t.rect=rect
    for (z,x,y),t in slots.items():
        above=any(k>z and abs(a-x)<2 and abs(b-y)<2 for k,a,b in slots)
        if t.pattern == '白':
            # A darkened dragon with an upper face intruding into its painted
            # corners cannot be certified free from a nominal grid gap.
            u,v,w,h=t.rect
            center=image[v+h//4:v+3*h//4,u+w//4:u+3*w//4]
            def intrudes(other):
                a,b,c,d=other.rect
                shared_w=max(0,min(u+w,a+c)-max(u,a))
                shared_h=max(0,min(v+h,b+d)-max(v,b))
                return shared_w*shared_h>.05*w*h
            if np.median(center)<248:
                above |= any(k>z and intrudes(other) for (k,_,_),other in slots.items())
        def side_blocked(side):
            # A detected face can miss the grid fit (especially unlabelled
            # winds/characters). Its measured edge is still a blocker.
            u,v,w,h = t.rect
            for neighbor in result.tiles:
                if neighbor is t:
                    continue
                nu,nv,nw,nh = neighbor.rect
                edge_gap = nu-(u+w) if side==1 else u-(nu+nw)
                if abs(edge_gap)<=.09*w and abs(nv-v)<=.045*h and .85*h<nh<1.16*h:
                    return True
            if z == 0:
                # No lower supporting layer is expected at the base phase.
                # Withhold a free side if pale tile paper fills the space
                # immediately beyond its bevel, even without a fitted slot.
                edge = u+w if side==1 else u
                a,b = sorted((edge+round(side*.10*w),edge+round(side*.20*w)))
                strip = image[v+round(.24*h):v+round(.76*h),max(0,a):max(0,b)]
                if strip.size and (strip.min(axis=2)>155).mean()>.48:
                    return True
            return any(abs(b-y)<2 and ((k==z and (a-x)*side==2) or
                      (k>z and 1<=(a-x)*side<=3)) for k,a,b in slots)
        left_blocked=side_blocked(-1)
        right_blocked=side_blocked(1)
        free=not above and not (left_blocked and right_blocked)
        t.visual.update(estimated_layer=z, half_grid=[x,y], left_blocked=left_blocked,
                        right_blocked=right_blocked, higher_blocker=above)
        t.auto_access='free' if free else 'unknown'
        t.auto_reason=('配置から上と片側の空きを検出' if free else
                       '上に牌があるか、左右が塞がっている可能性があります')
    return True
