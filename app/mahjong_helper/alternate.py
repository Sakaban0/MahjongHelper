"""Measurements and glyph recognition for three screenshot tile sets."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# Full row/column pitch and per-level offset in reference-image pixels.
PROJECTIONS = {'teal': (71.2, 88.15, -7.1, -11.5),
               'green': (47.5, 62.45, -8.2, -10.1),
               'jade': (59.4, 74.4, -4.25, -5.0)}


def identify(image):
    pixels = image[::6, ::6].reshape(-1, 3).astype(np.int16)
    def fraction(color):
        return np.mean(np.max(abs(pixels-color), axis=1) <= 1)
    if fraction([213,218,222])>.01 and fraction([235,240,245])>.01:
        return 'blue'
    if fraction([255,254,255])>.05 and fraction([145,129,55])>.012:
        return 'teal'
    green_felt = (pixels[:,1]>pixels[:,0]+26) & (pixels[:,1]>pixels[:,2]+27)
    if green_felt.mean()>.12 and min(fraction([253,254,246]),fraction([253,254,255]))>.008:
        return 'jade'
    if fraction([249,250,250])>.07 and fraction([141,141,141])>.003:
        return 'green'
    return None


def blue_size(image):
    # A connected component of ivory paper surrounds the printed strokes.
    # Bounding boxes of isolated ink marks cannot win this vote.
    mask = cv2.inRange(image, (233,238,243), (237,242,247))
    stats = cv2.connectedComponentsWithStats(mask)[2][1:]
    faces = [(w,h) for _,_,w,h,area in stats
             if w>=18 and h>=25 and .56<w/h<.75 and area>.38*w*h]
    if len(faces)<4:
        return
    w,h=np.median(faces,axis=0)
    if sum(abs(a/w-1)<.09 and abs(b/h-1)<.09 for a,b in faces)>=4:
        return float(round(w*1.085)),float(round(h*1.055))


def descriptor(crop):
    small=cv2.resize(crop,(36,48),interpolation=cv2.INTER_AREA)[3:-3,2:-2].astype(np.float32)
    paper=np.percentile(small,90,axis=(0,1)).astype(np.float32)
    ink=cv2.GaussianBlur(np.maximum(0,paper-small-12),(3,3),.65)
    return ink.ravel()/max(1.,np.linalg.norm(ink))


@lru_cache(maxsize=3)
def atlas(theme):
    with np.load(Path(__file__).with_name('alternate_patterns.npz'),allow_pickle=False) as data:
        return data[theme+'_names'],data[theme+'_ink']


def match(image, rect, theme):
    x,y,w,h=rect
    if x<0 or y<0 or x+w>image.shape[1] or y+h>image.shape[0]:
        return None
    names,refs=atlas(theme)
    feature=descriptor(image[y:y+h,x:x+w])
    scores=np.einsum('ij,j->i',refs,feature)
    best=int(np.argmax(scores))
    rival=float(np.max(scores[names!=names[best]]))
    if scores[best]>.89 and scores[best]-rival>.033:
        return str(names[best]),min(1.,float(scores[best])),max(0.,rival)


def fit_grid(tiles, theme, face_size):
    geometry=np.array(PROJECTIONS[theme])
    width,height=face_size
    faces=[t for t in tiles if .74*width<t.rect[2]<1.18*width and .72*height<t.rect[3]<1.18*height]
    if len(faces)<9:
        return
    points=np.array([t.rect[:2] for t in faces])
    zoom=np.median(np.array(face_size)/geometry[:2])
    best=None
    scale_factors = np.linspace(.98, 1.16, 37) if theme == 'jade' else np.linspace(.98, 1.065, 18)
    for scale in zoom*scale_factors:
        pitch=geometry[:2]*scale/2
        offsets=np.arange(6)[:,None]*geometry[2:]*scale
        for origin in points[::2]:
            raw=(points[:,None]-origin-offsets)/pitch
            errors=np.linalg.norm((raw-np.rint(raw))*pitch,axis=2)
            z=errors.argmin(axis=1)
            residual=errors.min(axis=1)
            accepted=residual<max(2.,3.1*scale)
            score=(accepted.sum(),-round(float(np.median(residual)), 0),-z[accepted].sum(),-float(residual[accepted].sum()))
            if theme == 'jade':
                # High layers approach the next half-cell phase. One stray
                # contour must not invert the whole stack at smaller zooms.
                support = accepted.sum() - .08 * z[accepted].sum()
                score = (support, accepted.sum(), -float(residual[accepted].sum()))
            if best is None or score>best[0]:
                best=(score,origin,scale,z,np.rint(raw[np.arange(len(z)),z]).astype(int),accepted)
    score,origin,scale,layers,grid,accepted=best
    if accepted.sum()<len(faces)*.77:
        return
    # Translate the reference to the lowest observed phase.
    slots={(int(z),int(x),int(y)):t for t,z,(x,y),ok in zip(faces,layers,grid,accepted) if ok}
    return slots,origin,geometry*scale


def read_board(image,result,settings,theme):
    if theme is None:
        return False
    w,h=map(round,result.face_size)
    if theme=='jade':
        from .jade import read_jade
        return read_jade(image,result,settings)
    if theme=='blue':
        for t in result.tiles:
            x,y,tw,th=t.rect
            if not (.85*w<=tw<=1.14*w and .85*h<=th<=1.14*h):
                continue
            shade=np.percentile(image[y:y+th,x:x+tw],80,axis=(0,1))
            bright=any(np.max(abs(shade-tone))<=3 for tone in ([235,240,245],[225,245,244]))
            dim=any(np.max(abs(shade-tone))<=3 for tone in ([213,218,222],[204,222,221]))
            matched=match(image,(x,y,w,h),theme)
            if matched and t.pattern not in {e['id'] for e in settings.templates if not e['id'].startswith('自動-')}:
                t.pattern,t.confidence,t.runner_up=matched
            if bright or dim:
                t.auto_access='free' if bright else 'unknown'
                t.auto_reason='この牌セットの明るい牌面を検出' if bright else 'この牌セットの暗い牌面を検出'
            set_group(t)
        return False
    fitted=fit_grid(result.tiles,theme,result.face_size)
    if fitted is None:
        return False
    slots,origin,geometry=fitted
    pitch=geometry[:2]/2
    delta=geometry[2:]
    # The same design can also identify a visible face whose contour was split
    # by a bevel. Search near the measured layout, respecting existing faces.
    from .core import Tile
    candidates=set()
    for (z,x,y) in slots:
        for a,b in ((-2,0),(2,0),(0,-2),(0,2)):
            candidates.add((z,x+a,y+b))
        for dz in (1,2):
            if z+dz<=5:
                candidates.update((z+dz,x+a,y+b) for a in (-1,0,1) for b in (-1,0,1))
    assigned={t.index for t in slots.values()}
    # A lone cap can be several layers above the nearest visible rim. Search
    # higher phases near the measured board even when the intervening faces
    # have no usable contours.
    if theme == "teal":
        for z,x,y in slots:
            for top in range(z+1, 5):
                candidates.update((top,x+a,y+b) for a in range(-3,4) for b in range(-3,4))
    manual={e['id'] for e in settings.templates if not e['id'].startswith('自動-')}
    for key in sorted(set(slots)|candidates):
        z,x,y=key
        u,v=map(int,np.rint(origin+pitch*[x,y]+z*delta))
        t=slots.get(key)
        others=[] if t else [other for other in result.tiles
                              if abs(other.rect[0]-u)<.32*w and abs(other.rect[1]-v)<.32*h]
        if t is None and (len(others)>1 or any(o.index in assigned for o in others)):
            continue
        matched=match(image,(u,v,w,h),theme)
        if t is None and theme == "teal" and z >= 3 and not matched:
            for shift in (3,4,5,-2):
                alternative=match(image,(u+shift,v,w,h),theme)
                if alternative and alternative[1]>.965:
                    matched=alternative
                    u+=shift
                    break
        if t is None:
            if not matched or matched[1]<.95:
                continue
            if others:
                t=others[0]
            else:
                t=Tile(len(result.tiles)+1,(u,v,w,h),board_x=u/w,board_y=v/h)
                result.tiles.append(t)
            slots[key]=t
            assigned.add(t.index)
        if matched and t.pattern not in manual:
            t.pattern,t.confidence,t.runner_up=matched
        set_group(t)
    for (z,x,y),t in slots.items():
        above=any(k>z and abs(a-x)<2 and abs(b-y)<2 for k,a,b in slots)
        def occupied(side):
            return any(abs(b-y)<2 and ((k==z and (a-x)*side==2) or
                       (k>z and 1<=(a-x)*side<=3)) for k,a,b in slots)
        t.visual['estimated_layer']=z
        t.auto_access='free' if not above and not (occupied(-1) and occupied(1)) else 'unknown'
        t.auto_reason='配置から上と片側の空きを検出' if t.auto_access=='free' else '上・左右の重なりで空き判定を保留'
    return True


def set_group(tile):
    if tile.pattern in ('梅','蘭','竹','菊'):
        tile.visual['match_group']='花牌'
    elif tile.pattern in ('春','夏','秋','冬'):
        tile.visual['match_group']='季節牌'
