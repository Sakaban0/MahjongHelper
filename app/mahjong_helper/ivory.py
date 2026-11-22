"""Image geometry for the 75 by 100 ivory Mah Jong tile set."""
import cv2
import numpy as np
from functools import lru_cache
from pathlib import Path

def fit_board(image, tiles):
    faces = [t for t in tiles if .65 < t.rect[2]/t.rect[3] < .85]
    if len(faces) < 12:
        return
    rects = np.array([t.rect for t in faces])
    w,h = np.median(rects[:,2:], axis=0)
    points = rects[:,:2]
    # Verify the renderer palette and measure spacing independently of the bevel.
    sx = np.minimum(points[:,0]+int(.08*w), image.shape[1]-1)
    sy = np.minimum(points[:,1]+int(.09*h), image.shape[0]-1)
    pixels = image[sy, sx].astype(float)
    if np.mean(np.max(abs(pixels-[233,250,255]),axis=1)<6)<.55:
        return
    dx = points[:,0,None]-points[:,0]
    dy = points[:,1,None]-points[:,1]
    widths = dx[(abs(dy)<=2)&(dx>.96*w)&(dx<1.10*w)]
    heights = dy[(abs(dx)<=2)&(dy>.96*h)&(dy<1.10*h)]
    def solve(scale):
        pitch = np.array([38.5,51])*scale
        offsets = 8*scale*np.arange(7)[:,None]
        def fit(origin):
            raw = (points[:,None,:]-origin+offsets)/pitch
            err = np.linalg.norm((raw-np.rint(raw))*pitch,axis=2)
            z = err.argmin(axis=1)
            dist = err.min(axis=1)
            ok = dist<3.5*scale
            return ((int(ok.sum()),-int(z[ok].sum()),-float(dist[ok].sum())), z, raw, ok)
        origin=max(points.tolist(),key=lambda p:fit(p)[0])
        return fit(origin)
    fit_result = None
    if min(len(widths),len(heights))>=3 and abs(np.median(widths)/77-np.median(heights)/102)<.04*np.median(widths)/77:
        fit_result = solve((np.median(widths)/77+np.median(heights)/102)/2)
    if fit_result is None or fit_result[0][0]<.88*len(faces):
        # Isolated stacks may have no adjacent faces on the same level.
        # Recover zoom from face size and validate against the entire lattice.
        nominal = np.median([w/75,h/100])
        fits = [solve(nominal*factor) for factor in np.linspace(.97,1.035,18)]
        fit_result = max(fits,key=lambda fit:(fit[0][0],fit[0][2],fit[0][1]))
    score,z,raw,ok=fit_result
    # A repeated plane of exposed summits may be the only measured layer.
    # Require well separated full faces and strong agreement in both axes.
    single_plane = len(set(z[ok])) < 2
    if single_plane and (score[0]<.96*len(faces) or len(faces)<15 or len(widths) or len(heights)):
        return
    if score[0]<.88*len(faces):
        return
    # Refine the shared zoom and translation using all reliable face origins.
    grid=np.rint(raw[np.arange(len(points)),z])*[38.5,51]-8*z[:,None]
    valid=ok.copy()
    for _ in range(3):
        a,b=grid[valid],points[valid]
        centered=a-a.mean(axis=0)
        if len(a)<5 or np.sum(centered**2)<1:
            return
        scale=float(np.sum(centered*(b-b.mean(axis=0)))/np.sum(centered**2))
        origin=np.median(b-a*scale,axis=0)
        valid=ok & (np.linalg.norm(points-origin-grid*scale,axis=1)<2.5*scale)
    pitch=np.array([38.5,51])*scale
    slots={}
    for i,t in enumerate(faces):
        if ok[i] and .82*w < t.rect[2] < 1.12*w and .86*h < t.rect[3] < 1.12*h:
            x,y=np.rint(raw[i,z[i]]).astype(int)
            slots[(int(z[i]),int(x),int(y))]=t
    return slots,np.array(origin),pitch,scale

@lru_cache(maxsize=1)
def references():
    with np.load(Path(__file__).with_name('ivory_patterns.npz'),allow_pickle=False) as atlas:
        return atlas['names'],atlas['samples']

def signature(image):
    patch=cv2.resize(image,(60,80),interpolation=cv2.INTER_AREA)[8:-8,6:-6]
    ink=cv2.GaussianBlur(np.maximum(0,245-patch.astype(np.float32)),(3,3),.7)
    return ink / max(1.,np.linalg.norm(ink))

def recover_faces(image, result, slots, origin, pitch, scale, names, samples):
    from .core import Tile
    width,height=round(75*scale),round(100*scale)
    neighbors={(z,x+dx,y+dy) for z,x,y in slots for dx,dy in [(-2,0),(2,0),(0,-2),(0,2)]}
    assigned = {t.index for t in slots.values()}
    # A clipped contour can be isolated from other faces on its layer.
    for tile in result.tiles:
        if tile.index in assigned:
            continue
        left, top, w, h = tile.rect
        for z in range(7):
            position = (np.array([left, top]) - origin + 8*scale*z) / pitch
            x, y = np.rint(position).astype(int)
            if np.linalg.norm((position - [x, y]) * pitch) <= 3*scale:
                neighbors.add((z, int(x), int(y)))
    for z,x,y in sorted(neighbors-slots.keys()):
        u,v=np.rint(origin+pitch*[x,y]-8*scale*z).astype(int)
        if u<2 or v<2 or u+width>=image.shape[1] or v+height>=image.shape[0]:
            continue
        nearby = [t for t in result.tiles
                  if abs(t.rect[0]-u)<.30*width and abs(t.rect[1]-v)<.30*height]
        if len(nearby)>1 or (nearby and nearby[0].index in assigned):
            continue
        if nearby:
            # A clipped face can retain its bottom/right edges even when the
            # contour's top/left corner misses the lattice fit.
            left, top, w, h = nearby[0].rect
            aligned_x = min(abs(left-u), abs(left+w-u-width))
            aligned_y = min(abs(top-v), abs(top+h-v-height))
            if max(aligned_x, aligned_y) > 3*scale or min(w/width, h/height)<.72:
                continue
        crop=image[v:v+height,u:u+width]
        scores=np.einsum('ijkl,jkl->i',samples,signature(crop))
        best,second=scores.argsort()[-2:][::-1]
        if scores[best]<.92 or scores[best]-scores[second]<.035:
            continue
        # Require the expected pale face just inside the top and left boundaries.
        visible = crop
        if nearby:
            visible = image[top:top+h, left:left+w]
        vh, vw = visible.shape[:2]
        strips = (visible[4, vw//8:-(vw//8)], visible[vh//8:-(vh//8), 4])
        if any((strip.min(axis=1)>205).mean()<.85 for strip in strips):
            continue
        if nearby:
            tile = nearby[0]
        else:
            tile = Tile(len(result.tiles)+1,(int(u),int(v),width,height),str(names[best]),
                        float(scores[best]),float(scores[second]),float(u/width),float(v/height))
            result.tiles.append(tile)
        assigned.add(tile.index)
        slots[z,x,y]=tile

def apply_board(image,result,settings):
    model=fit_board(image,result.tiles)
    if model is None:
        return False
    slots,origin,pitch,scale=model
    names,samples=references()
    recover_faces(image,result,slots,origin,pitch,scale,names,samples)
    registered = {entry["id"] for entry in settings.templates}
    for (z,x,y),tile in slots.items():
        corner=np.rint(origin+pitch*[x,y]-8*scale*z).astype(int)
        u,v=corner
        crop=image[v:v+round(100*scale),u:u+round(75*scale)]
        if crop.size==0:
            continue
        scores=samples.reshape(len(samples),-1) @ signature(crop).ravel()
        # Fine strokes on white dragons and numeral tiles shift by a pixel
        # with subpixel rendering. Search translations only for weak matches.
        if scores.max() < .96:
            width,height = crop.shape[1],crop.shape[0]
            for du,dv in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)):
                a,b = u+du,v+dv
                if a<0 or b<0 or a+width>image.shape[1] or b+height>image.shape[0]:
                    continue
                patch = image[b:b+height,a:a+width]
                scores = np.maximum(scores, np.einsum('ijkl,jkl->i',samples,signature(patch)))
        order=np.argsort(scores)
        best=order[-1]
        manual_match = tile.pattern in registered and not tile.pattern.startswith("自動-")
        if not manual_match and scores[best]>.89 and scores[best]-scores[order[-2]]>.025:
            tile.pattern=str(names[best])
            tile.confidence=float(min(1.,scores[best]))
            tile.runner_up=float(scores[order[-2]])
        if tile.pattern and tile.pattern.startswith(('枝絵・','鉢絵・')):
            tile.visual['match_group']=tile.pattern.split('・')[0]
        tile.visual['estimated_layer']=z
        tile.visual['half_grid']=[x,y]
        above=any(k>z and abs(a-x)<2 and abs(b-y)<2 for k,a,b in slots)
        # An upper face can hide an adjacent blocker. Reserve that footprint.
        def blocked_side(side):
            return any((k==z and (a-x)*side==2 and abs(b-y)<2)
                       or (k>z and 1<=(a-x)*side<=3 and abs(b-y)<2)
                       for k,a,b in slots)
        free=not above and not (blocked_side(-1) and blocked_side(1))
        tile.auto_access='free' if free else 'unknown'
        tile.auto_reason='段の配置から上と片側の空きを検出' if free else '上または同じ段の両側に牌を検出、または隣が隠れている'
    return True
