"""Local image evidence for exposed faces; does not reconstruct hidden tiles.

A bright rectangle alone never implies accessibility. A complete pale rim
plus either an empty exterior corridor or a continuous exposed body edge is
required. Body colour is calibrated from repeated ribs outside several faces.
No absolute layer number is assigned by this detector.
"""
from __future__ import annotations

import cv2
import numpy as np


def background_mask(hsv):
    h, w = hsv.shape[:2]
    border = max(2, round(min(h, w)*.025))
    samples = np.concatenate((hsv[:border].reshape(-1, 3), hsv[-border:].reshape(-1, 3),
                              hsv[:, :border].reshape(-1, 3), hsv[:, -border:].reshape(-1, 3)))
    saturated = samples[(samples[:, 1] > 50) & (samples[:, 2] > 30)]
    if len(saturated) > len(samples)*.55:
        counts = np.bincount(saturated[:, 0]//6, minlength=30)
        centre = int(counts.argmax())*6+3
        dist = np.minimum(abs(hsv[:, :, 0].astype(float)-centre), 180-abs(hsv[:, :, 0].astype(float)-centre))
        reference = saturated[np.minimum(abs(saturated[:, 0].astype(float)-centre), 180-abs(saturated[:, 0].astype(float)-centre)) <= 12]
        if len(reference) < len(samples)*.5:
            return np.zeros((h, w), bool), False
        median = np.median(reference, axis=0)
        mask = (dist <= 13) & (hsv[:, :, 1] >= max(35, median[1]-85)) & (abs(hsv[:, :, 2].astype(float)-median[2]) <= 85)
        return mask, True
    dark = samples[(samples[:, 1] < 45) & (samples[:, 2] < 130)]
    if len(dark) > len(samples)*.7:
        value = np.median(dark[:, 2])
        return (hsv[:, :, 1] < 55) & (abs(hsv[:, :, 2].astype(float)-value) < 35), True
    return np.zeros((h, w), bool), False


def _strip(mask, x0, y0, x1, y1):
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = map(round, (x0, y0, x1, y1))
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
        return None
    return mask[y0:y1, x0:x1]


def rim_quality(hsv, rect):
    x, y, w, h = rect
    pale = (hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 145)
    # Use narrow inset strips: glyphs and shaded bevels are excluded.
    strips = [_strip(pale, x+.04*w, y+.05*h, x+.085*w, y+.95*h),
              _strip(pale, x+.915*w, y+.05*h, x+.96*w, y+.95*h),
              _strip(pale, x+.08*w, y+.025*h, x+.92*w, y+.06*h),
              _strip(pale, x+.08*w, y+.94*h, x+.92*w, y+.975*h)]
    return [float(s.mean()) if s is not None and s.size else 0. for s in strips]


def exterior_clear(background, rect, side, diagonal_bevel=False):
    x, y, w, h = rect
    # The corridor is beyond the body's bevel/extrusion, rather than the face's
    # screen-neighbour centre. It also includes shifted/half-row side contacts.
    if side == 'left':
        near, far = max(0, x-.46*w), x-.28*w
        if far-near < max(3, .09*w):
            # At a cropped left margin we can still see the actual gap
            # immediately beside the face. Never sample beyond the image.
            near, far = max(0, x-.28*w), x-1
    else:
        near, far = x+1.28*w, min(background.shape[1], x+1.46*w)
    if far-near < max(3, .09*w):
        return False, 0.
    # A bevel from the row above/below may extend across the very end of
    # the corridor. Trim only the side probe; keep full face/overlap bounds.
    inset = min(2, round(.02*h)) if diagonal_bevel else -.025*h
    strip = _strip(background, near, y+inset, far, y+h-inset)
    if strip is None or strip.size == 0:
        return False, 0.
    score = float(strip.mean())
    rows = strip.mean(axis=1)
    # Felt texture can interrupt several pixels on one scan line. Count
    # a pixel as background only if the same column is background on both
    # adjacent lines. Keep the original overall coverage requirement.
    if score > .985 and np.any(rows <= .8):
        interior = strip[1:-1] | (strip[:-2] & strip[2:])
        rows[1:-1] = interior.mean(axis=1)
    return score > .985 and float(np.mean(rows > .8)) > .985, score


def ribs(hsv, rect, hue=None):
    x, y, w, h = map(round, rect)
    found = []
    # The supplied renderer has a bevel followed by a thin saturated body rib.
    # Search a range of widths, not a fixed green colour or absolute scale.
    for column in range(x+w+max(3, round(w*.08)), min(hsv.shape[1]-1, x+w+max(3, round(w*.34)))):
        offset = column-(x+w)
        start = round(y+max(h*.15, offset*.85))
        end = round(y+h*.98)
        if start < 0 or end > hsv.shape[0] or end-start < h*.65:
            continue
        values = hsv[start:end, column]
        valid = (values[:, 1] > 110) & (values[:, 2] > 45)
        if hue is not None:
            distance = np.minimum(abs(values[:, 0].astype(float)-hue), 180-abs(values[:, 0].astype(float)-hue))
            valid &= distance < 7
        if valid.mean() < .995:
            continue
        angles = values[valid, 0].astype(float)
        centre = float(np.median(angles))
        if np.mean(np.minimum(abs(angles-centre), 180-abs(angles-centre)) < 5) < .98:
            continue
        # Neutral bevel immediately before the rib distinguishes it from a
        # coloured background edge or most pattern strokes on a neighbour.
        bevel_end = column-max(1, round(w*.015))
        bx0 = max(x+w, bevel_end-max(2, round(w*.06)))
        bevel = hsv[start:end, bx0:bevel_end]
        neutral = (bevel[:, :, 1] < 80) & (bevel[:, :, 2] > 85)
        if not bevel.size or neutral.mean() < .8:
            continue
        found.append((column, centre, float(valid.mean())))
    return found


def joined_body_edges(hsv, background, rect):
    """Match a narrow coloured right edge to the same tile's bottom edge."""
    x,y,w,h = rect
    side = _strip(hsv,x+w,y+.16*h,x+1.18*w,y+.94*h)
    base = _strip(hsv,x+.12*w,y+h,x+.88*w,y+1.23*h)
    backdrop = _strip(background,x+w,y+.16*h,x+1.18*w,y+.94*h)
    if side is None or base is None or backdrop is None:
        return None
    for col in range(2, side.shape[1]):
        stripe = side[:,col].astype(float)
        hue = np.median(stripe[:,0])
        def coloured(pixels):
            delta = abs(pixels[...,0].astype(float)-hue)
            return (np.minimum(delta,180-delta)<5) & (pixels[...,1]>115) & (pixels[...,2]>65)
        if (coloured(stripe).mean()>.98 and backdrop[:,col].mean()<.05
                and (side[:,col-2,1]<65).mean()>.96
                and (coloured(base).mean(axis=1)>.97).any()):
            return float(hue)
    return None


def crosses_behind(gray, face, neighbor):
    # A shallow intersection is consistent with an underlying face if a
    # continuous light-to-dark step follows the foreground outline through it.
    x,y,w,h = face
    u,v,s,t = neighbor
    if x < 3 or y < 3:
        return False
    if u < x < u+s <= x+.14*w:
        a,b = max(y+3,v+2),min(y+h-3,v+t-2)
        step = gray[a:b,x+3].astype(float)-gray[a:b,x-2]
    elif v < y < v+t <= y+.14*h:
        a,b = max(x+3,u+2),min(x+w-3,u+s-2)
        step = gray[y+3,a:b].astype(float)-gray[y-2,a:b]
    else:
        return False
    return len(step)>=8 and np.percentile(step,10)>22


def infer_accessibility(image, tiles, settings):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background, bg_reliable = background_mask(hsv)
    candidates = {}
    joined_hues = {}
    colours = []
    for tile in tiles:
        joined_hues[tile.index] = joined_body_edges(hsv, background, tile.rect)
        candidates[tile.index] = ribs(hsv, tile.rect)
        for column, hue, quality in candidates[tile.index]:
            x, y, w, h = map(round, tile.rect)
            probe = _strip(background, column, y+h*.2, column+1, y+h*.9)
            if probe is not None and probe.mean() < .15:
                colours.append((tile.index, hue))
    body_hue = None
    if colours:
        for _, hue in colours:
            support = {idx for idx, other in colours if min(abs(hue-other), 180-abs(hue-other)) < 7}
            if len(support) >= 3:
                body_hue = hue
                break
    for tile in tiles:
        x, y, w, h = map(round,tile.rect)
        rim = rim_quality(hsv, tile.rect)
        normal = (.90 <= w/settings.tile_width <= 1.12 and .90 <= h/settings.tile_height <= 1.12)
        overlaps = [other.index for other in tiles if other is not tile and _overlap(tile.rect, other.rect) > min(w*h, other.rect[2]*other.rect[3])*.025]
        complete = normal and min(rim) >= .91 and not overlaps
        left, left_score = exterior_clear(background, tile.rect, 'left') if bg_reliable else (False, 0.)
        right, right_score = exterior_clear(background, tile.rect, 'right') if bg_reliable else (False, 0.)
        if complete and bg_reliable:
            for side, clear in [('left', left), ('right', right)]:
                if clear:
                    continue
                adjacent = [(v,t) for other in tiles if other is not tile
                            for u,v,s,t in [other.rect]
                            if abs(u+s-x if side=='left' else u-(x+w)) <= .12*w]
                diagonal = any(0 <= y-(v+t) <= .10*h or 0 <= v-(y+h) <= .10*h
                               for v,t in adjacent)
                if diagonal and not any(v < y+h and v+t > y for v,t in adjacent):
                    evidence = exterior_clear(background,tile.rect,side,diagonal_bevel=True)
                    if side=='left':
                        left, left_score = evidence
                    else:
                        right, right_score = evidence
        # At a staircase corner the diagonal neighbour's bottom bevel and
        # shadow enter the corridor, although its face ends above this row.
        corner = False
        if bg_reliable and body_hue is not None and normal and min(rim)>=.95 and not left:
            diagonal = any(abs(u+s-x)<.10*w and abs(v+t-y)<.08*h
                           for u,v,s,t in (other.rect for other in tiles if other is not tile))
            band = _strip(hsv, x-.38*w, y, x-.12*w, y+.18*h)
            lower = _strip(background, x-.38*w, y+.32*h, x-.12*w, y+h)
            # Permit only a narrow intrusion at the opposite bevel. Half-row
            # side blockers still occupy the lower corridor.
            opposite_only = all(other.rect[0]>=x+.93*w for other in tiles if other.index in overlaps)
            if diagonal and opposite_only and band is not None and lower is not None and lower.mean()>.99:
                distance = abs(band[:,:,0].astype(float)-body_hue)
                corner = bool(np.any(((np.minimum(distance,180-distance)<7) &
                                      (band[:,:,1]>110)).mean(axis=1)>.95))
                if corner:
                    left = complete = True
        rib = bool(body_hue is not None and ribs(hsv, tile.rect, body_hue))
        joined = normal and min(rim) >= .97 and joined_hues[tile.index] is not None
        if joined and overlaps and all(crosses_behind(gray,tile.rect,other.rect)
                                      for other in tiles if other.index in overlaps):
            complete = True
        right = right or rib
        # A raised white face casts a continuous rightward shadow across the
        # lower surface. Adjacent flush faces interrupt that gradient.
        shadow = False
        if complete and not right and np.percentile(gray[y:y+h,x:x+w],80) >= 245:
            strip = _strip(gray,x+w-.04*w,y+.04*h,x+w+.36*w,y+.98*h)
            if strip is not None and strip.shape[1]>=10:
                profiles = strip.astype(float)
                trough = profiles[:,:max(3,round(.18*w))].argmin(axis=1)
                rows = np.arange(len(trough))
                near = profiles[rows,trough]
                middle = profiles[rows,np.minimum(trough+max(2,round(.08*w)), profiles.shape[1]-1)]
                far = profiles[rows,np.minimum(trough+max(4,round(.18*w)), profiles.shape[1]-1)]
                exposed = (near<75) & (middle>near+10) & (far>middle+12)
                # Ink on an underlying face can interrupt the fading tail.
                # Still require the shadow core along the entire side at a
                # consistent offset, including where the gradient is broken.
                core_end = min(profiles.shape[1],max(4,round(.19*w)))
                dark_columns = (profiles[:,2:core_end] < 85).mean(axis=0)
                core = int(dark_columns.argmax())+2
                tail = np.median(profiles[:,core:],axis=0)
                shadow = bool(exposed.mean()>.95 or
                              (dark_columns.max()>.98 and len(tail)>=6 and
                               tail[-1]-tail[0]>80 and np.all(np.diff(tail)>1)))
                if not shadow and body_hue is not None:
                    # Rounded corners shorten the straight part of the shadow.
                    # Require an independent, continuous bottom body rib as well.
                    bottom = _strip(hsv, x+.1*w, y+h, x+.9*w, y+1.18*h)
                    if bottom is not None:
                        delta = abs(bottom[:, :, 0].astype(float)-body_hue)
                        body = (np.minimum(delta, 180-delta)<7) & (bottom[:, :, 1]>110)
                        inset = max(1, round(.09*h))
                        middle_rows = profiles[inset:-inset]
                        if len(middle_rows) and np.any(body.mean(axis=1)>.97):
                            core_limit = min(profiles.shape[1]-5, round(.25*w))
                            support = (middle_rows[:,2:core_limit]<85).mean(axis=0)
                            if not support.size:
                                support = np.array([0.])
                            core = int(support.argmax())+2
                            tail = np.median(middle_rows[:,core:], axis=0)
                            shadow = bool(support.max()>.98 and len(tail)>=6 and
                                          tail[-1]-tail[0]>80 and np.all(np.diff(tail)>1))
                shadow = bool(shadow and np.median(gray[y:y+h,x:x+w]) >= 245)
                right = shadow
                if not right and bg_reliable and not overlaps and min(rim) >= .97:
                    # Part of an exposed column can share its bottom edge with
                    # the next tile. Verify the straight shadow AND the narrow
                    # body stripe, calibrated on separate L-shaped edges.
                    side = _strip(hsv,x+w-.08*w,y+.18*h,x+w+.12*w,y+.90*h)
                    interior = profiles[round(.15*h):round(.83*h)]
                    if side is not None and len(interior):
                        stripe = (side[:,:,1]>110) & (side[:,:,2]>70)
                        for col in np.where(stripe.mean(axis=0)>.98)[0]:
                            hue = np.median(side[:,col,0])
                            if sum(v is not None and min(abs(v-hue),180-abs(v-hue))<5
                                   for v in joined_hues.values()) < 3:
                                continue
                            median = np.median(interior, axis=0)
                            core = int(median[:max(3,round(.19*w))].argmin())
                            tail = median[core:]
                            if ((interior[:,core]<85).mean()>.95 and len(tail)>=6
                                    and tail[-1]-tail[0]>70 and np.all(np.diff(tail)>=0)):
                                right = shadow = True
                                break
        right = right or joined
        rib = rib or joined
        tile.visual = {'rim': [round(v, 3) for v in rim], 'complete_face': bool(complete),
                       'left_background': round(left_score, 3), 'right_background': round(right_score, 3),
                       'right_body_rib': rib, 'body_hue': body_hue, 'overlaps': overlaps}
        tile.visual['right_cast_shadow'] = shadow
        tile.visual['joined_body_edges'] = bool(joined)
        tile.visual['left_diagonal_corner'] = corner
        tile.auto_access = 'free' if complete and (left or right) else 'unknown'
        if not complete:
            tile.auto_reason = '表面の縁の欠け・サイズ差・重なりがあり、自動の空き判定を保留'
        elif left or right:
            evidence = '右側面に沿う連続した影' if shadow else ('右の露出側面' if rib else ('左の盤面背景' if left else '右の盤面背景'))
            tile.auto_reason = f'表面の連続した縁と{evidence}を画像から検出（絶対段数は不明）'
        else:
            tile.auto_reason = '表面は見えますが、左右の空きの画像根拠が不足'

    if settings.use_bright_face_cue:
        # Calibrated theme: require both the observed dim and bright ivory
        # palettes in this frame before interpreting the game's own shading.
        shades = []
        for tile in tiles:
            x, y, w, h = map(round, tile.rect)
            face = image[y:y+h, x:x+w]
            shades.append(np.percentile(face.reshape(-1, 3), 80, axis=0))
        if any(np.max(abs(s-[213,218,222])) <= 3 for s in shades):
            for tile, shade in zip(tiles, shades):
                x,y,w,h = tile.rect
                if (np.max(abs(shade-[235,240,245])) <= 3 and
                    .94 <= w/settings.tile_width <= 1.06 and .94 <= h/settings.tile_height <= 1.06
                    and min(tile.visual['rim']) >= .95
                    and all(_overlap(tile.rect, o.rect)/(w*h) < .065 for o in tiles if o is not tile)):
                    tile.auto_access = 'free'
                    tile.auto_reason = 'このテーマの明るい牌面と四辺を検出'


def _overlap(a, b):
    x, y, w, h = a
    u, v, s, t = b
    return max(0, min(x+w, u+s)-max(x, u))*max(0, min(y+h, v+t)-max(y, v))
