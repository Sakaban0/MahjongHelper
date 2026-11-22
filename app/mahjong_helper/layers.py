"""Experimental relative layers for staggered, regularly spaced renderers."""
import numpy as np


def estimate_layers(tiles, settings):
    faces = [t for t in tiles if .91 < t.rect[2]/settings.tile_width < 1.10
             and .91 < t.rect[3]/settings.tile_height < 1.10]
    for tile in tiles:
        tile.visual['estimated_layer'] = None
    if len(faces) < 6:
        return
    points = np.array([(x+w/2, y+h/2) for x,y,w,h in (t.rect for t in faces)])
    delta = np.abs(points[:, None] - points)
    pitch = []
    for axis, size in enumerate((settings.tile_width, settings.tile_height)):
        distances = delta[:,:,axis]
        near = distances[(distances > .92*size) & (distances < 1.18*size)
                         & (delta[:,:,1-axis] < 3)]
        if len(near) < 6:
            return
        pitch.append(np.median(near)/2)
    pitch = np.array(pitch)
    offsets = np.arange(5)[:,None]*[settings.layer_dx, settings.layer_dy]
    # Use the most widely repeated lattice phase as the reference. Offsets
    # indistinguishable on a half-tile lattice produce ambiguous hypotheses.
    def errors(anchor):
        residual = (points[:,None]-anchor-offsets+pitch/2) % pitch-pitch/2
        return np.linalg.norm(residual, axis=2)

    tolerance = max(2., min(pitch)*.12)
    anchor = max(points, key=lambda point: np.sum(errors(point)[:,0] < tolerance))
    distances = errors(anchor)
    if np.sum(distances[:,0] < tolerance) < max(4, len(faces)*.22):
        return
    ranked = np.argsort(distances, axis=1)
    votes = []
    for row, order in zip(distances, ranked):
        best, second = order[:2]
        votes.append(int(best) if row[best] < tolerance and row[second]-row[best] > tolerance*.6 else None)
    for tile, vote in zip(faces, votes):
        if vote is not None and votes.count(vote) >= 3:
            tile.visual['estimated_layer'] = vote
