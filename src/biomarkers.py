"""
Clinical biomarkers from the lesion masks.

Shared by the fusion experiment and the app, so the numbers a clinician sees are
exactly the numbers the model was tested on. Everything is scaled by the limbus
(a fixed ~11.7 mm anatomical ruler), so features are comparable across
magnifications.

Design notes tied to clinical signs:
  hypopyon flatness   Feret / equivalent-diameter. A settled bacterial hypopyon
                      is a flat mobile crescent (high ratio); a fungal one tends
                      thicker/immobile.
  infiltrate solidity area / convex-hull area. A compact suppurative bacterial
                      ulcer is solid (high); a feathery/serrated fungal margin
                      is spiky (low).
  n_components        multifocal / satellite spread favours fungal.
  cellularity ratio   spread halo relative to core - EXPLORATORY (annotation was
                      subjective and it appears in healing), gated on external
                      validation before it is trusted.
"""

import numpy as np
import cv2

# feature order is fixed and shared with the fusion model
FEATURES = [
    "hypopyon_present", "hypopyon_area_frac", "hypopyon_flatness", "hypopyon_y_rel",
    "infiltrate_area_frac", "infiltrate_n_comp", "infiltrate_solidity",
    "infiltrate_circularity",
    "cellularity_area_frac", "cellularity_to_infiltrate",     # exploratory
    "glare_area_frac",
]
EXPLORATORY = {"cellularity_area_frac", "cellularity_to_infiltrate"}


def _largest_geom(binmask):
    """area, n_components, solidity, circularity, feret, equiv_diam for a binary mask."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(binmask.astype(np.uint8), 8)
    if n <= 1:
        return dict(area=0.0, n_comp=0, solidity=0.0, circ=0.0, feret=0.0, equiv=0.0)
    # count real components (ignore tiny specks < 0.5% of the largest)
    areas = st[1:, cv2.CC_STAT_AREA]
    big = int((areas > 0.005 * areas.max()).sum())
    i = 1 + int(np.argmax(areas))
    comp = (lab == i).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True))
    hull = float(cv2.contourArea(cv2.convexHull(c)))
    (_, _), (w, h), _ = cv2.minAreaRect(c) if len(c) >= 3 else ((0, 0), (0, 0), 0)
    feret = float(max(w, h))
    equiv = float(2 * np.sqrt(area / np.pi)) if area > 0 else 0.0
    return dict(area=area, n_comp=big,
                solidity=area / hull if hull > 0 else 0.0,
                circ=4 * np.pi * area / (per * per) if per > 0 else 0.0,
                feret=feret, equiv=equiv)


def compute(label_map, class_ids, limbus_mask):
    """
    label_map: HxW int class map from the lesion segmenter (same size as limbus_mask).
    class_ids: {'infiltrate':id, 'hypopyon':id, 'cellularity':id, 'glare':id}
    limbus_mask: HxW {0,1}. Returns an ordered dict of FEATURES.
    """
    limb_area = float(limbus_mask.sum()) or 1.0
    cy_limb = np.where(limbus_mask.sum(1) > 0)[0]
    limb_top, limb_bot = (int(cy_limb.min()), int(cy_limb.max())) if len(cy_limb) else (0, 1)
    limb_h = max(1, limb_bot - limb_top)

    infl = (label_map == class_ids["infiltrate"]) & (limbus_mask > 0)
    hypo = (label_map == class_ids["hypopyon"]) & (limbus_mask > 0)
    cell = (label_map == class_ids["cellularity"]) & (limbus_mask > 0)
    glare = (label_map == class_ids["glare"]) & (limbus_mask > 0)

    gi = _largest_geom(infl)
    gh = _largest_geom(hypo)

    hypo_area = float(hypo.sum())
    infl_area = float(infl.sum())
    cell_area = float(cell.sum())

    # hypopyon vertical position (1 = bottom of cornea) - a sanity/quality cue
    if hypo_area > 0:
        yy = np.where(hypo.sum(1) > 0)[0]
        hypo_y_rel = float((yy.mean() - limb_top) / limb_h)
    else:
        hypo_y_rel = 0.0

    f = {
        "hypopyon_present": 1.0 if hypo_area / limb_area > 0.002 else 0.0,
        "hypopyon_area_frac": hypo_area / limb_area,
        "hypopyon_flatness": (gh["feret"] / gh["equiv"]) if gh["equiv"] > 0 else 0.0,
        "hypopyon_y_rel": hypo_y_rel,
        "infiltrate_area_frac": infl_area / limb_area,
        "infiltrate_n_comp": float(gi["n_comp"]),
        "infiltrate_solidity": gi["solidity"],
        "infiltrate_circularity": gi["circ"],
        "cellularity_area_frac": cell_area / limb_area,
        "cellularity_to_infiltrate": (cell_area / infl_area) if infl_area > 0 else 0.0,
        "glare_area_frac": glare.sum() / limb_area,
    }
    return {k: float(f[k]) for k in FEATURES}
