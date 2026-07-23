# LVP Model 2 — Bacterial vs Fungal Keratitis

A rebuild of the corneal infection-etiology classifier, from scratch, with an
evaluation protocol that can survive review.

---

## Why this exists

The incumbent CornealAI Model 2 reports **AUC 0.862 / accuracy 0.775** on a
138-image validation split (best epoch 7 of 7). The figure usually quoted for it
— `n=686, acc 0.862, AUC 0.949` — is computed over all 686 images *including the
548 it trained on*, and is not a test result.

Both of those numbers are also contaminated: the incumbent's bag builder
(`ROIMILDataset._plan_tiles`) adds an extra lesion tile when the label is
bacterial and an extra hypopyon tile when it is fungal, and that object serves
**both train and validation**. The bag composition encodes the answer. Its
inference-time dataset has no such bump, so production sees a distribution
validation never did.

**There is currently no trustworthy number for this task.** Producing the first
one is the primary deliverable; beating 0.862 is secondary and expected to
follow from doing the first part properly.

---

## Design rules

These are non-negotiable and every stage is checked against them.

1. **Splits are patient-disjoint and assigned once**, in Phase 0. No later stage
   re-splits. The locked test set is touched exactly once, at the end.
2. **Nothing about the label may influence preprocessing, tiling or bag
   construction.** This is the specific defect that invalidated the incumbent.
3. **Aspect ratio is preserved.** `Resize((H, W))` with a tuple does not preserve
   it and silently turns a circular limbus into an ellipse *by a device-dependent
   factor* — 1.5× for the 3:2 Canon images, 1.33× for the 4:3 ones. Every resize
   letterbox-pads to square first. This corrupts shape features otherwise.
4. **Metrics are reported at source prevalence as well as cohort prevalence.**
   The cohort is curated to ~1:1; the source population is 1 bacterial : 10.6
   fungal. PPV at 1:1 is not PPV in deployment.
5. **Abstention is a first-class output.** The system may return
   *indeterminate*; coverage and risk are reported together, never accuracy alone.

---

## Data

| | |
|---|---|
| Images | 682 (335 bacterial / 347 fungal) |
| Patients | 617 — 1.105 images/patient |
| Source | `DatasetModel2`, the same cohort the incumbent used (Dataset9 + Dataset14) |
| Duplicates | none (content-hashed) |
| Unreadable | none |
| Resolutions | 5472×3648 ×641 · 3648×2432 ×23 · 1600×1200 ×18 |
| Device | Canon EOS 7D Mark II ×662, no EXIF ×20 |

**Labels are folder names.** No culture, KOH, smear or confocal result is linked
to any image. This is the single largest constraint on what the model can claim
and is logged as the top open item below.

---

## Pipeline

| Phase | What | Status | Result |
|---|---|---|---|
| **0** | Manifest: identity parsing, integrity probe, patient-grouped splits | ✅ | 682 img / 617 patients, 0 dupes, 0 leakage |
| **1a** | Metadata confound audit | ✅ | AUC **0.577 ± 0.015** — mild |
| **1b** | Image QC + handcrafted signal floor | ✅ | AUC **0.543**; 69 QC flags |
| **1c** | Frozen-representation probe (resolution × framing) | ✅ | AUC **0.747 ± 0.013** (DINOv2, eye crop @448) |
| **1d** | De-identification | ✅ | 0 PHI hits; lossless verified |
| **2a** | Limbus segmentation, native coordinates | ✅ | 6/6 anatomy checks pass, 0 failures |
| **2b** | Native-resolution tiling within the limbus (224px) | ✅ | 60,450 tiles / 682 images, 0 skipped |
| **3** | Attention MIL + pooling ablation | ✅ | mean 0.746 / max 0.749 / gated 0.689 — **no gain** |
| **2c** | Multi-scale retile (448 / 896 px) | ✅ | 29,669 tiles |
| **3b** | Tile-scale comparison (224/448/896) | ✅ | **s896 mean = 0.791 ± 0.009** ← best |
| **3c** | s896 at full 896 input | ✅ | 0.782 — fidelity adds nothing |
| **3d** | Large-scale sweep (1344/1792) | ✅ | 0.775 / 0.722 — **3.67 mm is the peak** |
| **4** | Margin feature branch: Frangi ridges, fractal dim, satellite count | | |
| **5** | Calibration + abstention: risk–coverage at target recall | | |
| **6** | Sign transfer: feathery-margin detector *(blocked — needs Dataset15/18)* | | |

### Evaluation protocol (learned the hard way)

A single `StratifiedGroupKFold` draw has **sd ≈ 0.015** on this cohort, so one
number carries ±0.03 of noise — enough to move a verdict a whole category.
Selecting a hyper-parameter on the same folds you then report adds a further
~0.012 of optimism. **Every AUC here is repeated nested CV**: inner loop selects,
outer loop reports, over ≥10 fold assignments, quoted as mean ± sd.

### Measured facts driving the design

- **Global statistics carry almost nothing (0.543); frozen deep features carry a lot (0.747).** The whole-eye view is not empty — crude statistics were blind to it.
- **Whole-eye pooled representations plateau at ~0.745**, and you can reach that plateau either by resolution (full frame @896 = 0.743) or by framing (eye crop @448 = 0.747). They are **substitutes, not additive** — the two are statistically indistinguishable, and combining both views gives no gain.
- **Past that plateau, more pixels stop helping**: eye@896 (0.738) is no better than eye@448 (0.747). A globally pooled token averages fine texture away regardless of input size.
- **The diagnostic unit is the PATTERN, not the filament.** `feathery_margin` spans a median **3.44 mm** across ~4 components. The full scale curve, mean-pooled, all fed at 448:

  | tile | field of view | AUC |
  |---|---|---|
  | 224 px | 0.92 mm | 0.745 |
  | 448 px | 1.83 mm | 0.759 |
  | **896 px** | **3.67 mm** | **0.791 ± 0.009** ← peak |
  | 1344 px | 5.50 mm | 0.775 |
  | 1792 px | 7.33 mm | 0.722 |
  | whole eye | 11.7 mm | 0.747 |

  **Unimodal, peaking at 3.67 mm — which matches the 3.44 mm anatomical scale of
  the sign itself.** A tile must be large enough to contain the pattern and small
  enough not to average it away. Geometry is settled.

- **Pixel fidelity contributes nothing.** The same 3.67 mm crops fed at 896 instead
  of 448 scored **−0.009** (0.782 vs 0.791). Two independent observations
  (also `eye@896` in Phase 1c) point to position-embedding interpolation degrading
  a backbone trained at 518. Keep the cheaper 448 input.

- **Attention is not the mechanism.** Mean pooling beats gated attention at every
  scale (0.791 vs 0.781 at s896; 0.745 vs 0.686 at s224). The signal is
  distributed across the lesion rather than concentrated in a few tiles — it just
  needs to be *seen at the right scale*. Attention adds ~100k parameters on 551
  training bags and overfits, most severely when bags are large.
- **Scale is known: 4.09 µm/px median.** A 50–200 µm fungal filament spans **12–49 px** natively. `mm_per_px` is stored per image, so lesions can be reported in mm.

Hardware: RTX 1000 Ada, **6 GB VRAM** → EfficientNet-B0 with AMP and tile
chunking. B3 was never justified at n=682 anyway.

---

## Layout

```
data/raw/{Bacterial,Fungal}/     682 source JPGs
data/interim/                    letterboxed + segmented cache
data/processed/                  tile bags, feature tables
src/                             pipeline stages, numbered
configs/                         run configs
outputs/manifests/               manifest.csv  <- single source of truth
outputs/reports/                 per-phase markdown reports
outputs/checkpoints/             model weights
```

Run order:

```bash
python src/00_build_manifest.py
```

---

## Open items

| # | Item | Blocks | Owner |
|---|---|---|---|
| **1** | **Reference standard** — culture / KOH / smear / confocal for Dataset9+14. Who assigned the organism, on what evidence, and when relative to imaging? | every clinical claim | clinical team |
| 2 | **2 patients carry both labels** (4 images). Mixed infection, uncertain, or revised diagnosis? | label integrity | clinical team |
| 3 | **Dataset15 + Dataset18** (212 fungal images, 811 `feathery_margin` polygons) — not on this machine | Phase 6 | data team |
| 4 | **Dataset22** (110 bacterial/fungal) — unused, would grow the cohort ~16% | cohort size | data team |
| 5 | **Vision Centre coverage is 11 images** (3 bacterial / 8 fungal). The deployment target is rural Vision Centres but training is ~98% tertiary-centre capture. | external validity | data team |
| 6 | Clinical covariates at intake — vegetative trauma, contact lens, symptom duration, prior steroid | Phase 4+ | clinical team |
