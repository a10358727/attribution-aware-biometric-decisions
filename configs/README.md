# Training configurations of 160 fingerprint detector runs

The records comprise 120 principal detector runs and 40 supplemental NPR runs, with ten seeds per
configuration (16 detector × protocol × policy configurations). One directory per trained run, copied
unchanged from the authors' run records:

- `config_resolved.yaml` — the fully resolved training configuration (data manifest path relative to the
  authors' data root, normalization arm `data.norm` where present — runs without the key are legacy-arm
  runs that predate the key and used the default, per-image min–max; model type; epochs; batch size;
  learning rate).
- `meta.json` — seed, epochs, selection window, the selected epoch and the validation AUCs (`clean`,
  `degraded`) of the selected checkpoint. Checkpoint selection used validation data only.
- `metrics.csv` — per-epoch training loss and validation AUC.

Layout: `configs/<EXPERIMENT>/<run>_s<seed>/`, where `<EXPERIMENT>` is the authors' internal experiment
label of the run. Mapping to the paper's detector configurations (the NPR rows are the 40 supplemental
runs; all other rows are the 120 principal runs):

| Detector (`--model-type` in `inference/score_patch.py`) | Protocol | Policy | Run directory |
|---|---|---|---|
| ResNet-18, modified stem (`resnet18_e0`) | P1 | legacy `minmax` | `E009/bl_resnet18_e0_tp1_s*` |
| | P2 | legacy `minmax` | `E009/bl_resnet18_e0_tp2_s*` |
| | P1 | fixed `div255` | `E027/e027_resnet18_e0_tp1_div255_s*` |
| | P2 | fixed `div255` | `E027/e027_resnet18_e0_tp2_div255_s*` |
| Benjamin et al. base detector, BCE + SupCon (`bcesupcon`) | P1 | legacy | `E025/bl_bcesupcon_tp1_s*` |
| | P2 | legacy | `E025/bl_bcesupcon_tp2_s*` |
| | P1 | fixed | `E027/e027_bcesupcon_tp1_div255_s*` |
| | P2 | fixed | `E027/e027_bcesupcon_tp2_div255_s*` |
| NPR, residual input (`npr`), supplemental | P1 | legacy | `E003/bl_npr_tp1_s*` |
| | P2 | legacy | `E027/e027_npr_tp2_minmax_s*` |
| | P1 | fixed | `E027/e027_npr_tp1_div255_s*` |
| | P2 | fixed | `E027/e027_npr_tp2_div255_s*` |
| D3 on CLIP ViT-L/14, 84-px crop (not supported by this code; see README) | P1 | legacy | `E026/bl_d3_84_tp1_s*` |
| | P2 | legacy | `E027/e027_d3_84_tp2_minmax_s*` |
| | P1 | fixed | `E027/e027_d3_84_tp1_div255_s*` |
| | P2 | fixed | `E027/e027_d3_84_tp2_div255_s*` |

No file here contains an image, a subject index or a per-image score. The training data themselves are
not released (see `PUBLIC_RELEASE_EXCLUSIONS.md`).
