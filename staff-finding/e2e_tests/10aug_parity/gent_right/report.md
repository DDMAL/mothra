# Parity sweep: GentAnt1475_0017_AC_rightcrop.jpg

- page: `../image-sets/gent/right/GentAnt1475_0017_AC_rightcrop.jpg`
- detections: `../image-sets/gent/right/inference/corrected/GentAnt1475_0017_AC_rightcrop.txt` (class 0)
- baseline scale_unit: 15.0 px · image 926x1438

| variant | fits/boxes | staves | mode | distribution | cut px | vs baseline |
|---|---|---|---|---|---|---|
| baseline | 86/86 | 18 | 5 | 1:4,2:1,3:2,4:2,5:5,6:3,7:1 | 15.5 | — (baseline) |
| no_crop_seed | 86/86 | 18 | 5 | 1:4,2:1,3:2,4:2,5:5,6:3,7:1 | 15.5 | Δstaves +0 · Δmode +0 · matched 70 (yMAE 0.0 px, max 0.0) · lost 0 / new 0 |
| bgr_channel | 86/86 | 17 | 5 | 1:3,2:1,3:1,4:4,5:5,6:2,8:1 | 15.4 | Δstaves -1 · Δmode +0 · matched 68 (yMAE 0.34 px, max 2.9) · lost 2 / new 1 |
| working_copy | 86/86 | 18 | 5 | 1:4,2:1,3:2,4:2,5:5,6:3,7:1 | 15.5 | Δstaves +0 · Δmode +0 · matched 70 (yMAE 0.0 px, max 0.0) · lost 0 / new 0 |
| conf_landing | 42/42 | 14 | 1 | 1:5,2:4,3:2,4:1,5:1,7:1 | 35.9 | Δstaves -4 · Δmode -4 · matched 34 (yMAE 1.24 px, max 6.0) · lost 36 / new 1 |

- **baseline** — standalone settings: original image, RGB, crop seed
- **no_crop_seed** — SF-5/D1: fit_centerline without crop
- **bgr_channel** — SF-4/D5c: BGR array into RGB2GRAY binarize
- **working_copy** — SF-2/D5a: client-resize simulated working copy
- **conf_landing** — SF-1/D3: detection at landing default conf 0.5

Skipped: paco_layer (<urlopen error [Errno 61] Connection refused>); landing_exact (<urlopen error [Errno 61] Connection refused>)
