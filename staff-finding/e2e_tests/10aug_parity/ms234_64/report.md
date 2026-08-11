# Parity sweep: McGill_MS234-064.jpg

- page: `/Volumes/Expansion/script_sorter_mss/McGill_MS234/McGill_MS234-064.jpg`
- detections: `/Users/kyriebouressa/Documents/mothra/staff-finding/e2e_tests/10aug/ms234_64/yolo_txt/McGill_MS234-064.txt` (class 0)
- baseline scale_unit: 51.5 px · image 2154x2750

| variant | fits/boxes | staves | mode | distribution | cut px | vs baseline |
|---|---|---|---|---|---|---|
| baseline | 50/50 | 8 | 4 | 4:7,5:1 | 62.0 | — (baseline) |
| no_crop_seed | 50/50 | 8 | 4 | 4:7,5:1 | 62.0 | Δstaves +0 · Δmode +0 · matched 33 (yMAE 0.0 px, max 0.0) · lost 0 / new 0 |
| bgr_channel | 50/50 | 8 | 4 | 4:6,5:2 | 62.2 | Δstaves +0 · Δmode +0 · matched 33 (yMAE 0.48 px, max 5.1) · lost 0 / new 1 |
| working_copy | 50/50 | 8 | 4 | 4:7,5:1 | 62.0 | Δstaves +0 · Δmode +0 · matched 33 (yMAE 0.0 px, max 0.0) · lost 0 / new 0 |
| conf_landing | 9/9 | 1 | 9 | 9:1 | 309.5 | Δstaves -7 · Δmode +5 · matched 9 (yMAE 0.44 px, max 2.4) · lost 24 / new 0 |

- **baseline** — standalone settings: original image, RGB, crop seed
- **no_crop_seed** — SF-5/D1: fit_centerline without crop
- **bgr_channel** — SF-4/D5c: BGR array into RGB2GRAY binarize
- **working_copy** — SF-2/D5a: client-resize simulated working copy
- **conf_landing** — SF-1/D3: detection at landing default conf 0.5

Skipped: paco_layer (<urlopen error [Errno 61] Connection refused>); landing_exact (<urlopen error [Errno 61] Connection refused>)
