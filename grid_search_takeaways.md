# Grid Search Takeaways

The grid search suggests that larger YOLO input sizes are important for this task. For manuscript layout detection, especially for thin staff lines and small musical details, `image_size = 1152` or `1280` appears more promising than smaller inputs such as `768`.

Model size does not seem to be the main bottleneck. `yolo11s` and `yolo11m` are both worth keeping, while `yolo11n` can probably be dropped from the next round.

The augmentation settings should be more conservative for geometric distortion. Small or no rotation works better, so `degrees` should likely stay around `0`, `1`, or `2`, rather than using stronger rotation such as `5`.

Mosaic augmentation still seems useful. The next round should focus on stronger mosaic settings such as `mosaic = 0.75` or `1.0`, while also testing whether `close_mosaic = 30` or `50` improves late-stage training.

The most reasonable next search space is narrower: use `image_size = 1152/1280`, `batch_size = 4/8`, `model_size = s/m`, and a learning rate roughly between `3e-4` and `1.5e-3`.
