"""uwseg: water / non-water pixel segmentation for a small underwater robot's OV2640 camera.

The deployed model is a small int8 CNN (cnn_lite, trained by train_cnn_lite) that ESP-DL runs on an ESP32-P4
(firmware/, exported by export_p4): one 160x120 RGB565 frame in, an 80x60 non-water mask out. Around it: the SUIM
data preparation and its near-duplicate audit (prepare_data, prepare_forest_training, prepare_lowfp_training,
near_duplicates, prepare_dedup_split), the baselines the CNN is compared with (udcp and validate; the integer pixel
forest pixel_forest, forest_native and embedded/, trained by train_pixel_forest and optimize_pixel_forest), and the
locked evaluations (compare_pixel_forest, compare_lowfp, verify_lowfp_control, compare_dedup, recheck_calibration,
compare_cnn_espdl, compare_cnn_lite, export_cnn_threshold_variant).
"""

__version__ = "0.1.0"
