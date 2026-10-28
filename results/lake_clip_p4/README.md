# Lake clip results, p4 build configuration

The results of the tools in `results/lake_clip/` that build the core (all but `mask_motion` and the OpenCV prototype
`mono_vo`, whose outputs do not depend on it), on the same clip, with the core built in its `p4` configuration
(`MVO_CONFIG_P4` in `core/mono_vo.h`: 300 features, map points from rays at least 1.5 degrees apart), the one the
ESP32-P4 firmware runs. Made with `UWSLAM_CONFIG=p4`; each `summary.json` and board expectation records
`"config": "p4"`. The re-run lines in the `RESULTS.md` files are the tools' own and name the default folders: run them
with `UWSLAM_CONFIG=p4` set. The OpenCV prototype rows in `mono_vo_c/RESULTS.md` are the default set's: the prototype
tracks 250 corners whichever configuration the core is built in. The board check of this configuration on an ESP32-P4
has not been run yet, so `mono_vo_c/` holds no board records. See README.md, ESP32-P4, for the comparison with the
default configuration.
