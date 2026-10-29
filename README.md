# underwater-perception

Camera perception for a small underwater robot on an ESP32-P4: a water / not-water segmentation model for its camera,
and a monocular-camera SLAM that turns the camera's frames and those masks into the robot's motion, a 3D map, the
blocked directions of the view and where to head next.

[中文说明](README.zh-CN.md)

| Folder | What it holds |
|---|---|
| [segmentation/](segmentation) | The segmentation model (`uwseg`): data preparation, training and locked evaluations on SUIM photographs in Python, its int8 export for the ESP32-P4 with ESP-DL, and the ESP-IDF component and app that run it. |
| [slam/](slam) | The SLAM (`uwslam`): a C core that gives the same results bit for bit on a PC and the ESP32-P4, Python tools that score it on a lake clip against a 3D reconstruction, and the firmware that streams a clip through the board. |

The two parts meet at the mask. For every camera frame the segmentation model gives a water / not-water mask, and the
SLAM's obstacle directions and path plan read it, as a 60×80 grid, beside the odometry and the map. The masks behind the
SLAM's results were made by the segmentation model on the lake clip and are committed with them
(`slam/results/lake_clip/mask_motion`).

Each folder is a project of its own, with its own README, Python package, tests and firmware. Install and run each
from inside its folder:

```bash
cd slam
pip install -e ".[experiments]"
python -m unittest discover -s tests -t .
```

```bash
cd segmentation
pip install -e ".[cnn]"
python -m unittest discover -s tests -t .
```

Neither the lake clip, its reconstruction nor the SUIM photographs are part of the repository; each README says how to
get or prepare them.
