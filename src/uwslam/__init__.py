"""uwslam: monocular-camera SLAM for a small underwater robot.

A portable C core runs bit-identically on a PC and an ESP32-S3 (firmware/esp32s3_mono_vo): core/mono_vo*.c (visual
odometry), core/voxel_map.c (the continuing 3D map), core/obstacle_dir.c (the blocked directions from the water mask),
core/path_plan.c (where to head next, from the map and the mask), core/mono_map.c (a top-view obstacle grid) and
core/mono_cloud.c (the tracked features' sparse points). Around it: the binding and lake-clip check of the odometry
(mvo), the board stream test (mvo_stream), the maps, the directions and the plan (voxel_map, obstacle_dir, path_plan,
mono_map, mono_cloud), and the OpenCV prototype the C port is compared with, the reference poses and the water masks
(experiments).
"""

__version__ = "0.1.0"
