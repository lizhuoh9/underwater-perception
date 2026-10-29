# underwater-perception

在 ESP32-P4 上运行的小型水下机器人相机感知：一个针对机器人相机的水体／非水体分割模型，以及一个单目相机 SLAM——它把相机帧和
这些掩码变成机器人的运动、三维地图、视野里被挡住的方向，以及下一步该往哪里走。

[English](README.md)

| 文件夹 | 内容 |
|---|---|
| [segmentation/](segmentation) | 分割模型（`uwseg`）：在 SUIM 照片上做数据准备、训练和锁定评估的 Python 代码，用 ESP-DL 为 ESP32-P4 导出的 int8 模型，以及运行它的 ESP-IDF 组件和程序。 |
| [slam/](slam) | SLAM（`uwslam`）：在电脑和 ESP32-P4 上逐位相同的 C 核心，在湖中片段上对照三维重建给它打分的 Python 工具，以及把整段视频送进板子运行的固件。 |

两部分在掩码处衔接。分割模型为每一帧给出水体／非水体掩码，SLAM 的障碍方向和路径规划把它当作 60×80 的网格，与里程计和地图
一起读取。SLAM 结果所用的掩码是分割模型在湖中片段上算出来的，和这些结果一起提交在
`slam/results/lake_clip/mask_motion`。

每个文件夹都是独立的项目，有自己的 README、Python 包、测试和固件。安装和运行都在各自的文件夹里进行：

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

湖中片段、它的重建和 SUIM 照片都不在仓库里；各自的 README 说明了如何获取或准备。
