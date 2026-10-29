# uwseg

小型水下机器人 OV2640 相机的水体／非水体像素分割。一个小型 int8 CNN 把一帧 160×120 RGB565 图像变成 80×60 掩膜：
相机看到的不是水（海底、岩石、礁石、鱼、潜水员、沉船、机器人）处为 1，看到水处为 0。ESP-DL 在 ESP32-P4 上运行它。
模型在 SUIM 水下照片上选定并计分，对照两个同样保留在本仓库的基线：水下暗通道先验（UDCP）和整数像素随机森林。
掩膜只说明哪里是水，不说明障碍物有多远。

[English](README.md)

```text
相机帧，160×120 RGB565
└── firmware/components/water_seg   ESP32-P4 上的 ESP-DL int8 CNN（模型文件：results/esp32p4_export）
    └── 80×60 掩膜，逐行存放：1 = 非水体，0 = 水体
```

## 主要结果

所有精度数字都是 SUIM TEST 照片上的像素比例，按照片原始分辨率计分，在电脑上计算。非水体是 SUIM 类别 1–7，水体是类别 0。
漏检率（FNR）是非水体像素被判成水的比例，误检率（FPR）是水体像素被判成非水体的比例，mIoU 是两类在全部像素上 IoU 的平均。
110 张 TEST 图在各轮研究中反复使用过；近重复审计发现其中 12 张是标定图或训练图的副本或相邻帧，另有 24 张是同一场景的其他画面，
因此最严格的子集保留 74 张。

**部署模型**是 [results/cnn_espdl_run_02](results/cnn_espdl_run_02) 的轻量 CNN：每帧 22.3 MMAC，39,569 个参数，16 个卷积层，
即 `cnn_espdl_run_01` 的普通卷积 CNN（69.7 MMAC）把 stride 1 的 3×3 卷积换成深度可分离卷积。结构只用 80 张标定图选出；
部署阈值的规则（标定集漏检不超过 5%）在训练前登记，训练后按它得到 int8 阈值码 +3，并在打开 TEST 之前锁定；TEST 只计分一次。最严格子集上的结果
（[报告](results/cnn_espdl_run_02/evaluation/RESULTS.md)）：

| 方法 | 漏检（FNR） | 误检（FPR） | mIoU |
|---|---:|---:|---:|
| 轻量 CNN int8，部署阈值 +3 | 6.70% | 8.34% | 0.8588 |
| 轻量 CNN 浮点参考，部署阈值 | 7.84% | 7.56% | 0.8530 |
| 轻量 CNN int8，对照阈值 +10（森林的漏检上限 7.88%） | 11.87% | 5.05% | 0.8308 |
| 普通卷积 CNN int8，阈值 +5（看过它的 TEST 分数后选定） | 6.92% | 7.29% | 0.8644 |
| 普通卷积 CNN int8，锁定阈值 +11 | 11.12% | 4.56% | 0.8417 |
| 像素随机森林，阈值 71（推荐的森林） | 7.57% | 24.84% | 0.7332 |
| 像素随机森林，阈值 57 | 5.92% | 31.43% | 0.7025 |
| UDCP，逐帧 Otsu 阈值，RGB565 输入 | 31.23% | 33.41% | 0.5086 |

- 同为 5% 上限时，轻量 CNN 比普通卷积 CNN 漏检少 0.22 个百分点（95% 自助区间 −0.83～+0.45），误检多 1.06 个百分点
  （+0.03～+2.21）。比推荐的森林误检少 16.50 个百分点（−21.72～−12.10）。全部 110 张上漏检 5.83%、误检 7.58%（mIoU 0.8739）。
- int8 各行是 ESP-PPQ 对当时按 ESP32-S3 量化的计算图的仿真。[CAVEATS.md](results/cnn_espdl_run_02/CAVEATS.md)
  限定了这些数字的解读：5% 上限是看过普通卷积 CNN 的 TEST 分数后才定的，锁定的轻量权重在标定集上超出了选结构规则的余量，
  逐类比例所依据的图很少（机器人只出现在 5 张图里）。
- 在 ESP32-P4 上，同一个浮点模型被重新量化（[results/esp32p4_export](results/esp32p4_export)），沿用部署阈值码 +3、
  输出指数 −3；这个阈值是在 S3 计算图的标定曲线上选出的。**P4 int8 计算图在 TEST 上的精度尚未测量**；板上只核对过一张自检帧。

**基线。** 全部 110 张上，整数像素随机森林（12 棵深度 6 的树，每像素 16 个 RGB565 颜色与纹理特征）在相同 RGB565 输入下
漏检 5.93%（UDCP-Otsu 为 29.37%），误检 27.44%（UDCP-Otsu 为 31.76%）
（[报告](results/forest_comparison/evaluation/RESULTS.md)）。把阈值从 57 提到 71（这是在 TEST 计分前就在标定集上冻结的对照），
误检降到 21.52%，漏检升到 7.74%（[报告](results/forest_lowfp/evaluation/RESULTS.md)；[去泄漏复评](results/evaluation_dedup/RESULTS.md)）。
在 56 张没有训练集邻居的标定图上，森林的阈值规则仍选出 57、71 和 72（[复核](results/calibration_recheck/RESULTS.md)）。

**板上结果。** 固件的 demo（[firmware/water_seg_demo](firmware/water_seg_demo)）先运行 ESP-DL 自检，再在每种运行配置下
把一张存储的标定帧分割 20 次。部署模型在 360 MHz 的 ESP32-P4 上的实测（[日志](results/esp32p4_export/board_timing.log)），
每帧耗时（预处理 + 推理 + 阈值）：

| 特征图位置 | 单核 | 双核 | 模型内存（内部 RAM + PSRAM） |
|---|---:|---:|---:|
| PSRAM | 47.0 ms | 43.8 ms | 8 KB + 312 KB |
| 内部 RAM 至多 128 KB，其余 PSRAM | 20.4 ms | 15.1 ms | 136 KB + 125 KB |
| 内部 RAM 至多 256 KB，其余 PSRAM | 15.5 ms | 10.5 ms | 264 KB + 50 KB |

ESP-DL 自检通过；每种配置下，板上对存储帧给出的掩膜在两种字节序下都与电脑端 int8 参考一致。最快配置的 10.5 ms 中，
预处理 1.4 ms、推理 9.0 ms、阈值 0.1 ms（每秒 95 帧）。

历史数据：在 240 MHz 的 ESP32-S3 上（特征图至多 128 KB 放内部 RAM、双核），按 S3 量化的同一模型把电脑逐张发来的全部 110 张
TEST 图分割完，平均每帧 27.46 ms，板上每张掩膜都与电脑端 int8 参考一致（[报告](results/board_eval_run02/RESULTS.md)）。
S3 固件和它的传图工具不在本仓库中。

## 仓库结构

```text
.
├── src/uwseg/                  Python 包（python -m uwseg.<模块>）
│   ├── paths.py                数据、源码和结果的位置
│   ├── prepare_data.py, prepare_forest_training.py, prepare_lowfp_training.py
│   │                           固定版本的 SUIM 压缩包、数据划分及其清单
│   ├── near_duplicates.py, prepare_dedup_split.py
│   │                           按解码像素的近重复审计与去泄漏 TEST 子集
│   ├── udcp.py, validate.py    UDCP 基线、指标、标定及其评估
│   ├── pixel_forest.py, forest_native.py, train_pixel_forest.py, optimize_pixel_forest.py
│   │                           整数像素随机森林：特征、训练、低误检一轮、C 接口
│   ├── compare_pixel_forest.py, compare_lowfp.py, verify_lowfp_control.py, compare_dedup.py, recheck_calibration.py
│   │                           森林的锁定评估
│   ├── cnn_seg.py, cnn_lite.py ESP-DL CNN：输入处理、普通卷积与轻量结构
│   ├── train_cnn_espdl.py, train_cnn_lite.py
│   │                           训练、按 ESP32-P4 做 ESP-PPQ 量化、在标定集上锁定
│   ├── compare_cnn_espdl.py, compare_cnn_lite.py, export_cnn_threshold_variant.py
│   │                           CNN 的锁定 TEST 评估
│   └── export_p4.py            把锁定的运行量化给 ESP32-P4：固件嵌入的文件
├── tests/                      unittest 测试
├── embedded/                   像素随机森林的 C99 实现（基线的 MCU 实现，见其 README）
├── firmware/
│   ├── components/water_seg/   ESP-IDF 组件：模型嵌入 Flash，输入 RGB565，输出掩膜
│   └── water_seg_demo/         ESP32-P4 工程：自检与测速，不接相机
├── data/                       数据集放这里（不提交），旁边是已提交的来源说明、目视复核记录和冻结的近重复审计
└── results/                    已提交的证据，每个结果集一个目录（见下文“结果”）
```

## 安装

需要 Python 3.10 或更新版本（CNN 工具受 ESP-PPQ 限制为 3.10–3.12）。所有命令都在本目录（即本 README 所在的项目根目录）运行：

```bash
pip install -e .                 # numpy、Pillow：数据准备、UDCP、森林推理
pip install -e ".[cnn]"          # torch、esp-ppq、onnx：CNN 的训练、评估和导出
pip install -e ".[forest]"       # scikit-learn：训练像素随机森林
python -m unittest discover -s tests -t .
```

`cnn` 依赖组固定了已提交运行所用的版本。那些运行用的是 torch 2.6.0 的 CUDA 12.4 版本（来自
`https://download.pytorch.org/whl/cu124`）；有 CUDA 时训练使用 CUDA。森林工具也会查找项目根目录下的本地包目录 `.deps/`。

装好 `cnn` 依赖组时，全新检出有 6 个测试会跳过：3 个需要森林的主机动态库（见 [embedded/README.md](embedded/README.md)），3 个需要准备好的数据。没有 torch 时 CNN 相关测试也会跳过。

## 数据

数据是 SUIM 水下照片及其语义掩膜，来自一个固定版本的第三方镜像（来源、划分规则与掩膜编码见 [data/SOURCES.md](data/SOURCES.md)）。
照片不在本仓库中。

```bash
python -m uwseg.prepare_data --download     # 174 MB 压缩包，校验 SHA-256；80 对标定、110 对 TEST
python -m uwseg.prepare_forest_training     # 前 512 对训练图
python -m uwseg.prepare_lowfp_training      # 全部 1,395 对训练图
```

每一步都按 SHA-256 校验压缩包和之前的清单，并拒绝覆盖内容不同的文件。CNN 的运行、去泄漏评估和标定复核使用近重复审计
`data/dedup_manifest.json`，它和生成它所依据的目视复核记录（`data/near_duplicate_review.json`）一起提交。审计按 SHA-256
固定上面三份清单，每个已提交的 CNN 运行又固定这份审计。审计还记录了生成它的筛查代码的 SHA-256，即整理成本包之前的研究脚本。
要确认本包的筛查能复现它，把重算的审计写到新文件：

```bash
python -m uwseg.prepare_dedup_split --output data/dedup_manifest_recomputed.json
```

它应当只在 `code_sha256` 上与已提交的审计不同。`compare_dedup` 和 `recheck_calibration` 会按本包重新核对这个字段，
因而拒绝已提交的审计；请给它们重算的那份。

## 训练

```bash
python -m uwseg.train_cnn_lite --architecture lite --exploration results/cnn_espdl_run_02/exploration --output <新目录>
python -m uwseg.train_cnn_espdl --output <新目录>                 # 普通卷积 CNN
python -m uwseg.train_pixel_forest --output <新目录>              # 第一版森林（需要 .[forest]）
python -m uwseg.optimize_pixel_forest --output <新目录>           # 低误检一轮
```

CNN 运行先写出 `experiment_plan.json` 再训练，然后写出 PyTorch 与 ONNX 模型、面向 ESP32-P4 的 ESP-PPQ 导出（`water_seg.espdl`、
原生计算图 `water_seg_native`）、固件头文件、`water_seg.compile_req.yml`、自检帧、标定曲线和 `LOCKED.json`；不打开任何 TEST 图。
轻量版在标定集上选两个阈值：部署阈值（漏检上限 5%）和对照阈值（森林的上限）。已提交的运行是按 ESP32-S3 量化的；新的运行按
ESP32-P4 量化，并在 P4 计算图上选阈值，所以不会复现已提交的 int8 数字。它的输入量化按“四舍六入五成双”取整，与 esp-dl 在 P4 上
的做法一致（S3 上是五入）。训练固定了随机种子，但已提交的权重出自一块 GPU，换硬件后未必能逐位复现。

## 评估

```bash
python -m uwseg.compare_cnn_lite --run <运行目录> --output <运行目录>/evaluation
python -m uwseg.compare_cnn_espdl --run <运行目录> --output <运行目录>/evaluation    # 普通卷积运行
python -m uwseg.validate --output <新目录>                                          # UDCP
```

`compare_cnn_lite` 先核对运行的锁定内容（产物、源码、输入、结构探索记录、两个阈值是否由标定曲线得出、头文件），再把标定图送进
重新加载的 int8 计算图复现，并按各基线自己的汇总核对它们的逐图计数。然后把该运行与普通卷积 CNN 的已记录结果
（`results/cnn_espdl_run_01`）以及 `results/evaluation_dedup` 中的森林和 UDCP 结果一起计分。所有工具都拒绝覆盖已有输出。

已提交的运行是整理成本包之前的研究脚本做的，锁定文件记录的是那些脚本的 SHA-256，所以用本包核对它们会报告源码已变。
CNN 运行的 ESP32-S3 导出文件也没有提交。因此要重跑某个已提交的评估，就得用本包做一次新的运行。森林的评估还记录了所用 C 动态库的
SHA-256，`compare_dedup` 只能用当时的二进制文件复现。用本包在同一份数据上重新计分，UDCP 评估和标定复核得到与已提交结果相同的数字，
重算的审计除代码哈希外与已提交的审计相同。

## 导出到 ESP32-P4

```bash
python -m uwseg.export_p4 --kernels firmware/water_seg_demo/managed_components/espressif__esp-dl/spec/kernels.yml \
    --output <新目录> [--run results/cnn_espdl_run_02]
```

导出用 ESP-PPQ 把运行的 ONNX 模型按 `esp32p4` 目标量化（标定输入与训练时相同的 128 张），写出 `water_seg.espdl`、
`water_seg_model_config.h`、`water_seg.compile_req.yml`（计算图的算子，以及 esp-dl `spec/kernels.yml` 中全部 P4 int8 卷积核）、
`water_seg_test_frame.h`（第一张标定帧和 P4 计算图对它给出的掩膜）、`export.json`，以及 ESP-PPQ 的转储文件 `water_seg.info` 和 `water_seg.json`。它需要准备好的数据和 esp-dl 的规格文件；
后者在 demo 第一次构建时由 ESP-IDF 下载到 `managed_components/`。阈值沿用运行锁定的阈值码，不在 P4 计算图上重新选择。
用相同输入再导出一次，得到与 `results/esp32p4_export` 相同的量化权重与指数、头文件和自检帧；`.espdl` 文件本身不同，
差在 ESP-PPQ 随机生成的测试值和张量的排列顺序。已提交的导出是研究阶段的导出脚本写的：它的 `export.json` 把阈值记为 `s3_threshold`，`compile_req.yml` 的注释写的是 `run_02`；本模块写的是 `locked_threshold` 和运行目录名。

## 固件

`firmware/components/water_seg` 把 `water_seg.espdl` 嵌入 Flash，并注册它的 `compile_req.yml`，让 esp-dl 保留模型需要的算子和卷积核。
`WaterSegModel::begin()` 按头文件核对模型的张量类型、形状和指数；`segment()` 用三张 256 项查找表把 RGB565 转成 int8 输入码
（即 esp-dl 的 NormQuant，但不链接 `ImagePreprocessor` 为其他各种像素格式带来的转换代码），再对输出做阈值。
esp-dl 3.3.12 由组件和工程的 `idf_component.yml` 固定。

在 ESP-IDF 5.3 或更新版本的环境中构建并烧录 demo（IDF 5.3 缺少 `MALLOC_CAP_SIMD`，demo 的 `CMakeLists.txt` 为此补了定义；
实际使用的固件应当用 IDF 5.4 或更新版本）：

```bash
idf.py -C firmware/water_seg_demo -B firmware/water_seg_demo/build -p <串口> flash monitor
idf.py -C firmware/water_seg_demo -B <构建目录> -D WATER_SEG_MODEL_DIR=<导出目录> build   # 换一个导出
```

`sdkconfig.defaults` 选定 ESP32-P4、360 MHz、PSRAM、16 MB Flash，控制台走芯片自带的 USB-Serial-JTAG 口。组件的模型目录默认是
`results/esp32p4_export`；因为 `WATER_SEG_MODEL_DIR` 会缓存在构建目录里，构建时会打印实际嵌入的模型路径和 SHA-256。

在别的固件里使用这个组件：在 `include(project.cmake)` 之前把 `firmware/components` 加入 `EXTRA_COMPONENT_DIRS`，在 `main` 的
`REQUIRES` 中加入 `water_seg`，并保持 PSRAM 开启：

```cpp
#include "water_seg_model.hpp"

static WaterSegModel water;
static uint8_t mask[WaterSegModel::kMaskWidth * WaterSegModel::kMaskHeight];

water.begin(WaterSegOptions{256 * 1024, true});  // P4 上实测最快的配置
water.segment(frame_rgb565, 160, 120, /*big_endian=*/true, mask);  // 每次一帧 160x120 RGB565
```

相机驱动给出的 RGB565 常是大端字节序；请先用色卡确认，因为字节序错了程序照样运行，只是分割结果很差。P4 上的相机采图不在本仓库中，也没有测试过。

## 板上评估

烧录 demo 并查看串口输出。每一轮先打印 CPU 频率和 PSRAM 容量、模型的阈值码和指数、ESP-DL 自检结果（期望 `pass`），然后对六种
运行配置分别打印每帧耗时及其中预处理、推理、阈值三段，存储帧在小端和大端字节序下与电脑端 int8 参考不同的掩膜格数（期望 0），
以及模型占用的内部 RAM 和 PSRAM；最后打印最快配置的逐层耗时。demo 每 5 秒重复一次。可与 `results/esp32p4_export/board_timing.log`
对照。这只核对了一张存储帧；P4 还没有跑过 TEST 集。

## 结果

| 目录 | 内容 | 生成工具 |
|---|---|---|
| [validation_results](results/validation_results) | 全部 110 张 TEST 图上的 UDCP：窗口与阈值在标定集上锁定，逐图指标 | `validate` |
| [forest_comparison](results/forest_comparison) | 第一版像素随机森林（12 棵树、深度 6、阈值 57）对 UDCP；候选模型与锁定文件 | `train_pixel_forest`、`compare_pixel_forest` |
| [forest_lowfp](results/forest_lowfp) | 低误检一轮：16×8 森林（阈值 72，未达预设目标）和只调阈值的对照（同样 12 棵树、阈值 71，推荐） | `optimize_pixel_forest`、`compare_lowfp`、`verify_lowfp_control` |
| [evaluation_dedup](results/evaluation_dedup) | 锁定的森林与 UDCP 结果在去泄漏 TEST 子集（110、98、74 张）上重新计分；CNN 对比的基线 | `compare_dedup` |
| [calibration_recheck](results/calibration_recheck) | 在 56 张没有训练集邻居的标定图上重新选择森林阈值 | `recheck_calibration` |
| [cnn_espdl_run_01](results/cnn_espdl_run_01) | 普通卷积 CNN：锁定文件、标定曲线，以及锁定阈值和事后 +5 阈值版本的评估 | `train_cnn_espdl`、`compare_cnn_espdl`、`export_cnn_threshold_variant` |
| [cnn_espdl_run_02](results/cnn_espdl_run_02) | 部署的轻量 CNN：锁定文件、浮点模型（`model.pt`、`water_seg.onnx`）、头文件、标定曲线、结构探索、评估、限制说明 | `train_cnn_lite`、`compare_cnn_lite` |
| [exploration_v3](results/exploration_v3) | 只用标定集尝试超过轻量 CNN：没有方案过关（报告与结果记录） | 未包含 |
| [board_eval_run02](results/board_eval_run02) | 历史数据：轻量 CNN 在 ESP32-S3 上跑全部 110 张 TEST 图 | 未包含 |
| [esp32p4_export](results/esp32p4_export) | 按 ESP32-P4 量化的轻量 CNN：固件嵌入的文件，以及板上测速日志 | `export_p4`（研究阶段的版本）、`water_seg_demo` |

多数报告是研究时写的中文。报告链接的一些文件不在本仓库中：由 SUIM 照片组成的图片和图集（用准备好的数据重跑相应工具即可重新生成）、S3 板上运行的逐图输出，以及探索用的脚本和权重。有两份报告把近重复审计链接为 `../data/dedup_manifest.json`，它现在位于项目根目录的 `data/dedup_manifest.json`。CNN 运行的 ESP32-S3 模型文件、编译好的森林动态库和 S3 编译记录也没有收录；锁定文件里仍列有它们的哈希。

## 局限

- SUIM 照片不是机器人的相机画面。水色、曝光、白平衡和防水窗都不同；阈值需要用机器人自己的 OV2640 画面按拍摄批次划分后复核。
- TEST 集在之前几轮中被看过，SUIM 也没有拍摄批次编号：去泄漏子集去掉的是审计发现的泄漏，不能证明场景级独立。
- P4 的 int8 计算图与测过精度的 S3 计算图不同（ESP-PPQ 对 P4 的卷积权重按通道量化，对 S3 按整层量化，
  见 [exploration_v3](results/exploration_v3/README.md)）。它在 TEST 上的精度，以及阈值是否应在它自己的标定曲线上重新选择，都还没有答案。
- 只用标定集的探索发现，ImageNet 预训练的 MobileNetV3-small 编码器在浮点下明显更好，但经过 S3 的整层量化后优势大部分消失（最好的 int8 版本没有达到选用余量）；在 P4 上还没有试过。
- 固件自检用的 `water_seg_test_frame.h` 是一张 SUIM 标定图的 160×120 RGB565 副本；再分发前请核对 SUIM 的使用条款（见 [data/SOURCES.md](data/SOURCES.md)）。
