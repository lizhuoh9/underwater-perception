# 去除测试集泄漏后的复评

同一批锁定模型、阈值与已编译库，在去掉与标定/训练图近重复或同场景的 TEST 图后重新计分；没有重新训练或调参。排除名单只依据图像内容（感知哈希、颜色直方图和目视复核），未使用标注或逐图得分。

## 推荐方案（阈值 57 → 71）

| 测试子集 | 张数 | 漏检率 57 → 71 | 误检率 57 → 71 | 漏检增加（95% 区间） | 达标概率 | 逐图 改善/退步/持平 |
|---|---:|---:|---:|---:|---:|---:|
| 全部 | 110 | 5.93% → 7.74% | 27.44% → 21.52% | +1.81（+1.38, +2.30） pp | 79% | 71/34/5 |
| 去掉先前报告的 7 张 | 103 | 5.83% → 7.61% | 28.24% → 22.03% | +1.78（+1.35, +2.28） pp | 82% | 67/32/4 |
| 去掉全部近重复 | 98 | 6.16% → 7.88% | 28.07% → 22.16% | +1.73（+1.28, +2.23） pp | 87% | 64/30/4 |
| 再去掉已发现的同场景 | 74 | 5.92% → 7.57% | 31.43% → 24.84% | +1.66（+1.09, +2.26） pp | 87% | 46/24/4 |

预设目标：误检率下降，且漏检率增加不超过 2 个百分点。区间与达标概率来自按图片重抽样的自助法（10000 次，种子 20261022）；漏检增加接近 2 个百分点，单次点估计的“达到/未达到”不稳定。

## 各方法两类全局 mIoU

| 方法 | 全部（110 张） | 去掉先前报告的 7 张（103 张） | 去掉全部近重复（98 张） | 再去掉已发现的同场景（74 张） |
|---|---:|---:|---:|---:|
| UDCP 固定阈值 RGB888 | 0.4915 | 0.4960 | 0.5009 | 0.4896 |
| UDCP-Otsu RGB888 | 0.5244 | 0.5295 | 0.5345 | 0.5012 |
| UDCP-Otsu RGB565 | 0.5297 | 0.5349 | 0.5402 | 0.5086 |
| 原森林，阈值 57 | 0.7280 | 0.7231 | 0.7198 | 0.7025 |
| 原森林，阈值 71（推荐） | 0.7538 | 0.7512 | 0.7470 | 0.7332 |
| 16×8 森林，阈值 72 | 0.7574 | 0.7552 | 0.7504 | 0.7360 |

## 去掉已发现泄漏后（74 张）

| 方法 | 漏检 FNR | 水体误检 FPR | 两类全局 mIoU | 逐图平均 mIoU |
|---|---:|---:|---:|---:|
| UDCP-Otsu RGB565 | 31.23% | 33.41% | 0.5086 | 0.5160 |
| 原森林，阈值 57 | 5.92% | 31.43% | 0.7025 | 0.6342 |
| 原森林，阈值 71（推荐） | 7.57% | 24.84% | 0.7332 | 0.6624 |
| 16×8 森林，阈值 72 | 8.09% | 23.77% | 0.7360 | 0.6672 |

16×8 主实验相对原森林：漏检 +2.18（+1.36, +3.08） 个百分点、误检 -7.66 个百分点，达标概率 35%。

## 排除的 TEST 图

近重复 12 张：同一照片的重编码、缩放、重新裁剪，或相邻视频帧。同场景 24 张：同一段视频或同一次拍摄的不同时刻。先前报告的 7 张均属近重复；完整筛查另发现 d_r_122_、f_r_1276_、f_r_93_、n_l_18_、n_l_3_。其中 3 张（d_r_379_、f_r_1452_、w_r_101_）只有低置信的同场景判断，按保守原则排除。

| TEST 图 | 类别 | 最接近的邻居 | 所在集合 | 哈希距离 | 颜色重合 | 说明 | 在原 512 张训练集 |
|---|---|---|---|---:|---:|---|---|
| d_r_122_ | near_duplicate | d_r_710_ | train | 501 | 0.63 | same snorkeler and device, near-identical frame | 是 |
| d_r_384_ | near_duplicate | d_r_541_ | train | 142 | 0.95 | adjacent frame of the same ROV clip | 是 |
| f_r_1276_ | near_duplicate | n_l_94_ | train | 201 | 0.68 | same photo, recropped | 是 |
| f_r_1892_ | near_duplicate | f_r_1827_ | train | 19 | 0.99 | 哈希判定 | 否 |
| f_r_403_ | near_duplicate | n_l_80_ | train | 2 | 1.00 | 哈希判定 | 否 |
| f_r_93_ | near_duplicate | f_r_1664_ | train | 493 | 0.76 | same stingray, near-identical frame | 否 |
| n_l_100_ | near_duplicate | f_r_953_ | train | 81 | 0.98 | 哈希判定 | 是 |
| n_l_102_ | near_duplicate | f_r_2033_ | train | 2 | 0.99 | 哈希判定 | 否 |
| n_l_18_ | near_duplicate | w_r_38_ | train | 293 | 0.64 | same boat deck and depth overlay, near-identical frame | 否 |
| n_l_26_ | near_duplicate | w_r_32_ | train | 0 | 1.00 | 哈希判定 | 是 |
| n_l_3_ | near_duplicate | f_r_1208_ | train | 474 | 0.71 | same moray eel and urchin, near-identical frame | 否 |
| n_l_79_ | near_duplicate | f_r_615_ | train | 4 | 0.99 | 哈希判定 | 否 |
| d_r_166_ | same_scene | d_r_217_ | train | 239 | 0.76 | same manta ray and diver encounter | 是 |
| d_r_277_ | same_scene | d_r_689_ | train | 468 | 0.82 | same AUV over the same reef | 是 |
| d_r_345_ | same_scene | d_r_712_ | train | 378 | 0.77 | same diver in the same green water | 是 |
| d_r_379_ | same_scene | d_r_549_ | train | 411 | 0.60 | same snorkelling session; lower confidence | 否 |
| d_r_386_ | same_scene | n_l_16_ | train | 480 | 0.71 | same snorkelling session; lower confidence | 否 |
| d_r_465_ | same_scene | d_r_695_ | train | 361 | 0.65 | same diver with the same red device | 否 |
| d_r_47_ | same_scene | d_r_113_ | train | 460 | 0.66 | same diver with the same red device | 是 |
| d_r_547_ | same_scene | d_r_310_ | train | 311 | 0.94 | same diver over the same reef | 否 |
| d_r_602_ | same_scene | d_r_17_ | train | 417 | 0.63 | same snorkelling session with a device | 是 |
| f_r_106_ | same_scene | f_r_1527_ | train | 451 | 0.67 | same fish school over the same reef | 是 |
| f_r_1165_ | same_scene | n_l_58_ | train | 333 | 0.83 | same angelfish group over the same seafloor | 是 |
| f_r_1199_ | same_scene | f_r_220_ | train | 308 | 0.92 | same fish school over the same reef | 是 |
| f_r_1229_ | same_scene | f_r_1884_ | train | 408 | 0.74 | same dolphin pod by the same reef wall | 否 |
| f_r_1282_ | same_scene | f_r_1301_ | train | 351 | 0.90 | same Moorish idol clip | 是 |
| f_r_1298_ | same_scene | f_r_973_ | train | 318 | 0.89 | same coral scene | 是 |
| f_r_1303_ | same_scene | f_r_1857_ | train | 380 | 0.88 | same tiger shark clip | 否 |
| f_r_1410_ | same_scene | f_r_613_ | train | 430 | 0.68 | same red coral reef | 是 |
| f_r_1452_ | same_scene | f_r_1083_ | train | 428 | 0.68 | probably the same reef-shark dive; lower confidence | 是 |
| f_r_1737_ | same_scene | f_r_1527_ | train | 458 | 0.74 | same fish school over the same reef | 是 |
| f_r_1782_ | same_scene | f_r_536_ | train | 265 | 0.89 | same reef shark clip | 否 |
| f_r_1965_ | same_scene | f_r_295_ | train | 426 | 0.81 | same reef, fish species and light | 是 |
| f_r_263_ | same_scene | f_r_1239_ | train | 404 | 0.80 | same hawksbill turtle | 否 |
| w_r_101_ | same_scene | w_r_206_ | train | 468 | 0.72 | possibly the same aircraft wreck; lower confidence | 是 |
| w_r_4_ | same_scene | w_r_81_ | train | 431 | 0.64 | probably the same wreck; lower confidence | 否 |

最后一列指是否有邻居在训练原森林（阈值 57/71）的 512 张图中（20/36 张）；36/36 张至少有一个邻居在 16×8 森林的 1395 张训练集中，6 张另有标定集邻居。

## 筛查深度与残留风险

筛查条件：哈希距离不超过 260/1024，或颜色直方图重合不低于 0.60（TEST 配对）/0.70（标定配对），不再设哈希上限。各泄漏 TEST 图按最先能找到它的条件计数：哈希 9 张，颜色 ≥0.75 15 张，0.70–0.75 6 张，0.65–0.70 4 张，0.60–0.65 2 张。每降一档新增的泄漏图逐渐减少但没有归零，0.60 以下很可能仍有同段视频的帧。

## 尚未解决的问题

- 80 张标定图中有 24 张在训练集里有邻居（近重复 7、同场景 17）。阈值 57、71、72 都在这批标定图上选出，本次没有重新标定。
- SUIM 没有视频或拍摄批次编号，筛查只能找出画面或颜色接近的帧；本表最后一个子集是“去掉已发现泄漏”，不能证明场景级独立。
- 哈希无法判定的配对均已目视复核；复核人、日期和逐条决定见 [data/near_duplicate_review.json](../data/near_duplicate_review.json)，可修改后生成新的审计清单。
- 真正独立的检验仍需新的盲测集，最好是自己的 OV2640 实拍数据，并按拍摄批次划分。

## 一致性核查

- 全部 110 张图、6 种方法的逐图混淆计数与此前各轮记录完全一致；三个已编译库的 C 分数和掩膜与 Python 参考逐像素一致。
- 模型、阈值、导出头文件、DLL、数据清单、筛查代码与复核记录均按记录的哈希校验；泄漏名单由审计中的配对决定重新推导。

[审计清单](../data/dedup_manifest.json) · [目视复核记录](../data/near_duplicate_review.json) · [逐图结果](per_image.csv) · [结果 JSON](summary.json)

![排除的 TEST 图（左）与最接近的标定/训练图（右）](leak_pairs.jpg)
