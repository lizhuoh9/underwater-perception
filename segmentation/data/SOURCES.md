# Image and annotation provenance

This validation uses **SUIM** underwater photographs and paired semantic masks.
No remote dataset loader, repository Python file, or other downloaded code is
executed. `prepare_data.py` uses the Python standard library to verify and read
the ZIP archive and extracts only the selected image/mask pairs.

- [Official SUIM dataset page](https://irvlab.cs.umn.edu/resources/suim-dataset)
- [Paper: Semantic Segmentation of Underwater Imagery: Dataset and Benchmark](https://arxiv.org/abs/2004.01241), Islam et al., 2020.
- [Third-party Hugging Face mirror](https://huggingface.co/datasets/SatwikKambham/suim)
- [Pinned mirror revision](https://huggingface.co/datasets/SatwikKambham/suim/tree/0f07fb8db0e067f66061cdf18a04a87c7cf247d8)
- [Pinned ZIP download](https://huggingface.co/datasets/SatwikKambham/suim/resolve/0f07fb8db0e067f66061cdf18a04a87c7cf247d8/SUIM.zip)

The archive is 173,794,160 bytes. Required SHA256:

```text
5ab7bd0763f7e674e5949cd2fa9b77023c40cdeb0ac2fca9924380dd664d7fba
```

The official page describes **1,525 train_val pairs and 110 TEST pairs**. The
checksum-pinned mirror used here actually contains **1,488 train_val pairs and
110 TEST pairs**. This discrepancy is recorded rather than treating the mirror
as a verified complete copy of the official train_val set.

## Split and leakage controls

All 110 mirror TEST pairs are retained for evaluation. Calibration contains 80
train_val pairs, selected deterministically by sorting the SHA256 of each image's
ZIP member filename (UTF-8 bytes), with the member name as a tie-breaker.
Before selection, train_val images with file-byte SHA256 hashes matching any
TEST image are excluded. Selection and duplicate removal depend on image
filenames/bytes, not TEST annotation content. TEST labels must not be used to
select algorithm parameters.

`data/manifest.json` records every selected original archive member, image/mask
checksum, output path, split, mirror discrepancy, duplicate exclusions and
within-TEST exact-byte duplicates. This check detects exact files only, not
re-encoded copies, adjacent video frames, or visually similar scenes.

### Decoded-pixel near-duplicate audit

`prepare_dedup_split.py` adds a decoded-pixel audit on top of the exact-byte
checks, without changing any existing manifest, split or extracted file. It
compares a 1024-bit grayscale difference hash (also mirrored) and a coarse RGB
histogram for TEST x calibration, TEST x all 1,395 training images, and
calibration x training. A pair is screened when the hash differs by at most 260
bits or the colour histograms overlap by at least 0.60 (TEST pairs) or 0.70
(calibration pairs). Hash distances of at most 100 bits are treated as the same
photo; the other 350 screened pairs were reviewed visually, with each decision
recorded in `data/near_duplicate_review.json`.

The resulting `data/dedup_manifest.json` lists 12 TEST images that are
re-encoded, resized or recropped copies or adjacent frames of training images,
24 more TEST images that are other frames of the same clip or encounter, and 24
calibration images with training neighbours. It also defines the TEST subsets
without these images (110, 98 and 74 images). The colour gate was lowered until
each band added few new leaked TEST images; the last three bands still added 6,
4 and 2, so frames of the same clip below the gate very likely remain. SUIM has
no video or session IDs, so scene-level independence is not established.

## Mask palette and evaluation meaning

**Actual pinned archive:** all 190 selected masks were audited. They are
uncompressed 8-bit BMP images, decoded by Pillow as mode `L` (80 calibration,
110 TEST). The pixel values are scalar **class IDs 0-7**, with an identity
grayscale palette. Across the selected set every ID 0-7 occurs. For example,
class 4 appears as gray byte 4, or `(4, 4, 4)` after RGB conversion, not as
`(255, 0, 0)`. Read these mask values directly. `prepare_data.py` validates the
BMP header, first eight palette entries and pixel-ID range using only the
standard library, and records the encoding audit in the manifest. Do not decode
these low grayscale values by thresholding the RGB color bits.

**Official documented palette:** official mask colors encode three RGB bits.
For masks actually using that palette, decode a mask pixel as
`class_id = 4 * R_bit + 2 * G_bit + B_bit`, where a binary channel is represented
by 0 or 255 in the image. Read RGB explicitly; OpenCV loads color images as BGR
by default. This palette metadata differs from the encoding observed in the
pinned mirror's BMP files; it must not override inspection of the image bytes.

| ID | RGB bits | RGB bytes | Class |
|---:|:---:|:---:|---|
| 0 | 000 | 0, 0, 0 | Background / waterbody |
| 1 | 001 | 0, 0, 255 | Human divers |
| 2 | 010 | 0, 255, 0 | Aquatic plants / sea-grass |
| 3 | 011 | 0, 255, 255 | Wrecks / ruins |
| 4 | 100 | 255, 0, 0 | Robots / AUVs / ROVs / instruments |
| 5 | 101 | 255, 0, 255 | Reefs / invertebrates |
| 6 | 110 | 255, 255, 0 | Fish / vertebrates |
| 7 | 111 | 255, 255, 255 | Sea-floor / rocks |

These are **semantic annotations, not collision-distance ground truth**. Mapping
classes 1-7 to a single foreground mask evaluates separation of labeled scene
content from waterbody. It does not establish whether an object is close enough
to collide with the robot, whether the floor blocks its intended path, metric
depth accuracy, or navigation safety. SUIM-Net results must not be attributed to
the UDCP heuristic implemented in this workspace.

## Reuse and attribution

The mirror displays an MIT license tag. This is mirror metadata and is not
assumed to establish an MIT license for all original photographs, which the
paper describes as collected from multiple sources and experiments. Preserve
SUIM attribution and check the original dataset/source terms before publishing
or redistributing photographs. The local downloads and derivative panels are
used here for the requested algorithm validation.
