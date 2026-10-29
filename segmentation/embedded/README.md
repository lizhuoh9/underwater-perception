# Integer pixel forest in C

`pixel_forest.c` is the C99 implementation of the pixel-forest baseline (`uwseg.pixel_forest` is its Python
reference): 16 integer colour and texture features per RGB565 pixel, a quantized random forest, a 3×3 median and a
fixed threshold. It allocates no memory and uses no floating point. The deployed segmenter is the ESP-DL CNN
(`firmware/`); this forest is the baseline the CNN is compared against, and it is not part of the firmware.

- `pixel_forest.h`: public API, feature order and error codes.
- `pixel_forest.c`: RGB565 features and forest inference.
- `pixel_forest_model.h`: a generated model (`uwseg.pixel_forest.write_model`), found on the include path. The
  recommended one, 12 trees of depth 6 at threshold 71, is in `results/forest_lowfp/threshold_control/model`;
  `results/forest_comparison/model` holds the same trees at threshold 57, and `results/forest_lowfp/model` the
  larger 16 × 8 forest that missed the low false-positive round's goal. Do not mix them up, and do not edit a
  header's threshold by hand and keep quoting that model's results.

## The host library the Python tools load

The comparison tools and `tests/test_forest_native.py` load the forest as a shared library through ctypes
(`uwseg.forest_native`), built next to the model it embeds. With gcc (MinGW-w64 on Windows), from the project root
(the folder above `embedded/`):

```bash
gcc -std=c99 -O2 -Wall -Wextra -Werror -shared -static-libgcc -Iembedded -Iresults/forest_comparison/model \
    embedded/pixel_forest.c -o results/forest_comparison/pixel_forest.dll
```

Use `.so` instead of `.dll` outside Windows. The committed evaluations record the SHA-256 of the libraries they were
run with; a library built elsewhere differs, so the tools that check those hashes (`compare_dedup`) refuse it, while
the parity checks (`check_native`, the tests) only need the C and Python outputs to agree.

## Memory and calls

Call `pf_workspace_bytes(width, height)` once and allocate that many bytes, 4-byte aligned and byte-addressable,
outside the task stack; reuse the workspace for every frame. The workspace is
`align_up(7·W·H, 4) + 4·(W+9)·(H+9)` bytes: 221,604 bytes at 160×120. With the RGB565 input (38,400 bytes) and the
score and mask outputs (19,200 bytes each), the image buffers total 298,404 bytes, without the model array (11,992
bytes for the 12-tree forest), camera buffers, drivers, stacks or the system.

```c
int status = pf_predict_rgb565(rgb565, width, height, scores, masks, workspace, workspace_bytes);
```

Check `status == PF_OK`. `scores` are the median-filtered forest scores (0–255), `masks` 0 for water and 1 for
non-water. Input, outputs and workspace are row-major and must not overlap; images are 1–160 wide and 1–120 high. One
workspace must not be shared by concurrent calls. `pf_extract_features` and `pf_predict_features` exist for checking
against the Python reference; the feature output takes 16 bytes per pixel.

## RGB565 values and camera byte order

The API takes `uint16_t` values: the top 5 bits are red, the middle 6 green, the low 5 blue (pure red is `0xf800`).
A camera's byte stream must not be cast to `uint16_t*` unchecked: confirm the driver's byte order with a colour
target first. If a red pixel arrives as the bytes `f8 00`, assemble

```c
rgb565[i] = ((uint16_t)camera_bytes[2 * i] << 8) | camera_bytes[2 * i + 1];
```

and swap the other way if it arrives as `00 f8`. Check the frame's size and format before passing at most 160×120
pixels.

The output is a water / non-water classification of pixels; non-water includes the sea floor, and it is not a
measured collision distance. The forest has not been run on a board.
