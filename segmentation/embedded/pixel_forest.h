#ifndef PIXEL_FOREST_H
#define PIXEL_FOREST_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define PF_API __declspec(dllexport)
#else
#define PF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PF_FEATURE_COUNT 16
#define PF_MAX_WIDTH 160
#define PF_MAX_HEIGHT 120

typedef struct PixelForestNode {
    /* Global child indices; feature 255 identifies a leaf. */
    uint16_t left;
    uint16_t right;
    uint8_t feature;
    uint8_t threshold;
    uint8_t probability;
    uint8_t reserved;
} PixelForestNode;

/* Feature order: R, G, B, R/sum, G/sum, (R+2G+B)/4, saturation,
 * (R-G+255)/2, (G-B+255)/2, luma mean3, luma mean9, central gradient,
 * abs(luma-mean3), luma range5, R/sum mean9, G/sum mean9.
 * Fractions/saturation are scaled by 255. Division always floors; all local
 * neighborhoods replicate image edges. RGB565 expansion is q*255/31 or /63.
 */

enum {
    PF_OK = 0,
    PF_ERR_INVALID = -1,
    PF_ERR_ALIGNMENT = -2,
    PF_ERR_WORKSPACE = -3,
    PF_ERR_OVERLAP = -4
};

/* Returns zero for dimensions outside 1..160 by 1..120. */
PF_API size_t pf_workspace_bytes(int width, int height);

/*
 * All buffers are caller-owned. Workspace must be 4-byte aligned; RGB565 must
 * be uint16_t-aligned and contain packed R5 G6 B5 words (red is 0xf800).
 * Input, output(s), and the required workspace region must not overlap.
 * Image buffers contain width*height elements in row-major order. The feature
 * output contains width*height*16 bytes; prediction outputs each contain
 * width*height bytes. No function allocates memory or uses floating point.
 */
PF_API int pf_extract_features(const uint16_t *rgb565, int width, int height,
                              uint8_t *features, void *workspace,
                              size_t workspace_bytes);

/* Scores are rounded forest probabilities on 0..255, after optional median.
 * Masks contain 0 or 1, using score >= PF_THRESHOLD_Q from the model header.
 */
PF_API int pf_predict_rgb565(const uint16_t *rgb565, int width, int height,
                            uint8_t *scores, uint8_t *masks, void *workspace,
                            size_t workspace_bytes);

/* Rows contain count*16 bytes; scores contain count bytes. Buffers must be
 * nonnull and disjoint. Zero rows are accepted with nonnull pointers.
 */
PF_API int pf_predict_features(const uint8_t *rows, size_t count,
                              uint8_t *scores);

#ifdef __cplusplus
}
#endif

#endif
