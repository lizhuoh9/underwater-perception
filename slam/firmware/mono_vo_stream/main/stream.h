/*
 * stream.h - what the command loop (main.c) and the live build's camera input (live.c) share.
 */
#ifndef STREAM_H
#define STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "mono_vo.h"

#define FRAME_BYTES (MVO_WIDTH * MVO_HEIGHT)
#define CHUNK_WAIT_MS 5000 /* longest pause allowed inside a frame's bytes */

/* main.c */
uint32_t fnv(uint32_t hash, const void *data, size_t size);
int read_exact(uint8_t *out, int size);
/* mvo_t.parallel: part 1 on the second core, part 0 on this one. */
void run_both(void (*work)(void *context, int part), void *context);
/* The frame now in mvo_frame(stream_state()) without a mask: the run's first frame or a step, with the maps and the
 * plan, reported in one line that ends with extra. */
void process_frame(int index, const char *extra);
mvo_t *stream_state(void);
const uint8_t *stream_last_frame(void); /* the newest frame the odometry stepped on */
int stream_open(void);                  /* a run started and not yet ended */
int stream_started(void);               /* its first frame taken */

/* live.c: the live build's commands (CONFIG_SPIRAM) */
int live_init(void); /* the JPEG engine, the buffers and the worker; raises the calling loop's priority */
void live_start(void);
void live_jpeg(const char *line);
void live_wait_idle(void);
void live_grey(void);
void live_report(void);

#endif
