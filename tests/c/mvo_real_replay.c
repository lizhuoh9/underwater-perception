/* Drive the real firmware's frame handlers over stdin/stdout with the real VO and maps.
 * This host-only harness mocks the ESP-IDF APIs, never connects to a physical board,
 * and uses the portable tracker kernels in place of the ESP32 vector assembly. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

typedef void *SemaphoreHandle_t;
typedef struct { int tx_buffer_size, rx_buffer_size; } usb_serial_jtag_driver_config_t;
#define portMAX_DELAY 0
#define pdMS_TO_TICKS(n) (n)
#define pdPASS 1
#define MALLOC_CAP_INTERNAL 1
#define MALLOC_CAP_8BIT 2
#define CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ 240
#define CONFIG_IDF_TARGET_ESP32S3 1 /* the S3's build: its vector kernels (forwarded to C below), the USB link */
#define CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG 1
#define ESP_ERROR_CHECK(x) (x)

int usb_serial_jtag_read_bytes(void *out, uint32_t size, int timeout)
{
    (void)timeout;
    return (int)fread(out, 1, size, stdin);
}
void usb_serial_jtag_vfs_use_driver(void) {}
int usb_serial_jtag_driver_install(const usb_serial_jtag_driver_config_t *config) { (void)config; return 0; }
uint32_t esp_cpu_get_cycle_count(void) { return 0; }
int64_t esp_timer_get_time(void) { return 0; }
void *heap_caps_aligned_alloc(unsigned a, unsigned b, unsigned c) { (void)a; (void)b; (void)c; return NULL; }
void *heap_caps_malloc(unsigned a, unsigned b) { (void)a; (void)b; return NULL; }
unsigned heap_caps_get_free_size(unsigned caps) { (void)caps; return 0; }
unsigned heap_caps_get_largest_free_block(unsigned caps) { (void)caps; return 0; }
SemaphoreHandle_t xSemaphoreCreateBinary(void) { return NULL; }
void xSemaphoreTake(SemaphoreHandle_t sem, int ticks) { (void)sem; (void)ticks; }
void xSemaphoreGive(SemaphoreHandle_t sem) { (void)sem; }
int xTaskCreatePinnedToCore(void (*fn)(void *), const char *name, int stack, void *arg, int priority,
                            void *handle, int core)
{ (void)fn; (void)name; (void)stack; (void)arg; (void)priority; (void)handle; (void)core; return pdPASS; }
unsigned uxTaskGetStackHighWaterMark(void *task) { (void)task; return 0; }
void vTaskDelay(int ticks) { (void)ticks; }

#include "../../firmware/esp32s3_mono_vo/main/main.c"

void mvo_pie_lk_template(mvo_window_t *w, const uint8_t *s, int stride, const int16_t q[4], int64_t g[3])
{ mvo_lk_template_c(w, s, stride, q, g); }
void mvo_pie_lk_sums(const mvo_window_t *w, const uint8_t *s, int stride, const int16_t q[4], int64_t sums[2])
{ mvo_lk_sums_c(w, s, stride, q, sums); }
void mvo_pie_block(uint8_t *out, int size, const uint8_t *img, int w, int h, const int origin[2])
{ mvo_block_c(out, size, img, w, h, origin); }

int main(void)
{
    static uint8_t pyramids[2][MVO_PYRAMID_BYTES] MVO_ALIGN16;
    static mvo_t state;
    static mm_map_t obstacle_map;
    static mc_map_t cloud_map;
    static vm_map_t voxel_map;
    char line[LINE_BYTES];
#ifdef _WIN32
    (void)_setmode(_fileno(stdin), _O_BINARY);
    (void)_setmode(_fileno(stdout), _O_BINARY);
#endif
    (void)setvbuf(stdout, NULL, _IONBF, 0);
    s_pyramid[0] = pyramids[0]; s_pyramid[1] = pyramids[1];
    s_state = &state; s_map = &obstacle_map; s_cloud = &cloud_map; s_voxels = &voxel_map;
    print_ready(); /* the firmware's own */
    while (fgets(line, sizeof line, stdin)) {
        size_t length = strlen(line);
        while (length && (line[length - 1] == '\n' || line[length - 1] == '\r')) line[--length] = '\0';
        if (strncmp(line, "selftest ", 9) == 0) selftest(line);
        else if (strncmp(line, "start ", 6) == 0) start(line);
        else if (strncmp(line, "frame ", 6) == 0) frame(line);
        else if (strcmp(line, "end") == 0) end();
        else if (strcmp(line, "cloud") == 0) cloud();
        else printf("error unknown command\n");
    }
    return 0;
}
