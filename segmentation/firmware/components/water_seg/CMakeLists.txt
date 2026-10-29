# ESP-DL water/non-water segmenter. WATER_SEG_MODEL_DIR must hold the files one export locks together:
# water_seg.espdl, water_seg_model_config.h and water_seg.compile_req.yml (water_seg_test_frame.h is
# only used by the demo). uwseg.export_p4 writes them for the ESP32-P4.
if (CMAKE_BUILD_EARLY_EXPANSION)
    # Requirements pass runs in script mode, before -D cache values such as WATER_SEG_MODEL_DIR exist.
    idf_component_register(REQUIRES espressif__esp-dl PRIV_REQUIRES esp_timer)
    return()
endif()

# Default: the deployed lite CNN (results/cnn_espdl_run_02) quantized for the ESP32-P4.
set(WATER_SEG_MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../results/esp32p4_export"
    CACHE PATH "Directory of a uwseg.export_p4 export")
get_filename_component(model_dir "${WATER_SEG_MODEL_DIR}" ABSOLUTE)

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif ("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
else()
    message(FATAL_ERROR "ESP-DL component was not resolved by the ESP-IDF Component Manager")
endif()
# target_add_aligned_binary_data() finds its helper script through cmake_dir.
set(cmake_dir "${espdl_dir}/fbs_loader/cmake")
include(${cmake_dir}/utilities.cmake)

foreach(required water_seg.espdl water_seg_model_config.h water_seg.compile_req.yml)
    if (NOT EXISTS "${model_dir}/${required}")
        message(FATAL_ERROR "Missing ${model_dir}/${required}. Set WATER_SEG_MODEL_DIR to an export's folder.")
    endif()
endforeach()
# WATER_SEG_MODEL_DIR is cached, so a forgotten -D keeps the previous model; say which one is embedded.
file(SHA256 "${model_dir}/water_seg.espdl" espdl_sha256)
message(STATUS "water_seg: embedding ${model_dir}/water_seg.espdl (sha256 ${espdl_sha256})")

idf_component_register(
    SRCS "water_seg_model.cpp"
    INCLUDE_DIRS "include" "${model_dir}"
    REQUIRES espressif__esp-dl
    PRIV_REQUIRES esp_timer
)

target_add_aligned_binary_data(${COMPONENT_LIB} "${model_dir}/water_seg.espdl" BINARY)

# esp-dl keeps only the ops/kernels listed by registered compile_req files; without this
# list a firmware that registers another model's file would strip this model's kernels.
include(${espdl_dir}/cmake/compile_register.cmake)
esp_dl_compile_add_yml("${model_dir}/water_seg.compile_req.yml")
