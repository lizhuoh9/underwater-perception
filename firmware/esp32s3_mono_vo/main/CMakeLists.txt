set(SOURCES "main.c" "../../../core/mono_vo.c" "../../../core/mono_vo_numerics.c" "../../../core/mono_vo_tracking.c"
    "../../../core/mono_vo_homography.c" "../../../core/mono_vo_essential.c" "../../../core/mono_map.c"
    "../../../core/mono_cloud.c" "../../../core/voxel_map.c" "../../../core/obstacle_dir.c" "../../../core/path_plan.c")
if(CONFIG_IDF_TARGET_ESP32S3) # the tracker's vector kernels, in each chip's instructions; other chips run the C ones
    list(APPEND SOURCES "mvo_pie.S")
elseif(CONFIG_IDF_TARGET_ESP32P4)
    list(APPEND SOURCES "mvo_pie_p4.S")
endif()
idf_component_register(
    SRCS ${SOURCES}
    INCLUDE_DIRS "." "../../../core"
    REQUIRES esp_timer esp_driver_usb_serial_jtag esp_driver_uart vfs
)

# Same floating-point contract as the PC build (uwslam.core.CFLAGS): no fused multiply-add, so results match bit for bit.
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)

# The board reports which cores it runs (sources and headers), so the host can refuse a stale build. MVO_CORE_SHA256 is
# the SHA-256 of the odometry's implementation files back to back in this order, the list uwslam.mvo.CORE_FILES gives.
set(CORE "${CMAKE_CURRENT_LIST_DIR}/../../../core")
set(MVO_CORE_FILES mono_vo.c mono_vo_numerics.c mono_vo_tracking.c mono_vo_homography.c mono_vo_essential.c
    mono_vo_internal.h)
set(MVO_CORE_TEXT "")
foreach(name IN LISTS MVO_CORE_FILES)
    file(READ "${CORE}/${name}" text)
    string(APPEND MVO_CORE_TEXT "${text}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CORE}/${name}")
endforeach()
string(SHA256 MVO_CORE_SHA256 "${MVO_CORE_TEXT}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CORE}/mono_vo.h" "${CORE}/mono_map.c" "${CORE}/mono_map.h"
             "${CORE}/mono_cloud.c" "${CORE}/mono_cloud.h" "${CORE}/voxel_map.c" "${CORE}/voxel_map.h"
             "${CORE}/obstacle_dir.c" "${CORE}/obstacle_dir.h" "${CORE}/path_plan.c" "${CORE}/path_plan.h")
file(SHA256 "${CORE}/mono_vo.h" MVO_HEADER_SHA256)
file(SHA256 "${CORE}/mono_map.c" MM_CORE_SHA256)
file(SHA256 "${CORE}/mono_map.h" MM_HEADER_SHA256)
file(SHA256 "${CORE}/mono_cloud.c" MC_CORE_SHA256)
file(SHA256 "${CORE}/mono_cloud.h" MC_HEADER_SHA256)
file(SHA256 "${CORE}/voxel_map.c" VM_CORE_SHA256)
file(SHA256 "${CORE}/voxel_map.h" VM_HEADER_SHA256)
file(SHA256 "${CORE}/obstacle_dir.c" OD_CORE_SHA256)
file(SHA256 "${CORE}/obstacle_dir.h" OD_HEADER_SHA256)
file(SHA256 "${CORE}/path_plan.c" PP_CORE_SHA256)
file(SHA256 "${CORE}/path_plan.h" PP_HEADER_SHA256)
target_compile_definitions(${COMPONENT_LIB} PRIVATE MVO_CORE_SHA256="${MVO_CORE_SHA256}" MVO_HEADER_SHA256="${MVO_HEADER_SHA256}"
                           MM_CORE_SHA256="${MM_CORE_SHA256}" MM_HEADER_SHA256="${MM_HEADER_SHA256}"
                           MC_CORE_SHA256="${MC_CORE_SHA256}" MC_HEADER_SHA256="${MC_HEADER_SHA256}"
                           VM_CORE_SHA256="${VM_CORE_SHA256}" VM_HEADER_SHA256="${VM_HEADER_SHA256}"
                           OD_CORE_SHA256="${OD_CORE_SHA256}" OD_HEADER_SHA256="${OD_HEADER_SHA256}"
                           PP_CORE_SHA256="${PP_CORE_SHA256}" PP_HEADER_SHA256="${PP_HEADER_SHA256}")
