#include "mp_esp_dl_human_face_recognition.hpp"

#if CONFIG_HUMAN_FACE_FEAT_MODEL_IN_FLASH_RODATA
extern const uint8_t human_face_feat_espdl[] asm("_binary_human_face_feat_espdl_start");
static const char *path = (const char *)human_face_feat_espdl;
#elif CONFIG_HUMAN_FACE_FEAT_MODEL_IN_FLASH_PARTITION
static const char *path = "human_face_feat";
#else
#if !defined(CONFIG_BSP_SD_MOUNT_POINT)
#define CONFIG_BSP_SD_MOUNT_POINT "/sd"
#endif
#endif

#if CONFIG_HUMAN_FACE_FEAT_MODEL_IN_SDCARD
// A9: la microSD esta montada por MicroPython (driver SPI en Python), no por
// ESP-IDF, asi que fopen() de C no ve /sd. Se lee el archivo con las mismas
// funciones mp_open/mp_readinto que usa face.db y se deja el modelo en PSRAM.
static fbs::FbsModel *load_fbs_from_micropython_fs(const char *model_path)
{
    if (!mp_isfile(model_path)) {
        mp_raise_msg_varg(&mp_type_RuntimeError, MP_ERROR_TEXT("Modelo no encontrado: %s"), model_path);
    }

    mp_file_t *f = mp_open(model_path, "rb");
    if (!f) {
        mp_raise_msg_varg(&mp_type_RuntimeError, MP_ERROR_TEXT("No se pudo abrir el modelo: %s"), model_path);
    }

    uint8_t hdr[16];
    if (mp_readinto(f, hdr, sizeof(hdr)) != (mp_int_t)sizeof(hdr)) {
        mp_close(f);
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Modelo: encabezado incompleto"));
    }

    size_t hdr_len;
    if (memcmp(hdr, "EDL2", 4) == 0) {
        hdr_len = 16;
    } else if (memcmp(hdr, "EDL1", 4) == 0) {
        hdr_len = 12;
    } else {
        mp_close(f);
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Modelo: formato no soportado (se esperaba EDL1/EDL2)"));
    }

    uint32_t mode, size;
    memcpy(&mode, hdr + 4, 4);
    memcpy(&size, hdr + 8, 4);
    if (mode != 0) {
        mp_close(f);
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Modelo cifrado no soportado"));
    }
    if (size == 0 || size > (16u * 1024u * 1024u)) {
        mp_close(f);
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Modelo: tamano invalido"));
    }

    uint8_t *buf = (uint8_t *)dl::tool::malloc_aligned(16, size, MALLOC_CAP_SPIRAM);
    if (!buf) {
        mp_close(f);
        mp_raise_msg_varg(&mp_type_RuntimeError,
                          MP_ERROR_TEXT("Sin PSRAM para el modelo (%u bytes)"),
                          (unsigned int)size);
    }

    mp_seek(f, (off_t)hdr_len, MP_SEEK_SET);

    const size_t chunk = 64 * 1024;
    size_t total = 0;
    while (total < size) {
        size_t want = size - total;
        if (want > chunk) {
            want = chunk;
        }
        mp_int_t n = mp_readinto(f, buf + total, want);
        if (n <= 0) {
            break;
        }
        total += (size_t)n;
    }
    mp_close(f);

    if (total != size) {
        heap_caps_free(buf);
        mp_raise_msg_varg(&mp_type_RuntimeError,
                          MP_ERROR_TEXT("Modelo incompleto: %u de %u bytes"),
                          (unsigned int)total,
                          (unsigned int)size);
    }

    // Mismos parametros que usa fbs_loader.cpp (esp-dl) para el caso SDCARD:
    // sin cifrado, sin rodata_move, auto_free, sin param_copy.
    return new fbs::FbsModel(buf, size, fbs::MODEL_LOCATION_IN_SDCARD, false, false, true, false);
}
#endif

namespace human_face_recognition {

MFN::MFN(const char *model_name)
{
    m_model = nullptr;
    m_image_preprocessor = nullptr;
    m_postprocessor = nullptr;
#if !CONFIG_HUMAN_FACE_FEAT_MODEL_IN_SDCARD
    m_model =
        new dl::Model(path, model_name, static_cast<fbs::model_location_type_t>(CONFIG_HUMAN_FACE_FEAT_MODEL_LOCATION));
#else
    char sd_path[256];
    snprintf(sd_path,
             sizeof(sd_path),
             "%s/%s/%s",
             CONFIG_BSP_SD_MOUNT_POINT,
             CONFIG_HUMAN_FACE_FEAT_MODEL_SDCARD_DIR,
             model_name);
    m_fbs_model_owned = load_fbs_from_micropython_fs(sd_path);
    m_model = new dl::Model(m_fbs_model_owned);
    if (m_model->get_outputs().empty()) {
        // El grafo no se pudo construir (normalmente falta de memoria).
        delete m_model;
        m_model = nullptr;
        delete m_fbs_model_owned;
        m_fbs_model_owned = nullptr;
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("No se pudo construir el modelo (memoria insuficiente?)"));
    }
#endif
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor = new dl::image::FeatImagePreprocessor(
        m_model, {127.5, 127.5, 127.5}, {127.5, 127.5, 127.5}, DL_IMAGE_CAP_RGB_SWAP | DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
#elif CONFIG_IDF_TARGET_ESP32S3
    m_image_preprocessor = new dl::image::FeatImagePreprocessor(m_model, {127.5, 127.5, 127.5}, {127.5, 127.5, 127.5});
#endif
    m_postprocessor = new dl::feat::FeatPostprocessor(m_model);
    m_feat_len = m_model->get_outputs().begin()->second->get_size();
}

MFN::~MFN()
{
    if (m_fbs_model_owned) {
        // El buffer del modelo debe sobrevivir al modelo: se destruye primero
        // lo que lo usa y luego el buffer.
        delete m_model;
        m_model = nullptr;
        delete m_image_preprocessor;
        m_image_preprocessor = nullptr;
        delete m_postprocessor;
        m_postprocessor = nullptr;
        delete m_fbs_model_owned;
        m_fbs_model_owned = nullptr;
    }
}

} // namespace human_face_recognition

HumanFaceFeat::HumanFaceFeat(model_type_t model_type)
{
    switch (model_type) {
    case model_type_t::MFN_S8_V1:
#if CONFIG_HUMAN_FACE_FEAT_MFN_S8_V1
        m_model = new human_face_recognition::MFN("human_face_feat_mfn_s8_v1.espdl");
#else
        ESP_LOGE("human_face_feat", "human_face_feat_mfn_s8_v1 is not selected in menuconfig.");
#endif
        break;
    case model_type_t::MBF_S8_V1:
#if CONFIG_HUMAN_FACE_FEAT_MBF_S8_V1
        m_model = new human_face_recognition::MBF("human_face_feat_mbf_s8_v1.espdl");
#else
        ESP_LOGE("human_face_feat", "human_face_feat_mbf_s8_v1 is not selected in menuconfig.");
#endif
        break;
    default:
        ESP_LOGE("human_face_feat", "Unknown model type.");
    }
    m_feat_len = m_model->m_feat_len;
}

std::vector<mp_esp_dl::recognition::result_t> HumanFaceRecognizer::recognize(const dl::image::img_t &img,
                                                                      std::list<dl::detect::result_t> &detect_res)
{
    std::vector<std::vector<mp_esp_dl::recognition::result_t>> res;
    if (detect_res.empty()) {
        ESP_LOGW("HumanFaceRecognizer", "Failed to recognize. No face detected.");
        return {};
    } else if (detect_res.size() == 1) {
        auto feat = m_feat_extract->run(img, detect_res.back().keypoint);
        return query_feat(feat, m_thr, m_top_k);
    } else {
        auto max_detect_res =
            std::max_element(detect_res.begin(),
                             detect_res.end(),
                             [](const dl::detect::result_t &a, const dl::detect::result_t &b) -> bool {
                                 return a.box_area() > b.box_area();
                             });
        auto feat = m_feat_extract->run(img, max_detect_res->keypoint);
        return query_feat(feat, m_thr, m_top_k);
    }
}

esp_err_t HumanFaceRecognizer::enroll(const dl::image::img_t &img, std::list<dl::detect::result_t> &detect_res, const char *name, uint16_t *new_id)
{
    if (detect_res.empty()) {
        ESP_LOGW("HumanFaceRecognizer", "Failed to enroll. No face detected.");
        return ESP_FAIL;
    } else if (detect_res.size() == 1) {
        auto feat = m_feat_extract->run(img, detect_res.back().keypoint);
        return enroll_feat(feat, name, new_id);
    } else {
        auto max_detect_res =
            std::max_element(detect_res.begin(),
                             detect_res.end(),
                             [](const dl::detect::result_t &a, const dl::detect::result_t &b) -> bool {
                                 return a.box_area() > b.box_area();
                             });
        auto feat = m_feat_extract->run(img, max_detect_res->keypoint);
        return enroll_feat(feat, name, new_id);
    }
}
