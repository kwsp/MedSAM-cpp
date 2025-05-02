#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <map>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

void ggml_graph_compute_helper(std::vector<uint8_t> &buf, ggml_cgraph *graph,
                               int n_threads);

// default hparams (ViT-B SAM)
struct sam_hparams {
  int32_t n_enc_state = 768;
  int32_t n_enc_layer = 12;
  int32_t n_enc_head = 12;
  int32_t n_enc_out_chans = 256;
  int32_t n_pt_embd = 4;
  int32_t n_dec_heads = 8;
  int32_t ftype = 1;
  float mask_threshold = 0.f;
  float iou_threshold = 0.88f;
  float stability_score_threshold = 0.95f;
  float stability_score_offset = 1.0f;
  float eps = 1e-6f;
  float eps_decoder_transformer = 1e-5f;

  int32_t n_enc_head_dim() const { return n_enc_state / n_enc_head; }
  int32_t n_img_size() const { return 1024; }
  int32_t n_window_size() const { return 14; }
  int32_t n_patch_size() const { return 16; }
  int32_t n_img_embd() const { return n_img_size() / n_patch_size(); }

  std::vector<int32_t> global_attn_indices() const {
    switch (n_enc_state) {
    case 768:
      return {2, 5, 8, 11};
    case 1024:
      return {5, 11, 17, 23};
    case 1280:
      return {7, 15, 23, 31};
    default: {
      fprintf(stderr, "%s: unsupported n_enc_state = %d\n", __func__,
              n_enc_state);
    } break;
    };

    return {};
  }

  bool is_global_attn(int32_t layer) const {
    const auto indices = global_attn_indices();

    for (const auto &idx : indices) {
      if (layer == idx) {
        return true;
      }
    }

    return false;
  }
};

struct sam_point {
  float x;
  float y;
};

struct sam_box {
  float x1;
  float y1;
  float x2;
  float y2;
};

enum sam_prompt_type {
  SAM_PROMPT_TYPE_POINT = 0,
  SAM_PROMPT_TYPE_BOX = 1,
};

struct sam_prompt {
  sam_prompt_type prompt_type = SAM_PROMPT_TYPE_POINT;
  sam_point pt = {414.375f, 162.796875f};
  sam_box box = {368.0f, 144.0f, 441.0f, 173.0f};
};
struct sam_layer_enc {
  struct ggml_tensor *norm1_w;
  struct ggml_tensor *norm1_b;

  struct ggml_tensor *rel_pos_w;
  struct ggml_tensor *rel_pos_h;

  struct ggml_tensor *qkv_w;
  struct ggml_tensor *qkv_b;

  struct ggml_tensor *proj_w;
  struct ggml_tensor *proj_b;

  struct ggml_tensor *norm2_w;
  struct ggml_tensor *norm2_b;

  struct ggml_tensor *mlp_lin1_w;
  struct ggml_tensor *mlp_lin1_b;

  struct ggml_tensor *mlp_lin2_w;
  struct ggml_tensor *mlp_lin2_b;
};

struct sam_encoder_image {
  struct ggml_tensor *pe;

  struct ggml_tensor *proj_w;
  struct ggml_tensor *proj_b;

  struct ggml_tensor *neck_conv_0;
  struct ggml_tensor *neck_norm_0_w;
  struct ggml_tensor *neck_norm_0_b;
  struct ggml_tensor *neck_conv_1;
  struct ggml_tensor *neck_norm_1_w;
  struct ggml_tensor *neck_norm_1_b;

  std::vector<sam_layer_enc> layers;
};

struct sam_encoder_prompt {
  struct ggml_tensor *pe;

  struct ggml_tensor *not_a_pt_embd_w;
  std::vector<struct ggml_tensor *> pt_embd;

  struct ggml_tensor *no_mask_embd_w;
  // std::vector<struct ggml_tensor *> mask_down_w;
  // std::vector<struct ggml_tensor *> mask_down_b;
};

struct sam_layer_dec_transformer_attn {
  // q_proj
  struct ggml_tensor *q_w;
  struct ggml_tensor *q_b;

  // k_proj
  struct ggml_tensor *k_w;
  struct ggml_tensor *k_b;

  // v_proj
  struct ggml_tensor *v_w;
  struct ggml_tensor *v_b;

  // out_proj
  struct ggml_tensor *out_w;
  struct ggml_tensor *out_b;
};

struct sam_layer_dec_transformer {
  sam_layer_dec_transformer_attn self_attn;

  // norm1
  struct ggml_tensor *norm1_w;
  struct ggml_tensor *norm1_b;

  sam_layer_dec_transformer_attn cross_attn_token_to_img;

  // norm2
  struct ggml_tensor *norm2_w;
  struct ggml_tensor *norm2_b;

  // mlp.lin1
  struct ggml_tensor *mlp_lin1_w;
  struct ggml_tensor *mlp_lin1_b;

  // mlp.lin2
  struct ggml_tensor *mlp_lin2_w;
  struct ggml_tensor *mlp_lin2_b;

  // norm3
  struct ggml_tensor *norm3_w;
  struct ggml_tensor *norm3_b;

  // norm4
  struct ggml_tensor *norm4_w;
  struct ggml_tensor *norm4_b;

  sam_layer_dec_transformer_attn cross_attn_img_to_token;
};

struct sam_layer_dec_output_hypernet_mlps {
  // mlps_*.layers.0
  struct ggml_tensor *w_0;
  struct ggml_tensor *b_0;

  // mlps_*.layers.1
  struct ggml_tensor *w_1;
  struct ggml_tensor *b_1;

  // mlps_*.layers.2
  struct ggml_tensor *w_2;
  struct ggml_tensor *b_2;
};

struct sam_decoder_mask {
  std::vector<sam_layer_dec_transformer> transformer_layers;

  // trasnformer.final_attn_token_to_image
  sam_layer_dec_transformer_attn transformer_final_attn_token_to_img;

  // transformer.norm_final
  struct ggml_tensor *transformer_norm_final_w;
  struct ggml_tensor *transformer_norm_final_b;

  // output_upscaling.0
  struct ggml_tensor *output_upscaling_0_w;
  struct ggml_tensor *output_upscaling_0_b;

  // output_upscaling.1
  struct ggml_tensor *output_upscaling_1_w;
  struct ggml_tensor *output_upscaling_1_b;

  // output_upscaling.3
  struct ggml_tensor *output_upscaling_3_w;
  struct ggml_tensor *output_upscaling_3_b;

  // output_hypernetworks_mlps
  std::vector<sam_layer_dec_output_hypernet_mlps> output_hypernet_mlps;

  // iou_prediction_head.0
  struct ggml_tensor *iou_prediction_head_0_w;
  struct ggml_tensor *iou_prediction_head_0_b;

  // iou_prediction_head.1
  struct ggml_tensor *iou_prediction_head_1_w;
  struct ggml_tensor *iou_prediction_head_1_b;

  // iou_prediction_head.2
  struct ggml_tensor *iou_prediction_head_2_w;
  struct ggml_tensor *iou_prediction_head_2_b;

  // iou_token.weight
  struct ggml_tensor *iou_token_w;

  // mask_tokens.weight
  struct ggml_tensor *mask_tokens_w;
};

struct sam_state {
  struct ggml_tensor *embd_img;

  struct ggml_tensor *low_res_masks;
  struct ggml_tensor *iou_predictions;

  // struct ggml_tensor * tmp_save = {};

  struct ggml_context *ctx;

  // buffer for `ggml_graph_plan.work_data`
  std::vector<uint8_t> work_buffer;
  // buffers to evaluate the model
  std::vector<uint8_t> buf_compute_img_enc;

  std::vector<uint8_t> buf_compute_fast;

  ggml_gallocr_t allocr = {};
};

struct sam_params {
  int32_t seed = -1; // RNG seed
  int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

  std::string model = "models/sam-vit-b/ggml-model-f16.bin"; // model path
  std::string fname_inp = "img.jpg";
  std::string fname_out = "img.out";
  float mask_threshold = 0.f;
  float iou_threshold = 0.88f;
  float stability_score_threshold = 0.95f;
  float stability_score_offset = 1.0f;
  float eps = 1e-6f;
  float eps_decoder_transformer = 1e-5f;

  sam_prompt prompt;
};

struct sam_model {
  sam_hparams hparams;

  sam_encoder_image enc_img;
  sam_encoder_prompt enc_prompt;
  sam_decoder_mask dec;

  //
  struct ggml_context *ctx;
  std::map<std::string, struct ggml_tensor *> tensors;
};

bool sam_params_parse(int argc, char **argv, sam_params &params);

bool sam_model_load(const sam_params &params, sam_model &model);

cv::Mat sam_image_preprocess(cv::Mat img);

struct ggml_cgraph *sam_encode_image(const sam_model &model, sam_state &state,
                                     const cv::Mat &img);

struct ggml_cgraph *sam_build_fast_graph(const sam_model &model,
                                         sam_state &state, int nx, int ny,
                                         sam_prompt prompt);

std::vector<cv::Mat> sam_get_masks(const sam_hparams &hparams, int nx, int ny,
                                   const sam_state &state);

class sam_predictor {
public:
  explicit sam_predictor(sam_params &params) {
    // load the model
    if (!sam_model_load(params, model)) {
      char buf[128];
      snprintf(buf, 128, "%s: failed to load model from '%s'\n", __func__,
               params.model.c_str());
      throw std::runtime_error(buf);
    }

    // Init state
    {
      static size_t buf_size = 256u * 1024 * 1024;

      struct ggml_init_params ggml_params = {
          .mem_size = buf_size,
          .mem_buffer = nullptr,
          .no_alloc = false,
      };

      state.ctx = ggml_init(ggml_params);

      state.embd_img = ggml_new_tensor_3d(
          state.ctx, GGML_TYPE_F32, model.hparams.n_img_embd(),
          model.hparams.n_img_embd(), model.hparams.n_enc_out_chans);

      state.low_res_masks = ggml_new_tensor_3d(
          state.ctx, GGML_TYPE_F32, model.hparams.n_enc_out_chans,
          model.hparams.n_enc_out_chans, 3);

      state.iou_predictions = ggml_new_tensor_1d(state.ctx, GGML_TYPE_F32, 3);
    }
  }

  void encode_image(const cv::Mat &mat0, int n_threads) {
    nx0 = mat0.cols;
    ny0 = mat0.rows;
    const auto mat1 = sam_image_preprocess(mat0);
    // Encode image

    state.buf_compute_img_enc.resize(ggml_tensor_overhead() *
                                         GGML_DEFAULT_GRAPH_SIZE +
                                     ggml_graph_overhead());
    state.allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    struct ggml_cgraph *gf = sam_encode_image(model, state, mat1);
    if (!gf) {
      char buf[128];
      snprintf(buf, 128, "%s: failed to encode image\n", __func__);
      throw std::runtime_error(buf);
    }

    ggml_graph_compute_helper(state.work_buffer, gf, n_threads);

    // print_t_f32("embd_img", state.embd_img);

    ggml_gallocr_free(state.allocr);
    state.allocr = nullptr;
    state.work_buffer.clear();
  }

  std::vector<cv::Mat> encode_prompts_and_decode_masks(const sam_prompt &prompt,
                                                       int n_threads) {
    // Encode prompt and decode mask
    {
      state.buf_compute_fast.resize(ggml_tensor_overhead() *
                                        GGML_DEFAULT_GRAPH_SIZE +
                                    ggml_graph_overhead());
      state.allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

      switch (prompt.prompt_type) {
      case SAM_PROMPT_TYPE_POINT:
        fprintf(stdout, "Using point prompt: (%f, %f)\n", prompt.pt.x,
                prompt.pt.y);
        break;
      case SAM_PROMPT_TYPE_BOX:
        fprintf(stdout, "Using box prompt: (%f, %f, %f, %f)\n", prompt.box.x1,
                prompt.box.y1, prompt.box.x2, prompt.box.y2);
        break;
      }

      struct ggml_cgraph *gf =
          sam_build_fast_graph(model, state, nx0, ny0, prompt);
      if (gf == nullptr) {
        char buf[128];
        snprintf(buf, 128, "%s: failed to build fast graph\n", __func__);
        throw std::runtime_error(buf);
      }

      ggml_graph_compute_helper(state.work_buffer, gf, n_threads);

      // print_t_f32("iou_predictions", state.iou_predictions);
      // print_t_f32("low_res_masks", state.low_res_masks);
      ggml_gallocr_free(state.allocr);
      state.allocr = nullptr;
    }

    return sam_get_masks(model.hparams, nx0, ny0, state);
  }
  ~sam_predictor() { ggml_free(model.ctx); }

private:
  sam_model model;
  sam_state state;
  int nx0{};
  int ny0{};
};