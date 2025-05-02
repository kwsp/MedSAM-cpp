#include <fmt/core.h>
#include <opencv2/core/utility.hpp>

#include "sam.hpp"

int main(int argc, char *argv[]) {
  fmt::println("OpenCV version: {}", cv::getVersionString());

  const int64_t t_main_start_us = ggml_time_us();

  sam_params params;
  params.model = "models/sam-vit-b/ggml-model-f16.bin";

  if (sam_params_parse(argc, argv, params) == false) {
    return 1;
  }

  if (params.seed < 0) {
    params.seed = time(nullptr);
  }
  fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

  // load the image
  const auto mat0 = cv::imread(params.fname_inp, cv::IMREAD_COLOR_RGB);

  sam_predictor predictor(params);

  predictor.encode_image(mat0, params.n_threads);

  const auto masks = predictor.encode_prompts_and_decode_masks(
      params.prompt, params.n_threads);

  for (int i = 0; i < masks.size(); ++i) {
    std::string filename = params.fname_out + std::to_string(i) + ".png";
    cv::imwrite(filename, masks[i]);
  }

  // report timing
  {
    const int64_t t_main_end_us = ggml_time_us();

    fprintf(stderr, "\n\n");
    fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__,
            (t_main_end_us - t_main_start_us) / 1000.0f);
  }

  return 0;
}
