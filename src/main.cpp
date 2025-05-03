#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "sam.hpp"

int main(int argc, char *argv[]) {
  fprintf(stdout, "OpenCV version: %s\n", cv::getVersionString().c_str());

  sam_params params;

  if (sam_params_parse(argc, argv, params) == false) {
    return 1;
  }

  params.model = "medsam_vit_b-ggml-f16.bin";
  params.fname_inp = "img_demo.png";
  params.multimask_output = false; // MedSAM uses single mask output

  // Example bounding box
  params.prompt.prompt_type = SAM_PROMPT_TYPE_BOX;
  params.prompt.box = {.x1 = 264, .y1 = 281, .x2 = 406, .y2 = 394};

  // MedSAM does not care about these thresholds
  params.stability_score_threshold = 0.0;
  params.iou_threshold = 0.0;

  if (params.seed < 0) {
    params.seed = time(nullptr);
  }
  fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

  // Initialize SAM predictor
  sam_predictor predictor(params);

  // load the input image
  const auto mat0 = cv::imread(params.fname_inp, cv::IMREAD_COLOR_RGB);

  // Compute the image embeddings once
  {
    const int64_t t_main_start_us = ggml_time_us();
    predictor.encode_image(mat0.cols, mat0.rows, medsam_image_preprocess(mat0),
                           params.n_threads);
    const int64_t elapsed_us = ggml_time_us() - t_main_start_us;
    fprintf(stderr, "encode_image took %8.2f ms\n", elapsed_us / 1000.0F);
  }

  // Continue to select ROI, compute prompt embeddings, and decode masks
  while (true) {
    const auto roi = cv::selectROI("Select ROI", mat0);
    cv::destroyWindow("Select ROI");

    // Convert cv::Rect to sam_box
    params.prompt.prompt_type = SAM_PROMPT_TYPE_BOX;
    params.prompt.box = {.x1 = static_cast<float>(roi.x),
                         .y1 = static_cast<float>(roi.y),
                         .x2 = static_cast<float>(roi.x + roi.width),
                         .y2 = static_cast<float>(roi.y + roi.height)};

    // Encode mask prompt, decode masks
    const auto masks = predictor.encode_prompts_and_decode_masks(
        params.prompt, params.multimask_output, params.n_threads);

    // masks will have size == 1 because we set `.multimask_output = false`
    for (int i = 0; i < masks.size(); ++i) {
      const std::string filename =
          params.multimask_output
              ? params.fname_out + std::to_string(i) + ".png"
              : params.fname_out + ".png";
      const auto &mask = masks[i];
      cv::imwrite(filename, mask);

      cv::Mat mask_green;
      cv::Mat zero = cv::Mat::zeros(mask.size(), mask.type());
      std::vector<cv::Mat> channels = {zero, mask, zero};
      cv::merge(channels, mask_green);

      cv::Mat masked;
      cv::addWeighted(mat0, 0.8, mask_green, 0.2, 0.0, masked);
      cv::imshow(filename, masked);
    }

    if (masks.size()) {
      cv::waitKey(0);
    }
  }

  return 0;
}
