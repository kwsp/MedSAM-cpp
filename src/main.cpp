#include "medsam.hpp"
#include <fmt/core.h>
#include <opencv2/core/utility.hpp>

int main(int argc, char *argv[]) {
  fmt::print("MedSAM inference\n");
  fmt::println("OpenCV version: {}", cv::getVersionString());

  return medsam_main(argc, argv);
}
