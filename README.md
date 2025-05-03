# MedSAM-cpp

## Installation

Requires CMake, ggml (git submodule), and OpenCV (installed by vcpkg).

## Get Started

Download the [checkpoint `medsam_vit_b.pth`](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN). Run [`scripts/convert-pth-to-ggml.py`](./scripts/convert-pth-to-ggml.py) to convert the `.pth` checkpoint to gguf.

Build and run `main.cpp` for an example on how to use the `sam_predictor` class to predict masks. Make sure to use `medsam_image_preprocess` instead of `sam_image_preprocess` and set `params.multimask_output` to `false`.