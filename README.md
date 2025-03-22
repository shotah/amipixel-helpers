# AmiPixel Helpers

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/shotah/amipixel-helpers/graphs/commit-activity)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Example Animation](gif_conversion/rabbit_converted_back.gif)

## Overview

AmiPixel Helpers is a project designed to assist in the creation of pixel art animations using AI. Leveraging the power of diffusion models like Stable Diffusion and the motion capabilities of AnimateDiff, this tool allows you to generate sequences of pixel art images based on text prompts and optional initial images.

## Key Features

- **Text-to-Pixel Art Animation:** Generate animated sequences from descriptive text prompts.
- **Base Image Support:** Optionally use a starting pixel art image to guide the animation.
- **Customizable Parameters:** Control the number of animation frames, inference steps, and guidance scale.
- **Utilizes Powerful AI Models:** Built on top of Stable Diffusion v1.5 and the AnimateDiff motion adapter.
- **Memory Optimization:** Includes options for memory-efficient attention (using `xformers`).
- **Simplified Workflow:** Utilizes a `Makefile` for easy management of common tasks.

## Getting Started

Follow these steps to get your environment set up and start generating pixel art animations.

### Prerequisites

- **Python 3.8 or higher:** Ensure you have Python installed.
- **pipenv:** This project uses `pipenv` for managing dependencies. You can install it with `pip install pipenv`.
- **CUDA-enabled GPU (Recommended):** While you can run this on a CPU, a CUDA-enabled NVIDIA GPU is highly recommended for reasonable generation speeds. Ensure you have the appropriate NVIDIA drivers installed.
- **Hugging Face Account and Token:** You'll need a Hugging Face account to download the pre-trained models. Create an account [here](https://huggingface.co/join) and generate an access token [here](https://huggingface.co/settings/tokens).
- **Git:** To clone the repository.

### Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/amipixel-helpers.git](https://github.com/YOUR_GITHUB_USERNAME/amipixel-helpers.git)
   cd amipixel-helpers
   ```

   _(Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username)_

2. **Install dependencies using pipenv:**

   ```bash
   make install
   ```

### Environment Setup

1. **Create a `.env` file:** In the root of the project directory, create a file named `.env`.
2. **Add your Hugging Face token:** Open the `.env` file and add the following line, replacing `YOUR_HUGGING_FACE_TOKEN` with your actual token:

   ```
   HUGGING_FACE_TOKEN=YOUR_HUGGING_FACE_TOKEN
   ```

### Running the Frame Generator

The primary way to interact with this project is through the `Makefile`.

## Using Make Commands

This project uses a `Makefile` to simplify common development tasks. You can see a list of available commands and their descriptions by running:

```bash
make help
```
