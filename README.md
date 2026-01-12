# Train

**Model training, experimentation, and optimization for on-device animal excrement detection.**

This repository contains the **training and experimentation pipeline** for the **BarnSight** computer vision models.
Its sole responsibility is to produce **efficient, accurate models** that can later be deployed to edge devices running inside barns.

If the edge device is the eyes, this repo is the school they went to.

### 🎯 Purpose

Farm environments are visually chaotic:

- uneven lighting
- dirty floors
- occlusions
- different animal species
- wildly inconsistent “ground truth”

This repository exists to turn that chaos into robust, production-ready models that can reliably detect visible animal excrement under real farm conditions.

The output of this repo is trained model artifacts, not a running service.


# 🧪 Training Workflow

1. Prepare and label farm images

    - Different animals
    - Different floor materials
    - Different lighting conditions
    - Multiple contamination levels

2. Configure training parameters

    - Model architecture
    - Input resolution
    - Confidence thresholds
    - Augmentation strategy

3. Train the model

    - Locally or in Colab
    - Monitor metrics (loss, precision, recall)

4. Export trained weights
    - Optimized for edge deployment
    - Ready to be placed in the edge device models/ directory


# ⚙️ Configuration

Training behavior is controlled via:

- **settings.ini** — model and dataset configuration
- **train.py** — training logic and orchestration

This keeps experiments reproducible and explicit.

# License

Licensed under the terms specified in the **LICENSE** file.
