# PESU_ISFCR
#### Jun - Jul 2023

# Weakly Supervised Video Anomaly Detection

## Overview

Video Anomaly Detection (VAD) is crucial in various domains like security, healthcare, and public safety due to the increasing prevalence of surveillance cameras. This project introduces a novel Weakly Supervised Anomaly Detection framework aimed at enhancing the accuracy and efficiency of identifying uncommon events or behaviors in video streams.

<img src="https://github.com/ArunAK111/PESU_ISFCR/blob/main/Results/1.png" width="600" height="300" />

### Project Goals

- Develop an efficient VAD model using weak supervision.
- Improve anomaly detection accuracy by capturing contextual information and integrating semantic priors.
- Enhance separability between different anomaly sub-classes.

## Project Structure

The project comprises several components:

### Preprocessing:

- `preprocess.py`: Custom I3D feature extractor for crucial image preprocessing required for the project.
  - Extracts and saves I3D features from surveillance videos.

### Model Development:

- `learner.py`: Implements the primary model for anomaly detection using MIL (Multiple Instance Learning) approach.
- `FFC.py`: Defines an alternative model (Feature Fusion Classifier) for comparison.
- `test.py`: Evaluates the trained model on anomaly detection in video streams.

### Modules Developed:

- **Temporal Context Aggregation (TCA):** Captures video snippet relationships over time, effectively combining local and global context for better temporal understanding.
- **Prompt-Enhanced Learning (PEL):** Utilizes external knowledge to enhance anomaly detection, aligning abnormal contexts with prompts for improved differentiation of anomaly types.

## Requirements

### Frameworks / Libraries Used:

- Python 3
- PyTorch
- OpenCV
- NumPy
- TorchVision


## How to Use

### Preprocessing

1. Set the input and output folders in `preprocess.py`.
2. Run `preprocess.py` to extract I3D features from surveillance videos.

### Training

1. Adjust parameters in `main.py` as needed (learning rate, weight decay, etc.).
2. Run `main.py` to train the model.

### Testing

1. Run `test.py` to evaluate the trained model on anomaly detection.

## Results and Future Work

<img src="https://github.com/ArunAK111/PESU_ISFCR/blob/main/Results/2.png" width="700" height="250" />

The project achieved effective context modeling and semantic enhancement for weakly supervised video anomaly detection. Notable outcomes include:

- Improved anomaly localization accuracy.
- Reduction in false alarms in normal videos.
- Enhanced overall detection performance.

<img src="https://github.com/ArunAK111/PESU_ISFCR/blob/main/Results/3.png" width="400" height="200" />

**Future Directions:**

- Explore multimodal information for improved anomaly detection.
- Investigate open-set anomaly detection methodologies for further advancements.

## Contributors

- **Intern:** Arun Amar Kurali
- **Guide:** Prof. Preet Kanwal
