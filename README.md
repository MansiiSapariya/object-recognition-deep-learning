# Object Recognition with Deep Learning

A deep learning project for recognizing everyday objects using convolutional neural networks, transfer learning with ResNet18, and CNN-LSTM based sequence modelling.

The project explores image classification on a custom dataset and extends the workflow toward extracting visual features from video frames for temporal modelling.

## Project Overview

This project consists of three main components:

1. **CNN-based Image Classification**
2. **ResNet18 Transfer Learning**
3. **CNN-LSTM Sequence Modelling**

The experiments demonstrate how deep learning can be used to learn visual features from images and how CNN-extracted features can be extended to sequential data.

## Dataset

The project uses a custom dataset containing **150 images across five object classes**:

* Bottle
* Calculator
* New folder
* Pen
* Watch

The dataset is divided into:

| Dataset    |  Images |
| ---------- | ------: |
| Training   |     105 |
| Validation |      22 |
| Testing    |      23 |
| **Total**  | **150** |

Images are resized to **64 × 64 × 3** for the CNN experiments.

The dataset is included in:

```text
data/object-recognition-dataset.zip
```

## 1. CNN Image Classification

A convolutional neural network is implemented for multi-class object classification.

The CNN consists of convolutional, pooling, flattening, and fully connected layers.

### Architecture

```text
Input Image
    │
    ▼
Conv2D
    │
MaxPooling
    │
Conv2D
    │
MaxPooling
    │
Conv2D
    │
MaxPooling
    │
Flatten
    │
Dense
    │
Dropout
    │
Softmax Output
```

The model uses:

* Adam optimizer
* Sparse categorical cross-entropy
* Early stopping
* Batch size of 16
* Maximum of 15 epochs

### Result

The TensorFlow/Keras CNN achieved:

**95.65% test accuracy**

The classification results showed strong performance across the five object classes, with only one test sample being misclassified.

## 2. ResNet18 Transfer Learning

The project also explores transfer learning using **ResNet18 pretrained on ImageNet**.

The pretrained layers are frozen and the final classification layer is replaced to match the five custom object classes.

This provides a comparison between:

* A CNN trained from scratch
* A pretrained ResNet18 model

### Results

| Model      | Train Accuracy | Validation Accuracy | Test Accuracy |
| ---------- | -------------: | ------------------: | ------------: |
| Simple CNN |        100.00% |             100.00% |   **100.00%** |
| ResNet18   |         99.05% |             100.00% |    **91.30%** |

For this particular small custom dataset, the CNN trained from scratch achieved better test performance than the frozen ResNet18 model.

This highlights that transfer learning does not necessarily outperform a simpler architecture on every small, domain-specific dataset.

## 3. CNN-LSTM Sequence Modelling

The final part of the project extends image-based feature extraction to video sequence modelling.

A video is processed frame by frame:

```text
Video
  │
  ▼
Frame Extraction
  │
  ▼
Image Preprocessing
  │
  ▼
CNN Feature Extraction
  │
  ▼
Feature Sequence
  │
  ▼
LSTM
  │
  ▼
Classification
```

The supplied video is:

```text
data/sequence/watch_video.mp4
```

The pipeline extracts up to **50 frames** from the video.

Each frame is passed through a CNN feature extractor, producing **128 features per frame**.

The resulting sequence has the shape:

```text
50 × 128
```

The sequence is then provided to an LSTM containing:

* 64 LSTM units
* Dense layer with 32 units
* Sigmoid output layer

### Important Note

The supplied experiment demonstrates the CNN-to-LSTM feature extraction pipeline, but the LSTM is **not trained on a sufficiently large collection of labelled video sequences**.

Therefore, its sequence prediction should not be interpreted as a properly evaluated classification result.

## Results Summary

The experiment demonstrates strong image-classification performance on the custom dataset.

| Experiment                 | Test Accuracy |
| -------------------------- | ------------: |
| TensorFlow/Keras CNN       |    **95.65%** |
| PyTorch CNN                |   **100.00%** |
| ResNet18 Transfer Learning |    **91.30%** |

The CNN-LSTM component demonstrates the construction of a spatial-temporal pipeline rather than providing a statistically reliable sequence-classification evaluation.

Because the dataset contains only 150 images, the reported classification results should be interpreted within the context of this small custom dataset rather than as evidence of general-purpose object-recognition performance.

## Repository Structure

```text
object-recognition-deep-learning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── object_recognition_cnn_resnet_lstm.ipynb
│
├── data/
│   ├── object-recognition-dataset.zip
│   └── sequence/
│       └── watch_video.mp4
│
└── docs/
    └── results.md
```

## Technologies Used

* Python
* TensorFlow
* Keras
* PyTorch
* TorchVision
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

## Key Concepts

This project provides practical experience with:

* Image preprocessing
* Convolutional neural networks
* Multi-class image classification
* Feature extraction
* Transfer learning
* ResNet18
* Model evaluation
* Classification reports
* Confusion matrices
* Video frame extraction
* CNN feature extraction
* LSTM sequence modelling
* Spatial-temporal learning

## Installation

Clone the repository:

```bash
git clone https://github.com/MansiiSapariya/object-recognition-deep-learning.git
cd object-recognition-deep-learning
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Open the main notebook:

```text
notebooks/object_recognition_cnn_resnet_lstm.ipynb
```

The notebook can be run using:

* Jupyter Notebook
* JupyterLab
* Google Colab

Extract the dataset and ensure that the dataset path matches the path expected by the notebook.

The video used for the sequence-modelling experiment is located at:

```text
data/sequence/watch_video.mp4
```

## Limitations

* The dataset contains only 150 images.
* The number of object classes is limited to five.
* The dataset has limited variation in backgrounds, lighting, and object orientation.
* The high classification accuracy may not generalize to unseen real-world environments.
* ResNet18 was evaluated using frozen pretrained layers rather than extensive fine-tuning.
* The CNN-LSTM component lacks a sufficiently large labelled sequence dataset for meaningful evaluation.

## Future Improvements

Possible extensions include:

* Increasing the dataset size.
* Adding more object categories.
* Introducing greater variation in lighting and backgrounds.
* Applying stronger data augmentation.
* Fine-tuning deeper ResNet18 layers.
* Comparing additional pretrained architectures.
* Creating a larger labelled video dataset.
* Training the CNN-LSTM model on multiple sequences.
* Developing a real-time webcam-based object recognition application.
* Deploying the trained model as an interactive application.

## Author

**Mansi Sapariya**

MSc Data Science
Christ (Deemed to be University), Bangalore

## Disclaimer

This project was developed as an academic deep learning experiment. The reported results are specific to the dataset, models, and experimental setup used in this project and should not be interpreted as general-purpose object recognition performance.
