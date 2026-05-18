# Car License Plate Detection using Deep Learning

## Repository Name
`car-license-plate-detection`

## Dataset
[Kaggle Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)

---

# Project Overview

This project focuses on detecting vehicle license plates from car images using Deep Learning and Computer Vision techniques. The system is trained to identify and localize license plates accurately from different vehicle images under various environmental conditions.

---

# Milestone 1 — Dataset Collection and Understanding

## Tasks
- Download the car license plate dataset from Kaggle
- Understand dataset structure and annotations
- Analyze image quality and license plate positions
- Split dataset into training and testing sets
- Organize image and annotation folders properly

### Deliverables
- Structured dataset
- Train and test dataset folders
- Dataset analysis report

---

# Milestone 2 — Image Preprocessing

## Tasks
- Resize images into a fixed format
- Normalize image pixel values
- Remove noisy or low-quality images
- Apply image enhancement techniques
- Prepare annotation files for training
- Perform data augmentation:
  - Rotation
  - Brightness adjustment
  - Scaling

### Deliverables
- Preprocessed image dataset
- Augmented training dataset
- Clean annotation files

---

# Milestone 3 — Model Training

## Tasks
- Select suitable object detection model
- Train the model using training images
- Configure training parameters:
  - Epochs
  - Batch size
  - Learning rate
- Validate model performance during training
- Save trained model checkpoints

### Deliverables
- Trained detection model
- Training logs
- Validation accuracy results

---

# Milestone 4 — Testing and Evaluation

## Tasks
- Test model using unseen vehicle images
- Detect and localize license plates
- Compare predicted and actual outputs
- Evaluate model using:
  - Accuracy
  - Precision
  - Recall
  - IoU
  - mAP
- Display output images with detected plates
- Prepare final project documentation

### Deliverables
- Detection output results
- Model evaluation report
- Final project documentation

---

# Technologies Used

- Python
- OpenCV
- TensorFlow / PyTorch
- YOLO / CNN
- NumPy
- Matplotlib

---

# Expected Outcome

The developed system should accurately detect vehicle license plates from images and provide reliable localization results for different vehicle conditions and viewing angles.
