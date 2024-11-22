# Machine Learning CI/CD Example

This project sets up a basic CI/CD pipeline for training and deploying a Convolutional Neural Network (CNN) on the MNIST dataset. The pipeline includes model training, automated tests, and deployment.

## Setup and Run Locally

1. Clone the repository.
2. Install dependencies: `pip install torch torchvision matplotlib`.
3. Run the training script: `python train.py`.
4. Run tests: `python test.py`.
5. Deploy the model: `python deploy.py`.

## Augmented Images
Here are a few samples from the augmented MNIST dataset:

![Augmented Image 1](./augmented_image1.png)
![Augmented Image 2](./augmented_image2.png)

## Build Status
![Build Status](https://github.com/SaiAnjani/MNSIT_CICD/actions/workflows/ci-cd.yml/badge.svg)
