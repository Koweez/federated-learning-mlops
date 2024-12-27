# Federated Learning on CIFAR-10

This project implements a Federated Learning framework using PyTorch, with a focus on training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. The implementation showcases how to train a global model across multiple simulated clients without sharing their local data, ensuring privacy.

## Overview

Federated Learning is a distributed machine learning technique where multiple clients collaboratively train a shared global model while keeping their data decentralized. This project implements the Federated Averaging (FedAvg) algorithm, which aggregates local model updates to refine the global model.

## Key Features

- **Dataset**: The CIFAR-10 dataset, consisting of 32x32 RGB images across 10 classes, is used for training and evaluation.
- **Federated Averaging**: Local models are trained independently on client data, and their weights are averaged to update the global model.
- **Evaluation**: The global model’s accuracy is evaluated after each federated training round, with results plotted to visualize performance.

## Model Architecture

The Convolutional Neural Network (CNN) is designed to process CIFAR-10’s RGB images and includes:

- Multiple convolutional layers with batch normalization and ReLU activations.
- Max-pooling layers for down-sampling.
- Fully connected layers for classification into 10 classes.
- Dropout layers for regularization.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/koweez/federated-learning-mlops.git
   cd federated-learning-mlops
   ```
2. Install the required packages:
   ```bash
   uv sync
   ```
3. **Optional**: Select the dataset size:
    - To use the full dataset, set 'FULL_DATASET = True'
    - To use a subset of the dataset, set 'FULL_DATASET = False'
    
    This change can be made in the 'train.py' script on line 16.
4. Run the training script:
   ```bash
   uv run train.py
   ```

## Results

The model is trained using Federated Learning across multiple clients, with the global model’s accuracy improving over each federated round. The training process is visualized using Matplotlib, showing the model’s performance over time.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
