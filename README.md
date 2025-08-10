# Visualizing Neural Network Training Dynamics

This project provides a hands-on, visual exploration of the core theoretical concepts behind how deep neural networks learn. Through three distinct computational experiments, we demystify abstract ideas like Information Geometry, the Neural Tangent Kernel (NTK), and the high-dimensional trajectory of optimizers.

The goal is to provide tangible, intuitive evidence for the theories that explain the success of modern deep learning.

## Project Visualizations

The project is divided into three phases, each producing a key visualization.

### Phase 1: Standard vs. Natural Gradient Descent

This experiment visualizes the difference between a standard optimizer and a "geometry-aware" one (Natural Gradient Descent) on a curved loss landscape. The result clearly shows how NGD finds a highly efficient path by understanding the parameter space's true geometry. In contrast, standard GD is far slower; while it heads in the correct general direction, it gets trapped in inefficient oscillations that prevent it from reaching the minimum in the same number of iterations.

<img src="https://github.com/user-attachments/assets/fd16c8ac-1db3-4e21-a772-2b7f0e35c8b2" alt="NGD vs SGD" width="70%">

### Phase 2: Comparing Lazy vs. Rich Regimes with the NTK

This experiment tracks the eigenvalues of the empirical Neural Tangent Kernel (eNTK) of a finite-width network during training. The deviation of the eNTK's eigenvalues (solid lines) from the static, theoretical NTK (dashed lines) serves as a visual proxy for **feature learning**, demonstrating that the network is operating in the powerful "rich" regime.

<img src="https://github.com/user-attachments/assets/e6e16653-14d4-445f-ab4e-7f26a8fb8ae8" alt="NTK Evolution" width="70%">

### Phase 3: Visualizing the High-Dimensional SGD Trajectory

This experiment uses Principal Component Analysis (PCA) to project the entire training path of a Convolutional Neural Network into a visible 3D space. The resulting plot reveals the multi-stage nature of optimization: an initial phase of rapid, broad exploration across the landscape followed by a final phase of slow, fine-tuning convergence within a stable basin.

<img src="https://github.com/user-attachments/assets/5852bebc-c591-4abc-864d-b8a375ab11ad" alt="PCA Trajectory" width="70%">

## Theoretical Background

This project demonstrates concepts from modern deep learning theory, including:
* **Information Geometry & the Fisher Information Metric (FIM):** Explains why parameter spaces are curved and how optimizers like NGD can navigate them efficiently.
* **The Neural Tangent Kernel (NTK):** Provides a theoretical baseline for "lazy" network training and allows us to measure the degree of feature learning in practical "rich" networks.
* **SGD Dynamics & Implicit Bias:** Shows that the path an optimizer takes is a complex, multi-stage process that is key to finding solutions that generalize well.

## Requirements

To run the code for these experiments, you will need the following Python libraries:
* `JAX`
* `PyTorch` & `torchvision`
* `NumPy`
* `Matplotlib`
* `scikit-learn`
* `neural-tangents`
* `tqdm`

## Usage

The project is divided into three Python scripts, one for each phase. You can run each script independently to reproduce the visualizations.
