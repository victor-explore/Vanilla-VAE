# Vanilla Variational Autoencoder (VAE) for Image Generation

This repository contains a Jupyter notebook implementing a Vanilla Variational Autoencoder (VAE) for image generation. The VAE is a powerful generative model that learns to encode images into a latent space and then decode them back into reconstructed images.

## Contents

- `08-10-2024 Vanilla VAE.ipynb`: The main Jupyter notebook containing the implementation of the Vanilla VAE.

## Features

- Implementation of a Vanilla VAE using PyTorch
- Training on a dataset of images
- Generation of new images from the learned latent space
- Visualization of training progress and generated images

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

## Usage

1. Open the `08-10-2024 Vanilla VAE.ipynb` notebook in a Jupyter environment.
2. Run the cells in order to train the VAE and generate images.
3. The notebook will save generated images in the `generated_images` folder and model checkpoints in the `saved_models` folder.

## Model Architecture

The VAE consists of an encoder network that maps input images to a latent space representation and a decoder network that reconstructs images from the latent space. The architecture is as follows:

### Encoder
- Convolutional layers to extract features from input images
- Fully connected layers to map features to mean and log-variance of the latent space distribution

### Latent Space
- Reparameterization trick to sample from the latent space distribution

### Decoder
- Fully connected layers to map latent space samples to feature maps
- Transposed convolutional layers to reconstruct the image from feature maps

## Training Process

The VAE is trained using two loss components:
1. Reconstruction loss: Measures how well the decoder can reconstruct the input image
2. KL divergence loss: Ensures the latent space distribution is close to a standard normal distribution

The total loss is optimized using the Adam optimizer.

## Results

The notebook demonstrates:
- Training progress with loss curves
- Original images vs. reconstructed images
- Generated images from random latent space samples
- Interpolation between images in the latent space

## Future Improvements

- Experiment with different architectures for encoder and decoder
- Try advanced VAE variants like β-VAE or VQ-VAE
- Apply the model to different datasets

## References

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.
- PyTorch documentation: https://pytorch.org/docs/stable/index.html

## License

This project is open-source and available under the MIT License.
