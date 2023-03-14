# 1. Audio Super Resolution with Deep Learning models

This repository contains the source code for the implementation of two deep learning models concerning the audio super resolution task. 

The original papers can be found here:

1. [Audio Super Resolution via Vision Transformer](https://link.springer.com/chapter/10.1007/978-3-031-16564-1_36)

2. [Super-Resolution for Music Signals Using Generative Adversarial Networks](https://ieeexplore.ieee.org/document/9515219)


# 2. Audio Super Resolution via ViT

This model is located in the "ViT-SR" folder. The "GAN" folder contains the code for the original paper; here the ViT is used in a Generative Adversarial Network, while the "Autoencoder" folder implements the Bandwidth Extension using an autoencoder-like architecture where the ViT is part of the encoder. For the autoencoder a checkpoint is available with the weights of the model trained on the FMA dataset. Inside the "out" directory there are some examples of the results obtained with the model.

Better results can be obtaines by using the GAN or by extending the training time for the autoencoder.
