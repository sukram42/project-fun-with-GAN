# GAN on functions

[![used Pytorch](https://img.shields.io/badge/PyTorch-blue.svg)](https://shields.io/)

In this small project, I want to train a GAN on simple distributions. The ```main.py``` file has the main logic for this. In the ```networks.py``` file I specified the networks for the generator and the discriminator. The ```data_generation``` script encapsulates the Dataset creation. Here the main functions are given.

For the Sinus-Dataset I used the function:

```
   $np.tanh(np.sin(x) + np.random.normal(0, 0.05) % 1)-0.5 
```
I am using a ```tanh``` Wrapper to map the function between -1 and 1 to give the generator a chance. (We use a ```tanh``` output of it)

## Results
![Training](https://github.com/sukram42/project-fun-with-GAN/blob/main/function_GAN_pytorch/training.gif)

The following video shows the training of the network for 10 000 epochs with a batch-size of 1024 datapoints. The video shows the training process. On left side the respective losses are shown for the discriminator and generator. On the original samples of the distribution are shown (green). These examples are sampled newly in each epoch, therefore the points are changing. The red points show the sampled data usin the same latent state.
 
