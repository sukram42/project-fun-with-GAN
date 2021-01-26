![gan_training](./GAN_on_MNIST/videos/gan_generate_0_latentstate10.gif)

# Project __FUN with GAN__

I am using this repository to make some experiments with **Generative Adversarial 
Networks (GAN)**. To not fully spam my github profile with all the different projects, I
am using this repository as a sort of monorepo for all the different experiments.
My overall aim is to get more experience in the development of advanced deep learning models as well as to get a deeper intuitive understanding of the algorithms. 
I plan to only use Python for the implementation of the GAN networks.


## My Subprojects

The following list shows all my subprojects. I use this list as an __TODO__-list for myself in which I also include some ideas which would be interested to research in. To challenge myself, I want to be able to use __Tensorflow__ as well as __PyTorch__ as DeepLearning libraries. Therefore I add this to the list.

| Project_name | year | Description | Frameworks used
| :---:        | :---:|       :---: | :---:   
| GAN on simple Function | 2020 | As a simple start I wanted to train a GAN on simple function like sin etc. This I have done in this part | ![Pytorch](https://img.shields.io/badge/PyTorch-blue)       
| GAN on MNIST |  2020| To begin with I want to create a simple GAN network which is able to generate numbers from the MNIST dataset. I further want to analyse the latent state | ![Tensorflow 2.4](https://img.shields.io/badge/Tensorflow2.4-orange)
| CondGANonMNIST| planned | Next I want to create a Conditioned GAN on the MNIST and compare it to the first GAN network | ![unspecified](https://img.shields.io/badge/unspecified-black)
| BI-GAN| planned | BI-GANs train an encoder network with the generator. I thought an comparison to a Variational Autoencoder would be nice to see. Especially focussed on Sample Efficiency|![unspecified](https://img.shields.io/badge/unspecified-black)
| BI-GAN / VAC for Anomaly Detection | planned | Bi-GANs encode train on 'normal' datapoints. Can't we use this to find anomalies in image or timeseries data?|![unspecified](https://img.shields.io/badge/unspecified-black)
| Bi-GAN / VAC for Compression | planned | With the help of encoders we map pictures, music, etc to a much smaller latent state. Can't we use this for lossy compression? For example for archieving time series? Or transmit time series (IOT?) |![unspecified](https://img.shields.io/badge/unspecified-black)


  