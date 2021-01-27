###
import time

import torch
import torch.nn as nn
import matplotlib.animation as anim

import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable

from function_GAN_pytorch.data_generation import Dataset, SinusDataset
from function_GAN_pytorch.networks import SimpleGenerator, SimpleDescriminator

##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##
dataset: Dataset = SinusDataset()


# Function to sample noise
def sample_noise(shape):
    return torch.Tensor(np.random.normal(-1, 1, size=shape))


##
latent_space = 10
generator = SimpleGenerator(input_dim=latent_space,
                            hidden_dims=100,
                            output_dim=2,
                            lr_gen=0.0002).to(device)
descriminator = SimpleDescriminator(input_dim=2,
                                    hidden_dims=100,
                                    output_dim=1,
                                    lr_disc=0.0002).to(device)

##
num_epochs = 10000
batch_size = 1024
d_rounds = 1
g_rounds = 1
g_losses, d_fake_losses, d_real_losses = [], [], []
loss = torch.nn.BCELoss()

fig1 = plt.figure(figsize=(15, 5))
ax11 = fig1.add_subplot(221)  # add subplot into first position in a 2x2 grid (upper left)
ax12 = fig1.add_subplot(223, sharex=ax11)  # add to third position in 2x2 grid (lower left) and sharex with ax11
ax13 = fig1.add_subplot(122)
imgss = []

validation_seed = sample_noise((batch_size, latent_space)).to(device)

for epoch in range(num_epochs):
    tic = time.time()
    real_data, generated_data = (None, None)
    g_ep_losses, d_fake_ep_losses, d_real_ep_losses = [], [], []

    descriminator.optimizer.zero_grad()
    generator.optimizer.zero_grad()

    # ##################################################
    # 1. DiscriminatorSteps
    # ##################################################

    # Generate and Train Real Samples
    real_data = torch.Tensor(dataset.sample_data(n=batch_size)).to(device)
    y_hat_real = descriminator(real_data)

    l_real = loss(y_hat_real, (torch.ones((batch_size, 1))).to(device))
    l_real.backward()

    # 2. Generate Fake Samples
    generator_input = sample_noise((batch_size, latent_space)).to(device)
    generator_input_var = Variable(generator_input, requires_grad=False)
    generated_data = generator(generator_input)

    # 3. Learn Fake Discriminator
    y_hat_fake = descriminator(generated_data.detach())
    l_gen = loss(y_hat_fake, torch.zeros((batch_size, 1)).to(device))
    l_gen.backward()
    descriminator.optimizer.step()

    d_real_ep_losses.append(l_real.item())
    d_fake_ep_losses.append(l_gen.item())

    # ##################################################
    # Generator Step
    # ##################################################
    pred = descriminator(generated_data).to(device)

    # We want to minimize the loss between prediction and 1 (real_data)
    desc_loss = loss(pred, torch.ones((batch_size, 1)).to(device))
    desc_loss.backward()

    generator.optimizer.step()

    g_ep_losses.append(desc_loss.item())
    g_losses.extend(g_ep_losses)
    d_fake_losses.extend(d_fake_ep_losses)
    d_real_losses.extend(d_real_ep_losses)

    toc = time.time()
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch} | D_LOSS_REAL: {np.mean(d_real_ep_losses):.4f} | D_LOSS_FAKE: {np.mean(d_fake_ep_losses):.4f} | G_LOSS: {np.mean(g_ep_losses):.4f} | {toc-tic:.4f} s")

    if epoch % 25 ==0:
        img1, = ax11.plot(g_losses, label="Generator Loss", color="blue")
        ax11.set_title('Generator')
        img2, = ax12.plot(d_fake_losses, label="Fake Discriminator Loss", color="darkred")
        img3, = ax12.plot(d_real_losses, label="Real Discriminator Loss", color="darkgreen")
        ax12.set_title('Discriminator')

        dat = dataset.sample_data(200)
        img4 = ax13.scatter(dat.T[0], dat.T[1], label="Real",  color="darkgreen", alpha="0.8")

        faked_val = generator(validation_seed).detach().cpu()
        img5 = ax13.scatter(faked_val.T[0], faked_val.T[1], label="Fake", color="darkred", alpha="0.8")
        fig1.suptitle(f"Epoch {epoch}")
        if epoch == 0:
            ax12.legend()
            ax13.legend()

        imgss.append([img1, img2, img3, img4, img5])

print("Saving video")
ani = anim.ArtistAnimation(fig1, imgss, interval=50, blit=False)
ani.save("training.mp4")

if __name__ == '__main__':
    pass
