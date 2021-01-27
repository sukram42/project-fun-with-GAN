import numpy as np
from matplotlib import pyplot as plt


class Dataset:
    def __init__(self, scale=1):
        self.scale = scale

    def _get_y(self, x):
        raise NotImplemented("The get_y method is not implemented!")

    def _gen_X(self, n):
        return self.scale * (np.random.random_sample((n,)))

    def sample_data(self, n):
        data = []

        x = self._gen_X(n)
        for i in range(n):
            yi = self._get_y(x[i])
            data.append([x[i], yi])

        return np.array(data)

    def plot_sample(self, samples=None, n=100):
        fig = plt.figure()
        if samples is not None:
            plt.scatter(samples.T[0], samples.T[1], label="Fake")
        dat = self.sample_data(n)
        plt.scatter(dat.T[0], dat.T[1], label="Real")
        plt.legend()
        plt.show()



###
class SinusDataset(Dataset):
    def _get_y(self, x):
        return np.tanh(np.sin(x) + np.random.normal(0, 0.05) % 1)-0.5


##
s = SinusDataset().sample_data(100)
plt.scatter(s.T[0], s.T[1], label="Fake")
plt.show()
##
SinusDataset().plot_sample()

##

