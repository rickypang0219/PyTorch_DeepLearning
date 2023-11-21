import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, Normal
import seaborn as sns
import numpy as np


class GMM_EM:
    def __init__(self, num_class, num_feature, guess_means, guess_covs, data):
        # Initialization of weights, Gaussian mean, and covariance
        self.weights = torch.ones(num_class) / num_class
        self.means = guess_means
        self.covs = guess_covs
        self.data = data

    # Expectation-Maximization algorithm
    def EM_algo(self):
        eps = torch.finfo(torch.float32).eps

        # E-step: Compute responsibilities
        distributions = [
            MultivariateNormal(self.means[i], self.covs[i])
            for i in range(num_components)
        ]
        responsibilities = torch.stack(
            [
                self.weights[i] * distributions[i].log_prob(self.data)
                for i in range(num_components)
            ]
        )
        # responsibilities = responsibilities / torch.sum(responsibilities, dim=0)
        responsibilities = torch.exp(
            responsibilities - torch.logsumexp(responsibilities, dim=0)
        )

        # M-step: Update parameters
        N_k = torch.sum(responsibilities, dim=1)
        weights = N_k / self.data.size(0)
        means = torch.matmul(responsibilities, self.data) / N_k.reshape(-1, 1)
        covs = torch.matmul(
            (
                responsibilities.unsqueeze(2) * (self.data - means.unsqueeze(1))
            ).transpose(1, 2),
            (self.data - means.unsqueeze(1)) / N_k.reshape(-1, 1, 1),
        )

        return weights, means, covs

    def training(self, num_epochs):
        for _ in range(int(num_epochs)):
            self.weights, self.means, self.covs = self.EM_algo()
        # print(f' learned weights {self.weights} \n learned means {self.means} \n learned cov {self.covs} ')
        return self.weights, self.means, self.covs


if __name__ == "__main__":
    data = torch.cat(
        [
            torch.randn(100, 1) + torch.Tensor([5]),
            torch.randn(200, 1) + torch.Tensor([15]),
        ]
    )

    # model params and guesses
    num_components = 2
    num_features = 1
    num_epochs = 10
    weights = torch.ones(num_components) / num_components
    means = torch.tensor([torch.randn(1) + 3, torch.randn(1) + 10]).reshape(
        num_components, num_features
    )
    covs = torch.stack([torch.eye(num_features) for _ in range(num_components)])

    model = GMM_EM(num_components, num_features, means, covs, data)
    trained_w, trained_mu, trained_sigma = model.training(num_epochs)

    # Plot result
    # Convert the learned parameters to numpy arrays
    weights_np = trained_w.numpy()
    means_np = trained_mu.numpy()
    covs_np = trained_sigma.numpy()

    # Generate a range of values
    x = np.linspace(-10, 30, 500)

    # Calculate the probability density for each value
    pdf = np.zeros_like(x)
    for weight, mean, cov in zip(weights_np, means_np, covs_np):
        pdf += (
            weight
            * np.exp(-((x - mean.flatten()) ** 2) / (2 * cov.flatten()))
            / np.sqrt(2 * np.pi * cov.flatten())
        )

    # Plot the histogram of the resulting distribution
    plt.hist(
        data.numpy(), bins=50, density=True, alpha=0.5, label="Data"
    )  # Assuming 'data' contains the original data points
    plt.plot(x, pdf, label="EM algo")
    sns.kdeplot(data, label="KDE")  # KDE of data
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.title("Gaussian Mixture Models Comparision")
    plt.legend()
    plt.show()
