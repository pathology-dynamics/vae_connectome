import torch
from torch import Tensor
from torch.nn import Parameter
from torch import nn
from utils import sample_from_gaussian, compute_sigma_s
import torch.nn.functional as f


class Probabilistic_NetworkDiffusion(nn.Module):
    def __init__(self, x_0, Sigma, sample_size, beta, soft_constr=1):
        # configurations
        super().__init__()
        self.x_0 = x_0
        self.Sigma = Sigma
        self.beta = beta
        self.soft_constr = soft_constr
        self.sample_size = sample_size
        self.Sigmainv = torch.linalg.inv(self.Sigma)
        self.dimension = max(self.x_0.size(0), self.x_0.size(1))
        self.nu = Parameter(torch.randn(self.dimension), requires_grad=True)

        self.diag_ = torch.ones(self.dimension)

        self.Psi = Parameter(
            torch.randn(self.dimension, self.dimension),
            requires_grad=True,
        )

    def decoder(self, time: Tensor, phi: Tensor) -> Tensor:
        """
        The original function of the network diffusion model F(x,t), dx/dt = f(x,t) with IVP x_0
        """
        # time  N
        # phi S x D
        # output N x S x D
        sigma = compute_sigma_s(phi)  # S x 1
        x0phi = phi.mm(self.x_0.view(-1, 1))  # S x 1
        dot_prod = phi.mm(phi.transpose(0, 1))  # S x S
        weights = x0phi / torch.diag(dot_prod, 0).view(-1, 1)  # S x 1
        fea_vec = phi * weights  # S x D
        exp_decay = torch.exp(-self.beta * time.outer(sigma.squeeze()))  # N x S
        x_t_bar = exp_decay.view(exp_decay.size(0), exp_decay.size(1), 1) * fea_vec.expand(
            time.size(0), -1, -1
        )  # N x S X D
        return x_t_bar.to(torch.float32)

    def forward(self, time: Tensor):
        cov_matrix_cholesky = torch.tril(self.Psi, diagonal=-1) + self.diag_ * torch.eye(
            self.dimension
        )
        phi = self.phi(self.nu, cov_matrix_cholesky, self.sample_size, self.dimension**0.5)
        x_t_pred = self.decoder(time, phi)
        x0xt_pred = self.soft_constraint(x_t_pred, phi)
        return x_t_pred, x0xt_pred, phi

    @staticmethod
    def phi(mean: Tensor, cov: Tensor, sample_size: int, norm: float) -> Tensor:
        """
        sample based on nu + Psi^1/2 * epsilon
        """
        eps = torch.randn(sample_size, mean.size(0))  # sample_size x n
        samples = eps.mm(cov).add_(mean) / norm
        return samples.exp_()  # take the exp

    @staticmethod
    def soft_constraint(x_t_pred: Tensor, phi: Tensor) -> Tensor:
        # output N x S
        dot_prod = torch.diag(phi.mm(phi.transpose(0, 1)), 0)  # S
        mat_prod = x_t_pred.matmul(phi.transpose(0, 1))  # N x S x S
        diag = torch.diagonal(mat_prod, offset=0, dim1=1, dim2=2)  # N x S
        return diag**2 / dot_prod

    @staticmethod
    def reconstruction_loss(x_reconstructed, x):
        x = x.view(x.size(0), 1, x.size(1))
        x_expanded = x.expand(-1, x_reconstructed.size(1), -1)
        loss = nn.MSELoss()
        return loss(x_reconstructed, x_expanded)

    def kl_divergence_loss(self):
        cov_matrix_cholesky = torch.tril(self.Psi, diagonal=-1) + self.diag_ * torch.eye(
            self.dimension
        )
        logcov = cov_matrix_cholesky.mm(cov_matrix_cholesky.transpose(0, 1))

        # we change the power from 4 to 2 for avoiding numerical issue
        return 0.5 * (
            torch.trace(self.Sigmainv.mm(logcov))
            + (-self.nu.view(1, -1)).mm(self.Sigmainv.mm(-self.nu.view(-1, 1)))
            - self.dimension
            # - 2*torch.log(torch.diag(cov_matrix_cholesky)).sum()
        )

    def loss_function(
        self, x_t_pred: Tensor, x_t: Tensor, x0xt_pred: Tensor, x0xt: Tensor
    ) -> Tensor:
        if self.soft_constr == 1:
            return (
                self.reconstruction_loss(x_t_pred, x_t),
                self.reconstruction_loss(x0xt_pred, x0xt),
                self.kl_divergence_loss(),
            )
        else:
            return (
                self.reconstruction_loss(x_t_pred, x_t),
                Tensor([0]),
                self.kl_divergence_loss(),
            )

    def get_connectome_posterior(self):
        cov_matrix_cholesky = (
            torch.tril(self.Psi, diagonal=-1) + self.diag_ * torch.eye(self.dimension).detach()
        )
        logcov = cov_matrix_cholesky.mm(cov_matrix_cholesky.transpose(0, 1))

        cov = torch.zeros_like(logcov)

        exp_nu = self.nu.detach().add_(0.5 * torch.diag(logcov)).exp_()

        for i in range(self.dimension):
            for j in range(self.dimension):

                cov[i, j] = torch.exp(
                    self.nu[i] + self.nu[j] + 0.5 * (logcov[i, i] + logcov[j, j])
                ) * (torch.exp(logcov[i, j]) - 1)

        auto_corr = cov + self.nu.outer(self.nu)

        auto_corr = auto_corr / torch.linalg.matrix_norm(auto_corr, 2)
        return auto_corr.detach()

    def get_connectome_posterior_sampling(self):
        cov_matrix_cholesky = (
            torch.tril(self.Psi, diagonal=-1) + self.diag_ * torch.eye(self.dimension).detach()
        )
        sample_num = self.sample_size * 100
        phi = self.phi(self.nu, cov_matrix_cholesky, sample_num, self.dimension**0.5).detach()
        cov = torch.zeros_like(cov_matrix_cholesky)

        for i in range(sample_num):
            cov += phi[i, :].outer(phi[i, :]) / (phi[i, :].dot(phi[i, :]))

        cov /= sample_num

        return cov
