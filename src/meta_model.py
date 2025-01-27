import torch
from torch import Tensor
from torch.nn import Parameter
from torch import nn


class NetworkDiffusion(nn.Module):
    def __init__(self, x_0, Covs):
        # configurations
        super().__init__()
        self.x_0 = x_0
        self.beta = Parameter(torch.randn(1), requires_grad=True)
        self.dimension = max(self.x_0.size(0), self.x_0.size(1))
        self.Covs = Covs

        lapl1 = self.compute_laplacian(self.Covs[0])
        lapl2 = self.compute_laplacian(self.Covs[1])
        L1, Q1 = torch.linalg.eigh(lapl1)
        L2, Q2 = torch.linalg.eigh(lapl2)

        self.lapl_eigs = ((L1, Q1), (L2, Q2))

    def forward(self, time: float, late: bool):
        if late == False:
            L, Q = self.lapl_eigs[0]
        else:
            L, Q = self.lapl_eigs[1]
        Q_inv_x0 = (Q.transpose(0, 1)).mm(self.x_0.view(-1, 1))  # N x 1
        weights = torch.diag(torch.exp(-self.beta * time * L))  # N x N
        return Q.mm(weights).mm(Q_inv_x0)

    @staticmethod
    def reconstruction_loss(x_reconstructed, x):
        loss = nn.MSELoss(reduction="sum")
        return loss(x_reconstructed.squeeze(), x.squeeze())

    @staticmethod
    def compute_laplacian(adjacency: Tensor) -> Tensor:
        off_diag = adjacency - torch.diagonal(adjacency, 0) * torch.eye(adjacency.size(0))  # N x N
        degree = torch.diag(torch.sum(off_diag, 1))  # N x N
        return degree - off_diag

    def loss_function(self, x_t_pred: Tensor, x_t: Tensor) -> Tensor:
        return self.reconstruction_loss(x_t_pred, x_t)

    def return_eigens(self) -> tuple:
        return self.lapl_eigs[0][0], self.lapl_eigs[1][0]


class NetworkDiffusionLinearSource(nn.Module):
    def __init__(self, x_0, Covs):
        # configurations
        super().__init__()
        self.x_0 = x_0
        self.beta = Parameter(torch.randn(1), requires_grad=True)
        self.dimension = max(self.x_0.size(0), self.x_0.size(1))
        self.r = Parameter(torch.randn(self.dimension), requires_grad=True)
        self.Covs = Covs

        lapl1 = self.compute_laplacian(self.Covs[0])
        lapl2 = self.compute_laplacian(self.Covs[1])
        L1, Q1 = torch.linalg.eigh(lapl1)
        L2, Q2 = torch.linalg.eigh(lapl2)

        self.lapl_eigs = ((L1, Q1), (L2, Q2))

    def forward(self, time: float, late: bool):
        if late == False:
            L, Q = self.lapl_eigs[0]
        else:
            L, Q = self.lapl_eigs[1]
        Q_inv_x0 = (Q.transpose(0, 1)).mm(self.x_0.view(-1, 1))  # N x 1
        weights = torch.diag(torch.exp(-self.beta * time * L))  # N x N
        homogeneous = Q.mm(weights).mm(Q_inv_x0)

        Q_inv_r = (Q.transpose(0, 1)).mm(self.r.view(-1, 1))

        top_eigen = L[-1]
        approx = 1 / self.beta / top_eigen * torch.exp(self.beta * time * top_eigen) - 1 / (
            self.beta**2
        ) / (top_eigen**2) * (torch.exp(self.beta * time * top_eigen) - 1)
        inhomogeneous = Q.mm(weights).mm(Q_inv_r) * approx
        return homogeneous + inhomogeneous

    @staticmethod
    def reconstruction_loss(x_reconstructed, x):
        loss = nn.MSELoss(reduction="sum")
        return loss(x_reconstructed.squeeze(), x.squeeze())

    @staticmethod
    def compute_laplacian(adjacency: Tensor) -> Tensor:
        off_diag = adjacency - torch.diagonal(adjacency, 0) * torch.eye(adjacency.size(0))  # N x N
        degree = torch.diag(torch.sum(off_diag, 1))  # N x N
        return degree - off_diag

    def loss_function(self, x_t_pred: Tensor, x_t: Tensor) -> Tensor:
        return self.reconstruction_loss(x_t_pred, x_t)

    def return_eigens(self) -> tuple:
        return self.lapl_eigs[0][0], self.lapl_eigs[1][0]


class NetworkDiffusionExpSource(nn.Module):
    def __init__(self, x_0, Covs):
        # configurations
        super().__init__()
        self.x_0 = x_0
        self.beta = Parameter(torch.randn(1), requires_grad=True)
        self.dimension = max(self.x_0.size(0), self.x_0.size(1))
        self.a = Parameter(torch.randn(self.dimension), requires_grad=True)
        self.xi = Parameter(torch.randn(1), requires_grad=True)
        self.Covs = Covs

        lapl1 = self.compute_laplacian(self.Covs[0])
        lapl2 = self.compute_laplacian(self.Covs[1])
        L1, Q1 = torch.linalg.eigh(lapl1)
        L2, Q2 = torch.linalg.eigh(lapl2)

        self.lapl_eigs = ((L1, Q1), (L2, Q2))

    def forward(self, time: float, late: bool):
        if late == False:
            L, Q = self.lapl_eigs[0]
        else:
            L, Q = self.lapl_eigs[1]
        Q_inv_x0 = (Q.transpose(0, 1)).mm(self.x_0.view(-1, 1))  # N x 1
        weights = torch.diag(torch.exp(-self.beta * time * L))  # N x N
        homogeneous = Q.mm(weights).mm(Q_inv_x0)

        Q_inv_a = (Q.transpose(0, 1)).mm(self.a.view(-1, 1))

        top_eigen = L[-1]
        approx = 1 / (self.beta * top_eigen + self.xi) * (
            torch.exp(time * (self.beta * top_eigen + self.xi)) - 1
        ) - 1 / (self.beta * top_eigen) * (torch.exp(self.beta * time * top_eigen) - 1)
        inhomogeneous = Q.mm(weights).mm(Q_inv_a) * approx
        return homogeneous + inhomogeneous

    @staticmethod
    def reconstruction_loss(x_reconstructed, x):
        loss = nn.MSELoss(reduction="sum")
        return loss(x_reconstructed.squeeze(), x.squeeze())

    @staticmethod
    def compute_laplacian(adjacency: Tensor) -> Tensor:
        off_diag = adjacency - torch.diagonal(adjacency, 0) * torch.eye(adjacency.size(0))  # N x N
        degree = torch.diag(torch.sum(off_diag, 1))  # N x N
        return degree - off_diag

    def loss_function(self, x_t_pred: Tensor, x_t: Tensor) -> Tensor:
        return self.reconstruction_loss(x_t_pred, x_t)

    def return_eigens(self) -> tuple:
        return self.lapl_eigs[0][0], self.lapl_eigs[1][0]
