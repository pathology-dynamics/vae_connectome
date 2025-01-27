import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from pandas import DataFrame
import numpy as np
from DataSimulation import DataGenerator
import matplotlib.pyplot as plt


def sample_from_gaussian(sample_size: int, mean: Tensor, cov: Tensor) -> Tensor:
    """
    resample batch i.i.d samples froms a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.
    Example:
                    >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
                    >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
                    tensor([-0.2102, -0.5429])

    Args:
                    batch_size: int
                    loc (Tensor): mean of the distribution
                    covariance_matrix (Tensor): positive-definite covariance matrix

    return:
                                    tensor[(batch sample size, mean size )]

    """
    distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)
    return distrib.sample((sample_size,))


def compute_sigma(x):
    """
    sigma = sum_{j=1:n}sum_{i=1:j} x_i x_j (x_i - x_j)^2
    """
    n = x.size(0)
    sigma = torch.tensor(0.0)  # Initialize sigma to 0

    for j in range(n):
        for i in range(j):
            term = x[i] * x[j] * (x[i] - x[j]) ** 2
            sigma = sigma + term
    sigma = sigma / x.dot(x)
    return sigma


def compute_sigma_test(x):

    return x.sum() / x.dot(x) * torch.pow(x, 3).sum() - x.dot(x)


def compute_sigma_s(x):
    s, n = x.size()
    one = torch.ones(n).view(-1, 1)
    dot_prod = x.mm(x.transpose(0, 1))
    diag_sq = torch.diag(dot_prod, 0).view(-1, 1)
    return x.mm(one) * torch.pow(x, 3).mm(one) / diag_sq - diag_sq


def compute_grad_sigma(x):
    r"""
    (\partial sigma/partial \x)_j  = sum_{i=1:j}
    """
    n = x.size(0)
    grad_x = torch.zeros(n)  # initialize the gradient
    for j in range(n):
        aggre = torch.tensor(0.0)
        for i in range(j):
            aggre += (x[i] - x[j]) * (x[i] ** 2 - 3 * x[i] * x[j])
        grad_x[j] = aggre

    return grad_x


def plot_biomarkers(
    df: DataFrame, L: int, n_markers: int, rid: int, random_seed: int, model: str
) -> None:
    modulo = n_markers % 2
    if modulo == 0:
        last_row = int(n_markers / 2)
        fig, axs = plt.subplots(last_row, 2)
    else:
        last_row = int((n_markers + 1) / 2)
        fig, axs = plt.subplots(last_row, 2)

    count = 0
    for i in range(int((n_markers - modulo) / 2)):
        for j in range(2):
            name = "biom_{}".format(count)
            axs[i, j].plot(df.time_gt, df[name])
            axs[i, j].set_title(name)
            count += 1

    if modulo > 0:
        i = last_row - 1
        name = "biom_{}".format(count)
        axs[i, 0].plot(df.time_gt, df[name])
        axs[i, 0].set_title(name)

    for ax in axs.flat:
        ax.label_outer()

    fig.savefig(
        "../results/figs/{}/{}d_synthetic_{}_random_seed_{}_rid_{}.png".format(
            L, n_markers, model, random_seed, rid
        )
    )
    plt.close()

    return None


def plot_gd_pred_biomarkers(
    attr: list,
    df: DataFrame,
    df_ground: DataFrame,
    n_markers: int,
    expr: int,
    rid_: int,
    random_seed: int,
    model_name: str,
    L: int,
    adaptive_lapl: int,
):

    fig, axs = plt.subplots(2, 2)

    count = 0
    
    for i in range(2):
        for j in range(2):
            name= attr[count]
            axs[i, j].plot(df_ground.time_gt, df_ground[name], linewidth=4)
            axs[i, j].plot(df.time_gt, df[name], linestyle="dashed")
            axs[i, j].set_title(name)
            count += 1
            

    for ax in axs.flat:
        ax.label_outer()

    fig.savefig(
        "../results/figs/{}/".format(L)
        + "rid={}".format(rid_)
        + "_expr={}".format(expr)
        + "_Nbiom={}".format(n_markers)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + "_adalapl={}".format(adaptive_lapl)
        + "_pred.png"
    )

    plt.close()

    return None


def plot_gd_pred_biomarkers_real(
    df: DataFrame,
    df_ground: DataFrame,
    attr: list,
    n_markers: int,
    rid_: int,
    random_seed: int,
    model_name: str,
    L: int,
):

    hal_ = 3
    fig, axs = plt.subplots(hal_, 2)

    count = 0
    for i in range(hal_):
        for j in range(2):
            name = attr[count]
            axs[i, j].plot(df_ground.time_gt, df_ground[name], linewidth=4)
            axs[i, j].plot(df.time_gt, df[name], linestyle="dashed")
            axs[i, j].set_title(name.replace("_SUVR", ""))
            count += 1

    for ax in axs.flat:
        ax.label_outer()

    fig.savefig(
        "../results/figs/{}/".format(L)
        + "rid={}".format(rid_)
        + "_Nbiom={}".format(n_markers)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + "_pred.png"
    )

    plt.close()

    return None


def synthetic_generator(
    Nbiom, max_=15, seed=111, noise=0.1, Nsubs=30, L=1, max_range=6
) -> DataFrame:
    if seed == -1:
        seed = None
    # time interval for the trajectories
    interval = [-max_, max_]

    # Number of biomarkers
    Nbiom = 4
    # Gaussian observational noise

    # Creating random parameter sets for each biomarker's progression
    flag = 0
    while flag != 1:
        CurveParam = []
        for i in range(Nbiom):
            CurveParam.append([L, 1 * (0.5 + np.random.rand()), noise])
            if CurveParam[i][1] > 0.0:
                flag = 1

    dg = DataGenerator(Nbiom, interval, CurveParam, Nsubs, seed, max_range=max_range)
    df = dg.get_df()
    attr = ["biom_{}".format(i) for i in range(Nbiom)]
    df[attr] = df[attr] + 1
    return df, attr
