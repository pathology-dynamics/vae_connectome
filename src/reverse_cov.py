import argparse
from model import Probabilistic_NetworkDiffusion, Probabilistic_NetworkDiffusion_LinearSource
from utils import synthetic_generator
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_cov_fig(
    model: Probabilistic_NetworkDiffusion,
    time_i: float,
    rid: int,
    random_seed: int,
    vmax: int,
    L: int,
):
    cov = model.get_connectome_posterior().detach()
    cov = cov - torch.diag(cov) * torch.eye(Nbiom)

    adjacency = cov.numpy()
    mask = np.triu(np.ones_like(adjacency))
    ax = sns.heatmap(adjacency, linewidth=0.5, vmin=0, vmax=vmax, mask=mask, annot=True)
    path_of_fig = (
        "../results/figs/{}/time_t={}".format(L, time_i.to(int).tolist())
        + "_RID={}".format(rid)
        + "_Nbiom={}".format(Nbiom)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + ".png"
    )
    plt.savefig(path_of_fig)
    plt.close()


def run(
    Nbiom: int,
    rid: int,
    random_seed: int,
    vmax: float,
    prior: str,
    sample_size: int,
    beta: float,
    model_name: str,
    L: int,
    window_len: int,
) -> None:
    df_name = (
        "../results/data/"
        + "_L={}".format(L)
        + "_Nbiom={}".format(Nbiom)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + ".csv"
    )
    attr = ["biom_{}".format(i) for i in range(Nbiom)]
    df = pd.read_csv(df_name)

    if rid == -1:
        ind_timestmaps = df.groupby("RID").count()["time"]
        rid_list = ind_timestmaps[ind_timestmaps >= window_len].index.to_list()
    else:
        rid_list = [rid]

    for rid_ in rid_list:
        df_rid = df[df.RID == rid_]
        if prior == "eye":
            Sigma = torch.eye(Nbiom)

        split_id_lb = 0
        split_id_ub = int(window_len / 2)

        for i in range(2):
            df_rid_i = df_rid.iloc[split_id_lb:split_id_ub]
            split_id_lb = split_id_ub
            split_id_ub += int(window_len / 2)
            x_0 = torch.Tensor(df_rid_i[attr].iloc[0].values).view(1, -1)
            time_i = torch.tensor(df_rid_i["time_gt"].iloc[1:].values)
            path_of_model = (
                "../results/data/{}/time_t={}".format(L, time_i.to(int).tolist())
                + "_RID={}".format(int(rid_))
                + "_Nbiom={}".format(Nbiom)
                + "_random_seed={}".format(random_seed)
                + "_model={}".format(model_name)
                + ".pkl"
            )
            if model_name == "no-source":
                model = Probabilistic_NetworkDiffusion(x_0, Sigma, sample_size, beta)
            elif model_name == "LinearSource":
                model = Probabilistic_NetworkDiffusion_LinearSource(x_0, Sigma, sample_size, beta)
            else:
                raise Exception("{} is not a valid model".format(model_name))
            model.load_state_dict(torch.load(path_of_model))
            model.eval()
            plot_cov_fig(model, time_i, int(rid_), random_seed, vmax, L)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--rid", type=int, default=29)
    parser.add_argument("-n", "--Nbiom", type=int, default=4)
    parser.add_argument("-r", "--random-seed", type=int, default=111)
    parser.add_argument("-v", "--vmax", type=float, default=2)
    parser.add_argument("-p", "--prior", type=str, default="eye")
    parser.add_argument("-s", "--sample-size", type=int, default=100)
    parser.add_argument("-b", "--beta", type=float, default=0.01)
    parser.add_argument("-m", "--model-name", type=str, default="no-source")
    parser.add_argument("-d", "--Pathological", type=int, default=1)
    parser.add_argument("-w", "--window-len", type=int, default=4)

    args = parser.parse_args()
    rid = args.rid
    Nbiom = args.Nbiom
    random_seed = args.random_seed
    vmax = args.vmax
    prior = args.prior
    sample_size = args.sample_size
    beta = args.beta
    model_name = args.model_name
    L = args.Pathological
    window_len = args.window_len

    run(Nbiom, rid, random_seed, vmax, prior, sample_size, beta, model_name, L, window_len)
