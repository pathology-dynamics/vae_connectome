from model import Probabilistic_NetworkDiffusion
from meta_model import NetworkDiffusion
import torch
from torch import Tensor
from torch.optim import Adam, RMSprop
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pandas import DataFrame
import pandas as pd
from main import train_eval_meta_model, train, plot_cov_mat
from utils import plot_gd_pred_biomarkers
import random
from meta_model import NetworkDiffusion


def get_df(data_path: str) -> tuple:
    df = pd.read_csv(data_path)
    return df


def define_path_name(
    time: str,
    rid: int,
    L: int,
    expr: int,
    Nbiom: int,
    random_seed: int,
    model_name: str,
    soft_constr: int,
    ven_len: int,
) -> str:
    path = (
        "../results/figs/{}/".format(L)
        + "rid={}".format(rid)
        + "_soft_constr={}".format(soft_constr)
        + "_time={}".format(time)
        + "_expr={}".format(expr)
        + "_Nbiom={}".format(Nbiom)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + "_ven_len={}".format(ven_len)
        + ".png"
    )
    return path


def run(
    data_path: str,
    Nbiom: int,
    prior: str,
    rid: int,
    random_seed: int,
    sample_size: int,
    epochs: int,
    lr: float,
    verbal: int,
    beta: float,
    checkpoint: int,
    model_name: str,
    L: int,
    train_len: int,
    val_len: int,
    experiments: int,
    soft_constr: int,
    adaptive_lapl: int,
    vmax: float,
) -> None:
    cov_matrix = [torch.zeros(Nbiom, Nbiom), torch.zeros(Nbiom, Nbiom)]
    attr = ["Hippocampus", "Ventricles", "Entorhinal", "WholeBrain"]
    eigen_vals_df = pd.DataFrame(columns=["time_zone"] + attr)
    samples_point = 0
    checkpoint_ls = [[], []]

    for expr in range(experiments):
        df = get_df(data_path)
        df = df[df.group == L].copy()
        target_size = 2 * train_len + val_len

        if rid == -1:
            ind_timestmaps = df.groupby("RID").count()["time_gt"]
            rid_list = ind_timestmaps[ind_timestmaps >= target_size].index.to_list()
        else:
            rid_list = [rid]

        samples_point += len(rid_list)

        for rid_ in rid_list:
            df_rid = df[df.RID == rid_]
            if prior == "eye":
                Sigma = torch.eye(Nbiom)

            split_id_lb = 0
            split_id_ub = train_len

            auto_mats = []
            for i in range(2):
                df_rid_i = df_rid.iloc[split_id_lb:split_id_ub]
                split_id_lb = split_id_ub
                split_id_ub += train_len

                x_0 = torch.Tensor(df_rid_i[attr].iloc[0].values).view(1, -1)
                x_i = torch.Tensor(df_rid_i[attr].iloc[1:].values)
                time_i = torch.tensor(df_rid_i["time_gt"].iloc[1:].values)
                t_0 = torch.Tensor([df_rid_i["time_gt"].iloc[0]])

                # train the model
                auto_corr_, loss_check = train(
                    x_i,
                    time_i,
                    x_0,
                    t_0,
                    Sigma,
                    rid_,
                    expr,
                    sample_size=sample_size,
                    epochs=epochs,
                    lr=lr,
                    verbal=verbal,
                    beta=beta,
                    checkpoint=checkpoint,
                    model_name=model_name,
                    soft_constr=soft_constr,
                )
                cov_matrix[i] += auto_corr_
                auto_mats.append(auto_corr_)
                checkpoint_ls[i].append(loss_check)

            # train and evaluate meta models using auto_corrs

            df_rid_pred, eigen1, eigen2 = train_eval_meta_model(
                df_rid,
                rid_,
                attr,
                auto_mats,
                1,
                train_len=train_len,
                val_len=val_len,
                verbal=verbal,
                model_name=model_name,
            )
            # plot_gd_pred_biomarkers(
            #     attr, df_rid_pred, df_rid, Nbiom, expr, rid_, random_seed, model_name, L, 1
            # )
            df_rid_pred_name = (
                "../results/data/{}/".format(L)
                + "rid={}".format(rid_)
                + "_expr={}".format(expr)
                + "_Nbiom={}".format(Nbiom)
                + "_random_seed={}".format(random_seed)
                + "_model={}".format(model_name)
                + "_adalapl={}".format(1)
                + "_val_len={}".format(val_len)
                + "_pred.csv"
            )
            df_rid_pred.to_csv(df_rid_pred_name)
            if adaptive_lapl == 1:
                # train and evaluate counterpart meta models using auto_corrs
                df_rid_pred_cou, _, _ = train_eval_meta_model(
                    df_rid,
                    rid_,
                    attr,
                    auto_mats,
                    0,
                    train_len=train_len,
                    val_len=val_len,
                    verbal=verbal,
                    model_name=model_name,
                )
                # plot_gd_pred_biomarkers(
                #     attr, df_rid_pred_cou, df_rid, Nbiom, expr, rid_, random_seed, model_name, L, 0
                # )
                df_rid_pred_cou_name = (
                    "../results/data/{}/".format(L)
                    + "rid={}".format(rid_)
                    + "_expr={}".format(expr)
                    + "_Nbiom={}".format(Nbiom)
                    + "_random_seed={}".format(random_seed)
                    + "_model={}".format(model_name)
                    + "_adalapl={}".format(0)
                    + "_val_len={}".format(val_len)
                    + "_pred.csv"
                )
                df_rid_pred_cou.to_csv(df_rid_pred_cou_name)

            eigen_vals_df.loc[len(eigen_vals_df.index)] = ["early"] + eigen1.tolist()
            eigen_vals_df.loc[len(eigen_vals_df.index)] = ["late"] + eigen2.tolist()

    path_diff_agg = define_path_name(
        "diff_gg", -1, L, -1, Nbiom, random_seed, model_name, soft_constr, val_len
    )
    mat_diff = (cov_matrix[0] - cov_matrix[1]) / samples_point
    plot_cov_mat(
        mat_diff, path_diff_agg, annot=True, vmin=mat_diff.min().item(), vmax=mat_diff.max().item()
    )
    pickle_path = (
        "../results/data/adni_3d/"
        + "L={}".format(L)
        + "_soft_constr={}".format(soft_constr)
        + "_Nbiom={}".format(Nbiom)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + "_val_len={}".format(val_len)
        + ".pkl"
    )
    with open(pickle_path, "wb") as f:
        pickle.dump(checkpoint_ls, f)

    eigen_csv_path = (
        "../results/data/adni_3d/"
        + "L={}".format(L)
        + "_soft_constr={}".format(soft_constr)
        + "_Nbiom={}".format(Nbiom)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + "_val_len={}".format(val_len)
        + "_eigenvals.csv"
    )
    eigen_vals_df.to_csv(eigen_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--data-path", type=str, required=True)
    parser.add_argument("-i", "--rid", type=int, default=29)
    parser.add_argument("-n", "--Nbiom", type=int, default=4)
    parser.add_argument("-r", "--random-seed", type=int, default=111)
    parser.add_argument("-s", "--sample-size", type=int, default=100)
    parser.add_argument("-e", "--epochs", type=int, default=4000)
    parser.add_argument("-l", "--lr", type=float, default=0.01)
    parser.add_argument("-v", "--verbal", type=int, default=1)
    parser.add_argument("-p", "--prior", type=str, default="eye")
    parser.add_argument("-b", "--beta", type=float, default=0.01)
    parser.add_argument("-c", "--checkpoint", type=int, default=50)
    parser.add_argument("-m", "--model-name", type=str, default="no-source")
    parser.add_argument("-d", "--Pathological", type=int, default=1)
    parser.add_argument("-w", "--train-len", type=int, default=2)
    parser.add_argument("-t", "--validation-len", type=int, default=1)
    parser.add_argument("-x", "--experiments", type=int, default=40)
    parser.add_argument("-y", "--soft-constr", type=int, default=1)
    parser.add_argument("-z", "--adaptive-laplacian", type=int, default=1)
    parser.add_argument("-u", "--vmax", type=float, default=1)

    args = parser.parse_args()
    data_path = args.data_path
    rid = args.rid
    Nbiom = args.Nbiom
    random_seed = args.random_seed
    sample_size = args.sample_size
    epochs = args.epochs
    lr = args.lr
    verbal = args.verbal
    prior = args.prior
    beta = args.beta
    checkpoint = args.checkpoint
    model_name = args.model_name
    L = args.Pathological
    train_len = args.train_len
    val_len = args.validation_len
    experiments = args.experiments
    soft_constr = args.soft_constr
    adaptive_lapl = args.adaptive_laplacian
    vmax = args.vmax

    if random_seed > -1:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    run(
        data_path,
        Nbiom,
        prior,
        rid,
        random_seed,
        sample_size,
        epochs,
        lr,
        verbal,
        beta,
        checkpoint,
        model_name,
        L,
        train_len,
        val_len,
        experiments,
        soft_constr,
        adaptive_lapl,
        vmax,
    )
