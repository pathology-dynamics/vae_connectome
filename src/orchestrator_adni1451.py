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
from utils import plot_gd_pred_biomarkers_real
import random


def get_df(data_path: str, name_path: str) -> tuple:
    df = pd.read_csv(data_path)
    names_df = pd.read_csv(name_path)

    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"])
    df["time_gt"] = df["EXAMDATE"].dt.year + df["EXAMDATE"].dt.day_of_year / 365

    rois_columns = ["RID", "time_gt"]
    attrs = []
    Nbm = 0
    for name in list(names_df.FS_label):
        name = name.replace("-", "_")
        attrs.append(name.upper() + "_SUVR")
        Nbm += 1

    rois_columns += attrs

    df_rois = df[rois_columns].copy().fillna(0)

    return df_rois, attrs, rois_columns, Nbm


def define_path_name(
    time: str, rid: int, L: int, Nbiom: int, random_seed: int, model_name: str, soft_constr: int
) -> str:
    path = (
        "../results/figs/{}/".format(L)
        + "rid={}".format(rid)
        + "_soft_constr={}".format(soft_constr)
        + "_time={}".format(time)
        + "_Nbiom={}".format(Nbiom)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + ".png"
    )
    return path


def run(
    data_path: str,
    name_path: str,
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
    train_len: int,
    soft_constr: int,
    save_folder: str,
    adaptive_lapl: int,
    vmax: float,
) -> None:

    samples_point = 0
    target_size = 2 * train_len

    # get dataframe
    df_rois_tau, attr, rois_cols, Nbiom = get_df(data_path, name_path)
    cov_matrix = [torch.zeros(Nbiom, Nbiom), torch.zeros(Nbiom, Nbiom)]

    # df_rois_tau = df_rois_tau[np.random.default_rng(seed=42).permutation(df_rois_tau.columns.values)]

    eigen_vals_df = pd.DataFrame(columns=["time_zone"] + attr)

    if rid == -1:
        ind_timestmaps = df_rois_tau.groupby("RID").count()["time_gt"]
        rid_list = ind_timestmaps[ind_timestmaps == target_size].index.to_list()
    else:
        rid_list = [rid]

    samples_point += len(rid_list)

    for rid_ in rid_list:
        df_rois_tau_rid = df_rois_tau[df_rois_tau.RID == rid_]
        if prior == "eye":
            Sigma = torch.eye(Nbiom)

        split_id_lb = 0
        split_id_ub = train_len

        auto_mats = []
        for i in range(2):
            df_rid_i = df_rois_tau_rid.iloc[split_id_lb:split_id_ub]
            split_id_lb = split_id_ub
            split_id_ub += train_len

            x_0 = torch.Tensor(df_rid_i[attr].iloc[0].values).view(1, -1)
            t_0 = torch.Tensor([df_rid_i["time_gt"].iloc[0]])
            x_i = torch.Tensor(df_rid_i[attr].iloc[1:].values)
            time_i = torch.tensor(df_rid_i["time_gt"].iloc[1:].values)

            # train the model
            auto_corr_, _ = train(
                x_i,
                time_i,
                x_0,
                t_0,
                Sigma,
                rid_,
                None,
                sample_size=sample_size,
                epochs=epochs,
                lr=lr,
                verbal=verbal,
                beta=beta,
                checkpoint=checkpoint,
                model_name=model_name,
                soft_constr=soft_constr,
            )

            auto_mats.append(auto_corr_)
            cov_matrix[i] += auto_corr_

        path_early = define_path_name(
            "early", rid_, save_folder, Nbiom, random_seed, model_name, soft_constr
        )
        path_late = define_path_name(
            "late", rid_, save_folder, Nbiom, random_seed, model_name, soft_constr
        )

        path_diff = define_path_name(
            "diff", rid_, save_folder, Nbiom, random_seed, model_name, soft_constr
        )

        plot_cov_mat(auto_mats[0], path_early, annot=False, vmin=0, vmax=vmax)
        plot_cov_mat(auto_mats[1], path_late, annot=False, vmin=0, vmax=vmax)
        diff_auto = auto_mats[0] - auto_mats[1]
        plot_cov_mat(
            diff_auto,
            path_diff,
            annot=False,
            vmin=diff_auto.min().item(),
            vmax=diff_auto.max().item(),
        )

        # train and evaluate meta models using auto_corrs

        df_rid_pred, eigen1, eigen2 = train_eval_meta_model(
            df_rois_tau_rid,
            rid_,
            attr,
            auto_mats,
            1,
            train_len=train_len,
            lr=0.009,
            val_len=0,
            epochs=20000,
            checkpoint=200,
            random_search=40,
            verbal=verbal,
            tol=0.01,
            optim="Adam",
            model_name=model_name,
        )

        df_rid_pred_name = (
            "../results/data/{}/".format(save_folder)
            + "rid={}".format(rid_)
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
                df_rois_tau_rid,
                rid_,
                attr,
                auto_mats,
                0,
                train_len=train_len,
                lr=0.009,
                val_len=0,
                epochs=20000,
                checkpoint=200,
                random_search=40,
                verbal=verbal,
                tol=0.01,
                optim="Adam",
                model_name=model_name,
            )
            # plot_gd_pred_biomarkers(
            #     attr, df_rid_pred_cou, df_rid, Nbiom, expr, rid_, random_seed, model_name, L, 0
            # )
            df_rid_pred_cou_name = (
                "../results/data/{}/".format(save_folder)
                + "rid={}".format(rid_)
                + "_Nbiom={}".format(Nbiom)
                + "_random_seed={}".format(random_seed)
                + "_model={}".format(model_name)
                + "_adalapl={}".format(0)
                + "_val_len={}".format(val_len)
                + "_pred.csv"
            )
            df_rid_pred_cou.to_csv(df_rid_pred_cou_name)

        # plot_gd_pred_biomarkers_real(
        #     df_rid_pred,
        #     df_rois_tau_rid,
        #     attr,
        #     len(attr),
        #     rid_,
        #     random_seed,
        #     model_name,
        #     save_folder,
        # )

        eigen_vals_df.loc[len(eigen_vals_df.index)] = ["early"] + eigen1.tolist()
        eigen_vals_df.loc[len(eigen_vals_df.index)] = ["late"] + eigen2.tolist()

    path_diff_agg = define_path_name(
        "diff_gg", -1, save_folder, Nbiom, random_seed, model_name, soft_constr
    )
    mat_diff = (cov_matrix[0] - cov_matrix[1]) / samples_point
    plot_cov_mat(
        mat_diff, path_diff_agg, annot=False, vmin=mat_diff.min().item(), vmax=mat_diff.max().item()
    )

    eigen_csv_path = (
        "../results/data/{}/".format(save_folder)
        + "soft_constr={}".format(soft_constr)
        + "_Nbiom={}".format(Nbiom)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + "_eiganvals.csv"
    )
    eigen_vals_df.to_csv(eigen_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--data-path", type=str, required=True)
    parser.add_argument("-q", "--name-path", type=str, required=True)
    parser.add_argument("-i", "--rid", type=int, default=29)
    parser.add_argument("-r", "--random-seed", type=int, default=111)
    parser.add_argument("-s", "--sample-size", type=int, default=100)
    parser.add_argument("-e", "--epochs", type=int, default=4000)
    parser.add_argument("-l", "--lr", type=float, default=0.01)
    parser.add_argument("-v", "--verbal", type=int, default=1)
    parser.add_argument("-p", "--prior", type=str, default="eye")
    parser.add_argument("-b", "--beta", type=float, default=0.01)
    parser.add_argument("-c", "--checkpoint", type=int, default=50)
    parser.add_argument("-m", "--model-name", type=str, default="no-source")
    parser.add_argument("-w", "--train-len", type=int, default=2)
    parser.add_argument("-t", "--validation-len", type=int, default=1)
    parser.add_argument("-y", "--soft-constr", type=int, default=1)
    parser.add_argument("-g", "--save-folder", type=str, default="adni1451")
    parser.add_argument("-u", "--vmax", type=float, default=1)
    parser.add_argument("-z", "--adaptive-laplacian", type=int, default=1)

    args = parser.parse_args()
    data_path = args.data_path
    name_path = args.name_path
    rid = args.rid
    random_seed = args.random_seed
    sample_size = args.sample_size
    epochs = args.epochs
    lr = args.lr
    verbal = args.verbal
    prior = args.prior
    beta = args.beta
    checkpoint = args.checkpoint
    model_name = args.model_name
    train_len = args.train_len
    val_len = args.validation_len
    soft_constr = args.soft_constr
    save_folder = args.save_folder
    vmax = args.vmax
    adaptive_lapl = args.adaptive_laplacian

    if random_seed > -1:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    run(
        data_path,
        name_path,
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
        train_len,
        soft_constr,
        save_folder,
        adaptive_lapl,
        vmax,
    )
