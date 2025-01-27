from model import Probabilistic_NetworkDiffusion
from meta_model import NetworkDiffusion, NetworkDiffusionLinearSource, NetworkDiffusionExpSource
import torch
from torch import Tensor
from torch.optim import Adam, RMSprop
import argparse
from utils import synthetic_generator, plot_gd_pred_biomarkers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pandas import DataFrame
import pandas as pd


def plot_cov_mat(adjacency, path, vmin=0, vmax=1, annot=True):
    mask = np.triu(np.ones_like(adjacency))
    ax = sns.heatmap(adjacency, linewidth=0.5, vmin=vmin, vmax=vmax, mask=mask, annot=annot)
    plt.savefig(path)
    plt.close()


def define_path_name(
    time: str,
    rid: int,
    L: int,
    expr: int,
    Nbiom: int,
    random_seed: int,
    model_name: str,
    soft_constr: int,
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
        + ".png"
    )
    return path


def train_eval_meta_model(
    df_rid: DataFrame,
    rid: int,
    attr: list,
    auto_corrs: list,
    adaptive_lapl: int,
    train_len=2,
    val_len=1,
    epochs=10000,
    checkpoint=100,
    random_search=40,
    lr=0.01,
    model_name="no-source",
    tol=0.0005,
    verbal=1,
    optim="RMSprop",
) -> Tensor:
    train_end = train_len * 2
    x_0 = torch.Tensor(df_rid[attr].iloc[0].values).view(1, -1)
    t_0 = df_rid["time_gt"].iloc[0]

    times_meta = df_rid["time_gt"].iloc[1:train_end].values
    x_ts = torch.Tensor(df_rid[attr].iloc[1:train_end].values)
    times_meta_shift = times_meta - t_0
    times_meta_eval = df_rid["time_gt"].iloc[train_end:].values
    x_ts_eval = torch.Tensor(df_rid[attr].iloc[train_end:].values)

    times_meta_eval_shift = times_meta_eval - t_0

    # repeat the following optim multiple time due to nonconvexity

    optim_loss = np.inf
    optim_model = None
    split_point = train_len - 1
    for ran in range(random_search):
        if model_name == "no-source":
            model = NetworkDiffusion(x_0, auto_corrs)
        elif model_name == "LinearSource":
            model = NetworkDiffusionLinearSource(x_0, auto_corrs)
        elif model_name == "ExpSource":
            model = NetworkDiffusionExpSource(x_0, auto_corrs)
        else:
            raise Exception("{} is not a valid meta model".format(model_name))

        if optim == "RMSprop":
            optimizer = RMSprop(model.parameters(), lr=lr)
        elif optim == "Adam":
            optimizer = Adam(model.parameters(), lr=lr)
        cov_ix = False
        model.train()
        loss_prev_iter = np.inf
        for epoch in range(epochs):
            optimizer.zero_grad()
            overall_loss = torch.zeros(1)
            for i in range(0, train_end - 1):
                if i == split_point and adaptive_lapl == 1:
                    cov_ix = True
                x_i_pred = model(times_meta_shift[i], cov_ix)
                overall_loss += model.loss_function(x_i_pred, x_ts[i, :])
            overall_loss /= train_end - 1
            if verbal == 1 and epoch % checkpoint == 0:
                print(
                    "Iteration "
                    + str(epoch + 1)
                    + " of "
                    + str(epochs)
                    + " || Cost (fit): %.2f" % overall_loss.item()
                )
            if overall_loss.isinf().item() == True:
                break
            elif np.abs(loss_prev_iter - overall_loss.item()) < tol or overall_loss.isnan().item():
                break

            loss_prev_iter = overall_loss.item()

            overall_loss.backward()
            optimizer.step()
        if overall_loss.isnan().item() == False:
            if overall_loss.item() < optim_loss:
                optim_loss = overall_loss.item()
                optim_model = model

        else:
            print("solution overflows, skipped")

    print("Optimal model's loss: {}".format(optim_loss))

    optim_model.eval()
    cov_ix = False
    df_rid_pred = pd.DataFrame(columns=["RID"] + attr + ["mse", "time_gt"])
    for i in range(0, train_end - 1):
        if i == split_point and adaptive_lapl == 1:
            cov_ix = True
        x_i_pred = optim_model(times_meta_shift[i], cov_ix)
        mse_loss = optim_model.loss_function(x_i_pred, x_ts[i, :])
        df_rid_pred.loc[i] = (
            [rid]
            + x_i_pred.detach().squeeze().tolist()
            + [mse_loss.detach().item()]
            + [times_meta[i]]
        )
    if val_len > 0:
        for j in range(val_len):
            x_i_pred_eval = optim_model(times_meta_eval_shift[j], cov_ix)
            mse_loss = optim_model.loss_function(x_i_pred_eval, x_ts_eval[j, :])
            df_rid_pred.loc[j + train_end - 1] = (
                [rid]
                + x_i_pred_eval.detach().squeeze().tolist()
                + [mse_loss.detach().item()]
                + [times_meta_eval[j]]
            )

    eigen1, eigen2 = model.return_eigens()

    return df_rid_pred, eigen1, eigen2


def train(
    x_t: Tensor,
    time_t: Tensor,
    x_0: Tensor,
    t_0: Tensor,
    Sigma: Tensor,
    rid: int,
    expr: int,
    sample_size=50,
    epochs=4000,
    lr=0.01,
    verbal=1,
    beta=0.01,
    checkpoint=50,
    model_name="no-source",
    soft_constr=1,
) -> Tensor:
    model = Probabilistic_NetworkDiffusion(x_0, Sigma, sample_size, beta, soft_constr)
    loss_checkpoint = []
    x0xt = x_0.mm(x_t.transpose(0, 1)).view(-1, 1)
    overall_loss = 0
    count = 1
    optimizer = Adam(model.parameters(), lr=lr)
    time_t_abs = time_t - t_0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_t_pred, x0xt_pred, phi = model(time_t_abs)
        loss_fit, loss_constr, loss_KL = model.loss_function(
            x_t_pred, x_t, x0xt_pred.view(x0xt_pred.size(0), x0xt_pred.size(1), 1), x0xt
        )
        loss = loss_fit + loss_constr + loss_KL
        overall_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
        if verbal == 1 and epoch % checkpoint == 0:
            print(
                "Iteration "
                + str(epoch + 1)
                + " of "
                + str(epochs)
                + " || Cost (DKL): %.2f" % loss_KL.item()
                + " - Cost (fit): %.2f" % loss_fit.item()
                + " - Cost (constr): %.2f" % loss_constr.item()
                + "|| Batch size %d" % sample_size
            )

            loss_checkpoint.append(loss_fit.item())

    # auto_corr = model.get_connectome_posterior()
    auto_corr = model.get_connectome_posterior_sampling()
    return auto_corr, loss_checkpoint


def run(
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
    eigen_vals_df = pd.DataFrame(columns=["time_zone"] + ["dim_{}".format(i) for i in range(Nbiom)])
    samples_point = 0
    checkpoint_ls = [[], []]
    target_size = 2 * train_len + val_len
    for expr in range(experiments):
        df, attr = synthetic_generator(Nbiom, seed=random_seed, L=abs(L), max_range=target_size + 1)
        df_name = (
            "../results/data/{}/".format(L)
            + "expr={}".format(expr)
            + "_Nbiom={}".format(Nbiom)
            + "_random_seed={}".format(random_seed)
            + "_model={}".format(model_name)
            + ".csv"
        )
        df.to_csv(df_name)
        if rid == -1:
            ind_timestmaps = df.groupby("RID").count()["time"]
            rid_list = ind_timestmaps[ind_timestmaps == target_size].index.to_list()
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

            path_early = define_path_name(
                "early", rid_, L, expr, Nbiom, random_seed, model_name, soft_constr
            )
            path_late = define_path_name(
                "late", rid_, L, expr, Nbiom, random_seed, model_name, soft_constr
            )

            path_diff = define_path_name(
                "diff", rid_, L, expr, Nbiom, random_seed, model_name, soft_constr
            )

            plot_cov_mat(auto_mats[0], path_early, annot=True, vmin=0, vmax=vmax)
            plot_cov_mat(auto_mats[1], path_late, annot=True, vmin=0, vmax=vmax)
            diff_auto = auto_mats[0] - auto_mats[1]
            plot_cov_mat(
                diff_auto,
                path_diff,
                annot=True,
                vmin=diff_auto.min().item(),
                vmax=diff_auto.max().item(),
            )

            # train and evaluate meta models using auto_corrs

            df_rid_pred, eigen1, eigen2 = train_eval_meta_model(
                df_rid,
                rid_,
                attr,
                auto_mats,
                1,
                train_len=train_len,
                val_len=val_len,
                model_name=model_name,
            )
            plot_gd_pred_biomarkers(
                attr, df_rid_pred, df_rid, Nbiom, expr, rid_, random_seed, model_name, L, 1
            )
            df_rid_pred_name = (
                "../results/data/{}/".format(L)
                + "rid={}".format(rid_)
                + "_expr={}".format(expr)
                + "_Nbiom={}".format(Nbiom)
                + "_random_seed={}".format(random_seed)
                + "_model={}".format(model_name)
                + "_adalapl={}".format(1)
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
                    model_name=model_name,
                )
                plot_gd_pred_biomarkers(
                    attr, df_rid_pred_cou, df_rid, Nbiom, expr, rid_, random_seed, model_name, L, 0
                )
                df_rid_pred_cou_name = (
                    "../results/data/{}/".format(L)
                    + "rid={}".format(rid_)
                    + "_expr={}".format(expr)
                    + "_Nbiom={}".format(Nbiom)
                    + "_random_seed={}".format(random_seed)
                    + "_model={}".format(model_name)
                    + "_adalapl={}".format(0)
                    + "_pred.csv"
                )
                df_rid_pred_cou.to_csv(df_rid_pred_cou_name)
            eigen_vals_df.loc[len(eigen_vals_df.index)] = ["early"] + eigen1.tolist()
            eigen_vals_df.loc[len(eigen_vals_df.index)] = ["late"] + eigen2.tolist()

    path_diff_agg = define_path_name(
        "diff_gg", -1, L, -1, Nbiom, random_seed, model_name, soft_constr
    )
    mat_diff = (cov_matrix[0] - cov_matrix[1]) / samples_point
    plot_cov_mat(
        mat_diff, path_diff_agg, annot=True, vmin=mat_diff.min().item(), vmax=mat_diff.max().item()
    )

    pickle_path = (
        "../results/data/synthetic/"
        + "L={}".format(L)
        + "_soft_constr={}".format(soft_constr)
        + "_Nbiom={}".format(Nbiom)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + ".pkl"
    )
    with open(pickle_path, "wb") as f:
        pickle.dump(checkpoint_ls, f)

    eigen_csv_path = (
        "../results/data/synthetic/"
        + "L={}".format(L)
        + "_soft_constr={}".format(soft_constr)
        + "_Nbiom={}".format(Nbiom)
        + "_random_seed={}".format(random_seed)
        + "_model={}".format(model_name)
        + "_eigenvals.csv"
    )
    eigen_vals_df.to_csv(eigen_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
