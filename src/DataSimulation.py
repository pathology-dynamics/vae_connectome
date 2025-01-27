import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


class DataGenerator(object):
    def __init__(self, Nbiom, interval, param, Nsubs, seed=None, max_range=7):
        self.Nbiom = Nbiom
        self.YData = []
        self.YDataNoNoise = []
        self.XData = []
        self.model = []
        self.Nsubs = Nsubs
        self.interval = interval
        self.param = param
        self.shiftData = []
        self.time_shift = []
        self.max_range = max_range
        if seed:
            np.random.seed(seed)

        for i in range(Nbiom):
            tot_range = (interval[1] - interval[0]) / max_range
            shift = np.random.randint(interval[0] + tot_range, interval[1] - tot_range)
            self.shiftData.append(
                [shift + l for l in range(int(float(interval[0]) / 2), int(float(interval[1]) / 2))]
            )
            self.model.append(self.f(self.shiftData[i], param[i][0], param[i][1]))
            self.time_shift.append(tot_range - shift + 1)
            # self.model.append(self.f(range(interval[0], interval[1]), param[i][0], param[i][1]))

        self.XData.append([])
        for k in range(self.Nsubs):
            start = np.random.randint(interval[1] - max_range)
            Nel = np.random.randint(1, max_range)
            sequence = np.sort(random.sample(range(max_range), Nel))
            Xd = [l + start for l in sequence]
            self.XData[0].append(Xd)

        for i in range(Nbiom - 1):
            self.XData.append([])
            self.XData[i + 1] = self.XData[0]

        for i in range(Nbiom):
            self.YData.append([])
            self.YDataNoNoise.append([])
            for l in range(self.Nsubs):
                obs = np.array(self.model[i])[self.XData[i][l]] + param[i][2] * np.random.randn(
                    len(self.XData[i][l])
                )
                self.YData[i].append(obs)
                self.YDataNoNoise[i].append(np.array(self.model[i])[self.XData[i][l]])

        self.ZeroXData = []
        for i in range(self.Nbiom):
            self.ZeroXData.append([])
            for k in range(self.Nsubs):
                self.ZeroXData[i].append(
                    np.array(self.XData[i][k])
                    - float(self.XData[i][k][len(self.XData[i][k]) - 1] + self.XData[i][k][0]) / 2
                )

        for i in range(self.Nbiom):
            for k in range(self.Nsubs):
                self.XData[i][k] = np.array([float(l) for l in self.XData[i][k]])

    def f(self, X, L, k):
        return [L / (1 + np.exp(-k * i)) for i in X]

    def get_df(self):
        # Creating empty data frame for output .csv
        df = pd.DataFrame({"RID": [], "time": []})
        col_biom = {}
        for i in range(self.Nbiom):
            col_biom["biom_" + str(i)] = []
        df = df.join(pd.DataFrame(col_biom))
        # Assigning RID and generated time
        accumulator = []
        for i in range(len(self.ZeroXData[0])):
            for t in range(len(self.ZeroXData[0][i])):
                df = df._append(
                    {
                        "RID": int(i),
                        "time": self.ZeroXData[0][i][t],
                        "time_gt": self.XData[0][i][t],
                    },
                    ignore_index=True,
                )
        # Assigning biomarkers values and zero-centered individual time points
        for b in range(self.Nbiom):
            for i in range(len(self.ZeroXData[b])):
                for t_i, t in enumerate(self.ZeroXData[b][i]):
                    df.loc[(df["RID"] == int(i)) & (df["time"] == t), "biom_" + str(b)] = (
                        self.YData[b][i][t_i]
                    )
        return df

    def plot(self, mode):
        datamin = 0
        datamax = 1

        if mode == "short":
            elements = [el for sublist in self.ZeroXData for item in sublist for el in item]
            datamin = np.min(elements)
            datamax = np.max(elements)
        if mode == "long":
            elements = [el for sublist in self.XData for item in sublist for el in item]
            datamin = np.min([elements])
            datamax = np.max([elements])

        datarange = [el for sublist in self.YData for item in sublist for el in item]
        axes = plt.gca()
        axes.set_xlim([datamin, datamax])
        axes.set_ylim([np.min(datarange) - 0.5, np.max(datarange) + 0.5])
        # fig = plt.figure()
        Blues = plt.get_cmap("prism")

        if mode == "short":
            plt.title("simulated short-term trajectories")
            for i in np.arange(self.Nbiom):
                for k in np.arange(self.Nsubs):
                    plt.plot(
                        self.ZeroXData[i][k],
                        self.YData[i][k],
                        color=Blues(i * 2),
                        linewidth=1,
                        linestyle="--",
                    )
        #  fig.savefig('/Users/mlorenzi/Desktop/ipmc/mode0.png')
        if mode == "long":
            plt.title("simulated long-term trajectories")
            for i in np.arange(self.Nbiom):
                # print(self.shiftData[i])
                plt.plot(
                    range(len(self.model[i])),
                    self.model[i],
                    color=Blues(i * 2),
                    linewidth=3,
                    label="biom.  " + str(i),
                )
                plt.annotate(
                    "biom.  " + str(i),
                    xy=(len(self.model[i]), self.model[i][len(self.model[i]) - 1]),
                    color=Blues(i * 2),
                    textcoords="data",
                )
                for k in np.arange(self.Nsubs):
                    plt.plot(
                        self.XData[i][k],
                        self.YData[i][k],
                        color=Blues(i * 2),
                        linewidth=1,
                        linestyle="--",
                    )
                plt.legend()
        plt.savefig("../results/{}.png".format(mode))
        plt.xlabel("time")
        plt.ylabel("biomarker severity")
        plt.close()
        # fig.show()

    def OutputTimeShift(self):
        time_shift = np.zeros(len(self.XData[0]))
        for l in range(len(self.XData[0])):
            time_shift[l] = self.XData[0][l][0]

        return time_shift
