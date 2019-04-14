import numpy as np
import matplotlib.pyplot as plt


def main():
    file_list = ["exp_10_30_0.01_wass_out.npy", "exp_10_30_0.0001_wass_out.npy",
                 "exp_10_30_0.09_wass_out.npy", "exp_10_30_0.25_wass_out.npy", "exp_10_30_0.0025_wass_out.npy",
                 "exp_10_30_1.0_wass_out.npy", "exp_10_30_4.0_wass_out.npy",
                 "exp_10_30_25.0_wass_out.npy"]

    data = {}
    for file in file_list:
        file_data = np.load("wass-dist/"+file).item()
        data[float(file[10:-13])] = {}
        for key in file_data.keys():
            data[float(file[10:-13])][key] = list(file_data[key].values())

    return data


def plot_all(data):
    x_ax = [100, 500, 1000, 2000, 5000, 10000, 50000, 100000, 200000, 400000, 800000]
    f, axarr = plt.subplots(int(len(data.keys())/2), 2, sharex=True, sharey=True)
    keys = sorted(data.keys())
    inner_keys = list(data[keys[0]].keys())
    for i in [10, 50, 500, 2000, 20000]:
        inner_keys.remove(i)
    for i in range(len(keys)):
        for key in inner_keys:
            axarr[int(i/2), i % 2].plot(x_ax, data[keys[i]][key], label=key)
            axarr[int(i/2), i % 2].set_ylim([0, 1])
            axarr[int(i/2), i % 2].set_title("Covariance = "+str(keys[i]), fontsize=7)
            axarr[int(i/2), i % 2].set_xlabel("Sample Size", fontsize=6)
            axarr[int(i/2), i % 2].set_ylabel("Distance", fontsize=6)
            for item in axarr[int(i/2), i % 2].get_xticklabels():
                item.set_fontsize(6)
            for item in axarr[int(i/2), i % 2].get_yticklabels():
                item.set_fontsize(6)
    f.subplots_adjust(hspace=0.5)
    plt.legend(fontsize=3)
    plt.show()
    return axarr


if __name__ == "__main__":
    data = main()
    plot_all(data)
