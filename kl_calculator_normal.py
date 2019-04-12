import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import datetime


def get_input(size, count):
    string = str(size) + "_" + str(count)
    data = np.load("mcmc_norm_out_" + string + ".npy")
    parameters = np.load("mcmc_norm_params_" + string + ".npy").item()
    mean, std_dev = parameters["post_mean"], np.sqrt(parameters["post_variance"])
    return data, mean, std_dev


def create_hist(data):
    interval = [-5, 5]
    bin_count = 1000
    hist, bins = np.histogram(data, bin_count, interval, density=True)
    return hist, bins


def calc_div(hist, bins, mean, std_dev):
    kl_div = 0
    for i in range(len(bins)-1):
        true_prob = norm.pdf(bins[i+1], mean, std_dev) - norm.pdf(bins[i], mean, std_dev)
        estimate = hist[i]
        if estimate > 0 and true_prob > 0:
            kl_div += estimate * np.log(estimate/true_prob)
    return kl_div


def main(size, rep):
    """do the job"""
    start = datetime.datetime.now()
    output = {}
    burn_list = [0, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    length_list = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 400000, 600000, 800000]
    for count in range(1, rep+1):
        data, mean, std_dev = get_input(size, count)
        output[count] = {}
        for burn in burn_list:
            output[count][burn] = {}
            for length in length_list:
                hist, bins = create_hist(data[burn: burn+length])
                div = calc_div(hist, bins, mean, std_dev)
                output[count][burn][length] = div
                print("count ", count, " burn ", burn, " len ", length, " div ", div, " time ", datetime.datetime.now() - start)
    np.save("norm_" + str(size) + "_" + str(rep) + "out.npy", output)
    return output


if __name__ == "__main__":
    main(100, 30)
