import numpy as np
from scipy.stats import norm
from scipy.stats import wasserstein_distance
import datetime


def get_input(size, count):
    string = str(size) + "_" + str(count)
    data = np.load("experiments/mcmc_norm_out_" + string + ".npy")
    parameters = np.load("experiments/mcmc_norm_params_" + string + ".npy").item()
    mean, std_dev = parameters["post_mean"], np.sqrt(parameters["post_variance"])
    return data, mean, std_dev


def create_hist(data, bin_count, hist_range):
    hist, bins = np.histogram(data, bin_count, hist_range, density=True)
    return hist, bins


def calc_kl_div(hist, prob, bin_count):
    kl_div = 0
    for i in range(bin_count):
        true_prob = prob[i]
        estimate = hist[i]
        if estimate > 0 and true_prob > 0:
            kl_div += estimate * np.log(estimate/true_prob)
    return kl_div


def calc_tv(hist, prob, bin_count):
    tv = 0
    for i in range(bin_count):
        true_prob = prob[i]
        estimate = hist[i]
        tv += np.abs(estimate - true_prob)
    return tv


def calc_true(bin_count, hist_range, mean, std_dev):
    scrap, bins = np.histogram([0], bin_count, hist_range, density=True)
    points = (bins[0: bin_count] + bins[1: bin_count+1])/2
    prob = []
    for i in range(len(bins) - 1):
        true_prob = norm.cdf(bins[i + 1], mean, std_dev) - norm.cdf(bins[i], mean, std_dev)
        prob.append(true_prob)
    return points, prob


def calc_wass(points, prob, hist):
    dist = wasserstein_distance(points, points, prob, hist)
    return dist


def main(size, rep, diff):
    """do the job"""
    bin_count = 500
    hist_range = [-5, 5]
    start = datetime.datetime.now()
    divs = {}
    burn_list = [0, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    length_list = [100, 500, 1000, 2000, 5000, 10000, 50000, 100000, 200000, 400000, 800000]
    for count in range(1, rep+1):
        data, mean, std_dev = get_input(size, count)
        points, prob = calc_true(bin_count, hist_range, mean, std_dev)
        divs[count] = {}
        for burn in burn_list:
            divs[count][burn] = {}
            for length in length_list:
                hist, bins = create_hist(data[burn: burn+length], bin_count, hist_range)
                hist_normed = hist/bin_count*(hist_range[1]-hist_range[0])
                if diff == "kl":
                    div = calc_kl_div(hist_normed, prob, bin_count)
                elif diff == "tv":
                    div = calc_tv(hist_normed, prob, bin_count)
                else:
                    div = calc_wass(points, prob, hist_normed)
                divs[count][burn][length] = div
                print("count ", count, " burn ", burn, " len ", length, " div ", div, " time ", datetime.datetime.now() - start)
    output = {}
    for key1 in divs[1].keys():
        output[key1] = {}
        for key2 in divs[1][key1].keys():
            out = []
            for count in range(1, rep+1):
                out.append(divs[count][key1][key2])
            output[key1][key2] = np.average(out)
    np.save("norm_" + str(size) + "_" + str(rep) + "_" + diff + "_out.npy", output)
    return output


if __name__ == "__main__":
    main(100, 30, "kl")
