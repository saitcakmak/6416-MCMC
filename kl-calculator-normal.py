import numpy as np
from scipy.stats import norm


def get_input():
    data = np.load("")
    parameters = np.load("")
    mean, std_dev = parameters["post_mean"], np.sqrt(parameters["post_variance"])
    return data, mean, std_dev


def create_hist(data):
    interval = [-5, 5]
    bin_count = 10000
    hist, bins = np.histogram(data, bin_count, interval, density=True)
    return hist, bins


def calc_div(hist, bins, mean, std_dev):
    kl_div = 0
    for i in range(len(bins)-1):
        true_prob = norm.pdf(bins[i+1], mean, std_dev) - norm.pdf(bins[i], mean, std_dev)
        estimate = hist[i]
        if estimate > 0:
            kl_div += estimate * np.log(true_prob/estimate)
    return kl_div


def main():
    """do the job"""


if __name__ == "__main__":
    main()
