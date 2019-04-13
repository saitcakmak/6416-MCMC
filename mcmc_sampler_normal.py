import numpy as np
from scipy.stats import norm
import datetime


def sample_from_true(n):
    """
    sample the data set of size n from the true distribution
    this one is for normal
    """
    global data
    data = np.random.normal(parameters[0], parameters[1], n)
    post_mean = 1 / (1/prior[1]**2 + n/parameters[1]**2) * (prior[0]/prior[1]**2 + np.sum(data)/parameters[1]**2)
    post_variance = 1 / (1/prior[1]**2 + n/parameters[1]**2)
    return post_mean, post_variance


def likelihood_ratio(candidate, current):
    """
    calculate the likelihood of a given dataset for the given theta
    we should avoid constantly sending the dataset here
    """
    ratio_list = []
    prior_ratio = norm.pdf(candidate, prior[0], prior[1]) / norm.pdf(current, prior[0], prior[1])
    ratio_list.append(prior_ratio)
    for val in data:
        p_cand = norm.pdf(val, candidate, parameters[1])
        p_curr = norm.pdf(val, current, parameters[1])
        ratio_list.append(p_cand/p_curr)
    prob = np.prod(ratio_list)
    return np.nan_to_num(prob)


def generate_candidate(theta):
    """
    generate the next candidate from normal distribution
    """
    return np.random.normal(theta, candidate_cov)


def theta_next(theta):
    """
    return the next theta according to the accept/reject decision
    """
    candidate = generate_candidate(theta)
    accept_prob = likelihood_ratio(candidate, theta)
    uniform = np.random.random_sample()
    if uniform < accept_prob:
        return candidate
    else:
        return theta


def mcmc_run(run_length, theta, string):
    """
    run the mcmc algorithm to do the sampling for a given length with given starting value
    """
    output = []
    for j in range(int(run_length/10000)):
        print("replication '0000s:", j, " time: ", datetime.datetime.now() - start)
        inner_out = []
        for i in range(10000):
            theta = theta_next(theta)
            inner_out.append(theta)
        output = output + inner_out
        np.save("experiments/mcmc_norm_out_" + string + ".npy", output)


def get_input():
    """
    ask user for the input
    """
    global prior, parameters
    n = int(input("enter input size: "))
    # string_parameters = input("enter the parameters of the normal distribution (mean, std dev), comma separated: ")
    # parameters = [int(s) for s in string_parameters.split(',')]
    parameters = [0, 1]
    # string_prior = input("enter the parameters of the normal prior of mean (mean, std dev), comma separated: ")
    # prior = [int(s) for s in string_prior.split(',')]
    prior = [0, 1]
    string = input("enter output string: ")
    return n, string


def main(n, string):
    global start, candidate_cov
    global prior, parameters
    parameters = [0, 1]
    prior = [0, 1]

    candidate_cov = 0.25
    start = datetime.datetime.now()
    # n, string = get_input()
    length = 1000000
    t_start = 1
    post_mean, post_variance = sample_from_true(n)
    params = {"len": length, "t_start": t_start, "n": n, "data": data,
              "covariance": candidate_cov, "parameters": parameters,
              "post_mean": post_mean, "post_variance": post_variance}
    np.save("experiments/mcmc_norm_params_" + string + ".npy", params)
    mcmc_run(length, t_start, string)
    end = datetime.datetime.now()
    print("done! time: ", end - start)


if __name__ == "__main__":
    n, string = get_input()
    main(n, string)
