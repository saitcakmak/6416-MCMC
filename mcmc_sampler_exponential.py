import numpy as np
import datetime


def sample_from_true(n):
    """
    sample the data set of size n from the true distribution
    start with exponential
    """
    global data
    data = np.random.exponential(1/parameters[0], n)
    post_alpha = prior[0] + n
    post_beta = prior[1] + np.sum(data)
    return post_alpha, post_beta


def likelihood_ratio(candidate, current):
    """
    calculate the likelihood of a given dataset for the given theta
    we should avoid constantly sending the dataset here
    """
    ratio_list = []
    prior_ratio = (candidate / current) ** (prior[0] - 1) * np.exp(- prior[1] * (candidate - current))
    ratio_list.append(prior_ratio)
    for val in data:
        p_cand = np.exp(- candidate * (val - delta)) - np.exp(- candidate * val)
        p_curr = np.exp(- current * (val - delta)) - np.exp(- current * val)
        ratio_list.append(p_cand/p_curr)
    prob = np.prod(ratio_list)
    return np.nan_to_num(prob)


def generate_candidate(theta, cov):
    """
    generate the next candidate from normal distribution
    """
    cand = np.random.normal(theta, cov)
    if not cand > 0:
        cand = generate_candidate(theta, cov)
    return cand


def theta_next(theta, cov):
    """
    return the next theta according to the accept/reject decision
    """
    candidate = generate_candidate(theta, cov)
    accept_prob = likelihood_ratio(candidate, theta)
    uniform = np.random.random_sample()
    if uniform < accept_prob:
        return candidate
    else:
        return theta


def mcmc_run(run_length, theta, string, cov):
    """
    run the mcmc algorithm to do the sampling for a given length with given starting value
    """
    output = []
    for j in range(int(run_length/10000)):
        print("replication '0000s:", j, " time: ", datetime.datetime.now() - start)
        inner_out = []
        for i in range(10000):
            theta = theta_next(theta, cov)
            inner_out.append(theta)
        output = output + inner_out
        np.save("experiments/mcmc_exp_out_" + string + ".npy", output)


def get_input():
    """
    ask user for the input
    """
    global prior, parameters
    n = int(input("enter input size: "))
    # string_parameters = input("enter the true rate of the exponential distribution: ")
    # parameters = [int(s) for s in string_parameters.split(',')]
    parameters = [1]
    # string_prior = input("enter the parameters of the prior gamma (shape, rate - integers), comma separated: ")
    # prior = [int(s) for s in string_prior.split(',')]
    prior = [1, 1]
    string = input("enter output string: ")
    return n, string


def main(n, string, cov):
    global start, delta
    global prior, parameters
    parameters = [1]
    prior = [1, 1]

    # candidate_cov = 0.25
    delta = 0.000001
    start = datetime.datetime.now()
    # n, string = get_input()
    length = 1000000
    t_start = 1
    post_alpha, post_beta = sample_from_true(n)
    params = {"len": length, "t_start": t_start, "n": n, "data": data,
              "covariance": cov, "delta": delta, "parameters": parameters,
              "post_alpha": post_alpha, "post_beta": post_beta}
    np.save("experiments/mcmc_exp_params_" + string + ".npy", params)
    mcmc_run(length, t_start, string, cov)
    end = datetime.datetime.now()
    print("done! time: ", end - start)


if __name__ == "__main__":
    n, string = get_input()
    main(n, string, 0.25)
