from multiprocessing import Pool
import mcmc_sampler_normal
import mcmc_sampler_exponential

n = int(input("enter data size n: "))
rep = int(input("enter number of replications: "))
# choice = input("enter exp or norm: ")
choice = "exp"
cov = float(input("enter candidate covariance: "))

arg_list = []
for i in range(1, rep+1):
    arg_list.append([n, str(n)+"_"+str(i)+"_"+str(cov), cov])

pool = Pool(rep)
if choice == "exp":
    results = pool.starmap(mcmc_sampler_exponential.main, arg_list)
# elif choice == "norm":
    # results = pool.starmap(mcmc_sampler_normal.main, arg_list)
pool.close()
pool.join()
