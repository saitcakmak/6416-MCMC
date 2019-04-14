import numpy as np
import csv

file_list = ["exp_10_30_0.01_wass_out.npy", "exp_10_30_0.0001_wass_out.npy", "exp_10_30_0.04_wass_out.npy",
             "exp_10_30_0.09_wass_out.npy", "exp_10_30_0.25_wass_out.npy", "exp_10_30_0.0025_wass_out.npy",
             "exp_10_30_0.64_wass_out.npy", "exp_10_30_1.0_wass_out.npy", "exp_10_30_4.0_wass_out.npy",
             "exp_10_30_25.0_wass_out.npy", "norm_10_30_wass_out.npy", "norm_100_30_wass_out.npy"]

header = ["burn\\len", 100, 500, 1000, 2000, 5000, 10000, 50000, 100000, 200000, 400000, 800000]

for file in file_list:
    with open(file[:-4]+'.csv', "w") as writer_file:
        writer = csv.writer(writer_file, delimiter=",")
        data = np.load("wass-dist/"+file).item()
        writer.writerow([file])
        writer.writerow(header)
        for key in data.keys():
            row = [key]
            for key2 in data[key].keys():
                row.append(data[key][key2])
            writer.writerow(row)
        writer.writerow("")
