import matplotlib.pyplot as plt
import numpy as np
import seaborn


def read_tab_seperated_file(file_name, skip="Epoch"):
    with open(file_name, "r") as f:
        lines = f.readlines()
    header = None
    data = list()
    for line in lines:
        if line.startswith(skip):
            header = line.strip().split("\t")
        else:
            data.append([float(x) for x in line.strip().split("\t")])
    return header, np.asarray(data)


def main():
    seaborn.set_palette("muted")
    seaborn.set_style("whitegrid")
    my_results = read_tab_seperated_file("results/my_model.txt")
    monai_results = read_tab_seperated_file("results/monai_model.txt")

    # plot mine
    plt.plot(my_results[1][:, 0], my_results[1][:, 1], label="Training", color="C0", alpha=0.5)
    plt.plot(my_results[1][:, 0], my_results[1][:, 2], label="Validation", color="C1", alpha=0.5)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("results/my_model_training_results.png")

    # plot monai
    plt.clf()
    plt.plot(monai_results[1][:, 0], monai_results[1][:, 1], label="Training", color="C2")
    plt.plot(monai_results[1][:, 0], monai_results[1][:, 2], label="Validation", color="C3")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("results/monai_model_training_results.png")

    # plot both mine and monai training and validation losses with mine transparent
    plt.clf()
    plt.plot(my_results[1][:, 0], my_results[1][:, 1], label="Training (Mine)", color="C0", alpha=0.5)
    plt.plot(my_results[1][:, 0], my_results[1][:, 2], label="Validation (Mine)", color="C1", alpha=0.5)
    plt.plot(monai_results[1][:, 0], monai_results[1][:, 1], label="Training (MONAI)", color="C2")
    plt.plot(monai_results[1][:, 0], monai_results[1][:, 2], label="Validation (MONAI)", color="C3")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("results/training_results_mine_transparent.png")

    # plot both mine and monai training and validation losses with monai transparent
    plt.clf()
    plt.plot(my_results[1][:, 0], my_results[1][:, 1], label="Training (Mine)", color="C0")
    plt.plot(my_results[1][:, 0], my_results[1][:, 2], label="Validation (Mine)", color="C1")
    plt.plot(monai_results[1][:, 0], monai_results[1][:, 1], label="Training (MONAI)", color="C2", alpha=0.5)
    plt.plot(monai_results[1][:, 0], monai_results[1][:, 2], label="Validation (MONAI)", color="C3", alpha=0.5)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("results/training_results_monai_transparent.png")

    # plot both mine and monai training and validation losses with none transparent
    plt.clf()
    plt.plot(my_results[1][:, 0], my_results[1][:, 1], label="Training (Mine)", color="C0")
    plt.plot(my_results[1][:, 0], my_results[1][:, 2], label="Validation (Mine)", color="C1")
    plt.plot(monai_results[1][:, 0], monai_results[1][:, 1], label="Training (MONAI)", color="C2")
    plt.plot(monai_results[1][:, 0], monai_results[1][:, 2], label="Validation (MONAI)", color="C3")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("results/training_results_none_transparent.png")

    # Compare epoch times
    plt.clf()
    plt.boxplot([my_results[1][:, 3], monai_results[1][:, 3]], labels=["Mine", "MONAI"])
    plt.ylabel("Epoch Time (s)")
    plt.savefig("results/epoch_times.png")


if __name__ == "__main__":
    main()
