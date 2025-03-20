import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from plot_training_results import read_tab_seperated_file


def main():
    seaborn.set_palette("muted")
    seaborn.set_style("whitegrid")

    results = pd.read_csv("results/signal_loss_and_transformation_time.tsv", sep="\t")
    long_results = pd.DataFrame()
    long_results["Time"] = pd.concat([results["monai_time"], results["my_time"]], axis=0)
    long_results["Augmentation Method"] = ["Stepwise"] * len(results) + ["Combined"] * len(results)
    long_results["MSE"] = pd.concat([results["monai_signal_loss"], results["my_signal_loss"]], axis=0)
    long_results["n transforms"] = pd.concat([results["n_params"], results["n_params"]], axis=0)
    seaborn.boxplot(long_results, x="n transforms", y="Time", hue="Augmentation Method")
    plt.ylabel("Time (s)")
    plt.xlabel("Number of transformations")
    plt.savefig("results/boxplot_time.png")

    plt.clf()
    seaborn.boxplot(long_results, x="n transforms", y="MSE", hue="Augmentation Method")
    plt.xlabel("Number of transformations")
    plt.savefig("results/boxplot_mse.png")


if __name__ == "__main__":
    main()
