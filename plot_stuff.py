import matplotlib.pyplot as plt
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-training", type=str, default="ensemble_training_results.csv")
    parser.add_argument("--ensemble-evaluation", type=str, default="ensemble_evaluation_results.csv")
    parser.add_argument("--monolith-training", type=str, default="monolith_training_results.csv")
    args = parser.parse_args()

    ensemble_training = pd.read_csv(args.ensemble_training)
    ensemble_evaluation = pd.read_csv(args.ensemble_evaluation)
    monolith_training = pd.read_csv(args.monolith_training)

    figure = plt.figure()
    individual_axis = figure.add_subplot(111)
    monolith_axis = individual_axis.secondary_xaxis(-0.15, functions=(lambda x: x / 5, lambda x: x * 5))

    for chunk_number in range(5):
        for model_number in range(5):
            is_valid = ensemble_training["valid"] == True
            individual_axis.plot(
                ensemble_training.loc[
                    (ensemble_training["chunk"] == chunk_number)
                    & (ensemble_training["model"] == model_number)
                    & is_valid
                ].index,
                ensemble_training.loc[
                    (ensemble_training["chunk"] == chunk_number)
                    & (ensemble_training["model"] == model_number)
                    & is_valid,
                    "accuracy",
                ],
                color="tab:blue",
            )
            individual_axis.plot(
                ensemble_training.loc[
                    (ensemble_training["chunk"] == chunk_number)
                    & (ensemble_training["model"] == model_number)
                    & ~is_valid
                ].index,
                ensemble_training.loc[
                    (ensemble_training["chunk"] == chunk_number)
                    & (ensemble_training["model"] == model_number)
                    & ~is_valid,
                    "accuracy",
                ],
                color="tab:red",
            )

    individual_axis.scatter(
        (ensemble_evaluation["chunk"] + 1) * 250, ensemble_evaluation["accuracy"], color="tab:purple", s=60
    )
    individual_axis.plot(range(0, len(monolith_training) * 5, 5), monolith_training["accuracy"], color="tab:orange")
    vertical_lines = [250 * i for i in range(5)]
    for line in vertical_lines:
        individual_axis.axvline(line, color="black", alpha=0.5, linestyle="--")
    individual_axis.set_title("Validation Accuracy")
    individual_axis.set_xlabel("individual epochs")
    individual_axis.xaxis.label.set_color("tab:blue")
    individual_axis.set_ylabel("accuracy")
    monolith_axis.set_xlabel("monolith epochs")
    monolith_axis.xaxis.label.set_color("tab:orange")

    # Legend with valid in tab:blue, invalid in tab:red, ensemble in tab:purple, and monolith in tab:orange.
    individual_axis.legend(
        [
            plt.Line2D([0], [0], color="tab:red", lw=2),
            plt.Line2D([0], [0], color="tab:blue", lw=2),
            plt.Line2D([0], [0], marker="o", color="white", markerfacecolor="tab:purple", markersize=10),
            plt.Line2D([0], [0], color="tab:orange", lw=2),
        ],
        ["new training", "continued training", "ensemble", "monolith"],
    )

    plt.tight_layout()
    plt.savefig("ensemble_training_accuracy.png", dpi=600)

    plt.clf()
    figure = plt.figure()
    individual_axis = figure.add_subplot(111)
    monolith_axis = individual_axis.secondary_xaxis(-0.15, functions=(lambda x: x / 5, lambda x: x * 5))

    for chunk_number in range(5):
        for model_number in range(5):
            is_valid = ensemble_training["valid"] == True
            individual_axis.plot(
                ensemble_training.loc[
                    (ensemble_training["chunk"] == chunk_number)
                    & (ensemble_training["model"] == model_number)
                    & is_valid
                ].index,
                ensemble_training.loc[
                    (ensemble_training["chunk"] == chunk_number)
                    & (ensemble_training["model"] == model_number)
                    & is_valid,
                    "loss",
                ],
                color="tab:blue",
            )
            individual_axis.plot(
                ensemble_training.loc[
                    (ensemble_training["chunk"] == chunk_number)
                    & (ensemble_training["model"] == model_number)
                    & ~is_valid
                ].index,
                ensemble_training.loc[
                    (ensemble_training["chunk"] == chunk_number)
                    & (ensemble_training["model"] == model_number)
                    & ~is_valid,
                    "loss",
                ],
                color="tab:red",
            )

    individual_axis.scatter(
        (ensemble_evaluation["chunk"] + 1) * 250, ensemble_evaluation["loss"], color="tab:purple", s=60
    )
    individual_axis.plot(range(0, len(monolith_training) * 5, 5), monolith_training["loss"], color="tab:orange")
    vertical_lines = [250 * i for i in range(5)]
    for line in vertical_lines:
        individual_axis.axvline(line, color="black", alpha=0.5, linestyle="--")
    individual_axis.set_title("Validation Loss")
    individual_axis.set_xlabel("individual epochs")
    individual_axis.xaxis.label.set_color("tab:blue")
    individual_axis.set_ylabel("loss")
    monolith_axis.set_xlabel("monolith epochs")
    monolith_axis.xaxis.label.set_color("tab:orange")

    individual_axis.legend(
        [
            plt.Line2D([0], [0], color="tab:red", lw=2),
            plt.Line2D([0], [0], color="tab:blue", lw=2),
            plt.Line2D([0], [0], marker="o", color="white", markerfacecolor="tab:purple", markersize=10),
            plt.Line2D([0], [0], color="tab:orange", lw=2),
        ],
        ["new training", "continued training", "ensemble", "monolith"],
    )

    plt.tight_layout()
    plt.savefig("ensemble_training_loss.png", dpi=600)


if __name__ == "__main__":
    main()
