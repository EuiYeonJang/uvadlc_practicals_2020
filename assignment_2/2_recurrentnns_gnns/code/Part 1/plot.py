import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

DATA_DIR = "./summaries/"
SEED = [0, 4, 17]
INPUT_LEN = [4, 5, 6]
COLOUR_MAP = {
    4: "tab:blue",
    5: "tab:green",
    6: "tab:red"
}

def summary(model_type):
    summ_data = list()
    stat_data = list()

    for input_length in INPUT_LEN:
        print(f":: input {input_length} ::")

        for s in SEED:
            with open(f"{DATA_DIR}{model_type}_seed_{s}_seq_{input_length}.pkl", "rb") as f:
                orig_data = pkl.load(f)["acc"]
                v = orig_data[-1]

                summ_data.append(len(orig_data))

                stat_data.append(v)

        m_a = np.mean(stat_data)
        s_a = np.std(stat_data)

        m_s = np.mean(summ_data)

        print(f"mean: {m_a:.2f} - std: {s_a:.2f} - steps {m_s:.2f}")

    


def plot_figure(model_type):
    summ_data = list()
    for input_length in INPUT_LEN:
        with open(f"{DATA_DIR}{model_type}_seed_0_seq_{input_length}.pkl", "rb") as f:
            plot_data = pkl.load(f)["acc"]
            plt.plot(plot_data, color=COLOUR_MAP[input_length], label=f"Seq Length {input_length}", alpha=0.8)

    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.legend()
    plt.savefig(f"{DATA_DIR}{model_type}.pdf")
    plt.show()
    


if __name__ == "__main__":
    print(":: LSTM ::")
    plot_figure("LSTM")
    summary("LSTM")

    print(":: GRU ::")
    plot_figure("GRU")
    summary("GRU")
