import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def summary(model_type, input_length):
    DATA_DIR = "./summaries/"
    seed = [0, 4, 17]

    print(f":: input {input_length} ::")
    for s in seed:
        with open(f"{DATA_DIR}{model_type}_seed_{s}_seq_{input_length}.pkl", "rb") as f:
            acc_dict = pkl.load(f)
            acc_list = acc_dict["acc"]
            v = acc_dict["acc"][-1]
            
            print(f"seed {s}: {v}\tsteps: {len(acc_list)}")

def whatever(model_type, input_length):
    DATA_DIR = "./summaries/"
    seed = [0, 4, 17]
    summ_data = list()
    for s in seed:
        with open(f"{DATA_DIR}{model_type}_seed_{s}_seq_{input_length}.pkl", "rb") as f:
            orig_data = pkl.load(f)
            
            summ_data.append(len(orig_data["acc"]))
            
            if s == 0:
                plot_data = orig_data["acc"]

    m = np.mean(summ_data)
    s = np.std(summ_data)


def plot_figure(model_type):
    DATA_DIR = "./summaries/"
    INPUT_LEN = [6, 5, 4]
    COLOUR_MAP = {
        4: "tab:blue",
        5: "tab:green",
        6: "tab:red"
    }

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
    # for i in [4, 5, 6]:
        # summary("LSTM", i)

    print(":: GRU ::")
    plot_figure("GRU")
    # for i in [4, 5, 6]:
        # summary("GRU", i)
