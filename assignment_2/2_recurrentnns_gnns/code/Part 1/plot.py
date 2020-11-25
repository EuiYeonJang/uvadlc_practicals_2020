import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def plot_figure(model_type, input_length):
    DATA_DIR = "./summaries/"
    seed = [0, 4, 17]

    data = list()
    for s in seed:
        with open(f"{DATA_DIR}{model_type}_seed_{s}_seq_{input_length*4}.pkl", "rb") as f:
            acc_data = pkl.load(f)
            
            data.append(acc_data["acc_list"])

    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)


    x = np.linspace(0, 3000, 3000)
    plt.plot(x, m, color="tab:blue")
    plt.fill_between(x, m-s, m+s, color="tab:green", alpha=0.5)
    plt.title(f"{model_type} with Sequence Length {input_length}")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim((None,1))
    plt.show()
    


if __name__ == "__main__":
    for i in [4, 5, 6]:
        plot_figure("GRU", i)