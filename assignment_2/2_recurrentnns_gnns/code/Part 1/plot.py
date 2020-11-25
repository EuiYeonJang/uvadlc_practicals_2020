import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def plot_figure(model_type, input_length):
    input_length *= 4
    DATA_DIR = "./summaries/"
    seed = [0, 4, 17]

    data_len = list()
    data = list()
    for s in seed:
        with open(f"{DATA_DIR}{model_type}_seed_{s}_seq_{input_length}.pkl", "rb") as f:
            acc = pkl.load(f)
            
            data.append(acc)
            data_len.append(len(acc))

    min_step = min(data_len)

    data = [i[:min_step] for i in data]
    for i in data:
        print(len(i))
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)


    x = np.linspace(0, min_step, min_step)
    plt.plot(x, m, color="tab:blue")
    plt.fill_between(x, m-s, m+s, color="tab:green")
    plt.title(f"{model_type} with Sequence Length {input_length}")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Training Steps")
    plt.show()
    


if __name__ == "__main__":
    for i in [4, 5, 6]:
        plot_figure("LSTM", i)