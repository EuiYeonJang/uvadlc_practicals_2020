import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def plot_figure(input_length):
    DATA_DIR = "./summaries/"
    seed = [0, 4, 17]

    min_step = 3000
    data = list()
    for s in seed:
        with open(f"{DATA_DIR}seed_{s}_seq_{input_length}.pkl", "rb") as f:
            acc = pkl.load(f)
            
            if len(acc) < min_step: 
                min_step = len(acc)
                data.append(acc)
            else:
                data.append(acc[:min_step])

    
    m = np.mean(data, axis=1)
    s = np.std(data, axis=1)


    x = np.linspace(0, min_step, min_step)
    plt.plot(x, m, 'k-')
    plt.fill_between(x, m-s, m+s)
    plt.show()
    


if __name__ == "__main__":
    pass