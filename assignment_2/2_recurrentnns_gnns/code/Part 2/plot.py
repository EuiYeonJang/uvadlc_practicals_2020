import pickle as pkl
import matplotlib.pyplot as plt
import torch
# import io


def plot_figure(acc_list, loss_list):
    DATA_DIR = "./summaries/"

    fig, ax = plt.subplots()
    ax.plot(acc_list, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Accuracy",color="tab:blue")

    ax2=ax.twinx()
    ax2.plot(loss_list, color="tab:orange", alpha=0.8)
    ax2.set_ylabel("Loss",color="tab:orange")
    fig.savefig(f'{DATA_DIR}LSTM.pdf')
    
    plt.show()


def print_sent(greedy, temper):
    DATA_DIR = "./summaries/"
    T = 30
    less_than = int(T/2)
    more_than = 2*T
    sent = [less_than, T, more_than]
    temp = [0.5, 1.0, 2.0]

    print("Greedy")
    for i, gn in enumerate(greedy):
        print(f"\nSAMPLE {i+1}")
        for t in sent:
            print(f"\tT={t} ::\t", gn[t])
    
    print("Tempered")
    for i, tn in enumerate(temper):
        print(f"\nSAMPLE {i}")
        for tao in temp:
            print(f"\ttemp={tao} ::\t", tn[tao])
        

    return


def get_data():
    DATA_DIR = "./summaries/"

    with open(f"{DATA_DIR}data.pkl", "rb") as f:
        d = pkl.load(f)

    acc_list = d["acc"]
    loss_list = d["loss"]

    greedy = d["greedy_sent"]
    temper = d["temperature_sent"]

    plot_figure(acc_list, loss_list)

    print_sent(greedy, temper)

if __name__ == "__main__":
    get_data()
    
