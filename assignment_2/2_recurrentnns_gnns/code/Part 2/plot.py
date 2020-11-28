import pickle as pkl
import torch
import io

DATA_DIR = "./summaries/"

# data = torch.load(f"{DATA_DIR}da/ta.pkl", map_location=torch.device("cpu"))
torch.load(f"{DATA_DIR}data.pkl", map_location=torch.device('cpu'))

# print(data.keys())
# gen_sent