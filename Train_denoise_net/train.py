# import matplotlib
# matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
import torch.optim as optim
from tqdm import tqdm
import glob

from net import *
from mini_batch_loader import MiniBatchLoader
from config import config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRAINING_DATA_PATH = "train.txt"
TESTING_DATA_PATH = "train.txt"
VAL_DATA_PATH = "train.txt"
IMAGE_DIR_PATH = "..//"

def main():

    model = Net().to(device)
    # model.load_state_dict(torch.load("../GaussianFilterModel/GaussianModela20000_.pth"))
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    i_index = 0

    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        VAL_DATA_PATH,
        IMAGE_DIR_PATH,
        config.img_size)

    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)

    val_data_size = MiniBatchLoader.count_paths(VAL_DATA_PATH)
    indices_val = np.random.permutation(val_data_size)
    r_val = indices_val[0: 12]
    raw_val = mini_batch_loader.load_training_data(r_val)

    val_PSNR = []

    for n_epi in tqdm(range(0, 9000000), ncols=70, initial=0):

        r = indices[i_index: i_index + config.BATCH_SIZE]
        raw_x = mini_batch_loader.load_training_data(r)

        label = copy.deepcopy(raw_x)
        raw_n = np.random.normal(0, config.sigma, label.shape).astype(label.dtype) / 255.
        ins_noisy = np.clip(label + raw_n, a_min=0., a_max=1.)
        ins_noisy = torch.Tensor(ins_noisy).float().to(device)
        label = torch.Tensor(label).float().to(device)

        optimizer.zero_grad()
        denoised = model(ins_noisy)
        loss = F.mse_loss(denoised, label)
        loss.backward()
        optimizer.step()

        # if n_epi % 100 == 0:
        #     print("loss: ", loss.item())
        #     plt.figure()
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(raw_x[0, 0, :, :])
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(ins_noisy[0, 0, :, :].cpu().detach().numpy())
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(denoised[0, 0, :, :].cpu().detach().numpy())
        #     plt.savefig("result/{}.png".format(n_epi))
        #     plt.close()
        #
        # if n_epi % 1000 == 0:
        #     torch.save(model.state_dict(), "model/{}.pth".