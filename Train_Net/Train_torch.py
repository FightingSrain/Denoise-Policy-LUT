# import matplotlib
# matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
import torch.optim as optim
from tqdm import tqdm
import glob
# import State as State
# import State_Bilateral as State
import State_Gaussian as State
# from FCN import *
from FCN_sm_4 import *
from mini_batch_loader import MiniBatchLoader
from pixelwise_a3c import *
from config import config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRAINING_DATA_PATH = "train.txt"
TESTING_DATA_PATH = "train.txt"
VAL_DATA_PATH = "train.txt"
IMAGE_DIR_PATH = "..//"

def main():
    model = PPO(config.N_ACTIONS).to(device)
    # model.load_state_dict(torch.load("../GaussianFilterModel/GaussianModela20000_.pth"))
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    i_index = 0

    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        VAL_DATA_PATH,
        IMAGE_DIR_PATH,
        config.img_size)

    current_state = State.State((config.BATCH_SIZE, 1, 63, 63), config.MOVE_RANGE)
    agent = PixelWiseA3C_InnerState(model, optimizer, config.BATCH_SIZE, config.EPISODE_LEN, config.GAMMA)

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
        current_state.reset(raw_x, raw_n)
        reward = np.zeros(label.shape, label.dtype)
        sum_reward = 0

        if n_epi % 10 == 0:
            image = np.asanyarray(label[10].transpose(1, 2, 0) * 255, dtype=np.uint8)
            image = np.squeeze(image)
            cv2.imshow("label", image)
            cv2.waitKey(1)

        for t in range(config.EPISODE_LEN):

            if n_epi % 10 == 0:
                #     # cv2.imwrite('./test_img/'+'ori%2d' % (t+c)+'.jpg', current_state.image[20].transpose(1, 2, 0) * 255)
                image = np.asanyarray(current_state.image[10].transpose(1, 2, 0) * 255, dtype=np.uint8)
                image = np.squeeze(image)
                cv2.imshow("temp", image)
                cv2.waitKey(1)

            previous_image = np.clip(current_state.image.copy(), a_min=0., a_max=1.)
            action, inner_state, action_prob, tst_act = agent.act_and_train(current_state.tensor, reward)

            if n_epi % 150 == 0:
                print(action[10])
                print(action_prob[10])
                paint_amap(tst_act[10])
                # paint_amap(action[10])
                # paint_scatter(tst_act[10], current_state.image[10])

            current_state.step(action, inner_state)
            # 是否可以自监督训练，即不需要label
            reward = np.square(label - previous_image) * 255 - \
                     np.square(label - current_state.image) * 255
            # reward = -np.square(current_state.image - label) * 255
            sum_reward += np.mean(reward) * np.power(config.GAMMA, t)

        agent.stop_episode_and_train(current_state.tensor, reward, True)

        torch.cuda.empty_cache()

        if n_epi % 100 == 0:
            temp_psnr = agent.val(agent, State, raw_val, config.EPISODE_LEN)
            val_PSNR.append(temp_psnr)
            patin_val(val_PSNR)


        if n_epi % 1000 == 0:
            torch.save(model.state_dict(), "../GaussianFilterModel/GaussianModela{}_.pth".format(n_epi))

        if i_index + config.BATCH_SIZE >= train_data_size:
            i_index = 0
            indices = np.random.permutation(train_data_size)
        else:
            i_index += config.BATCH_SIZE

        if i_index + 2 * config.BATCH_SIZE >= train_data_size:
            i_index = train_data_size - config.BATCH_SIZE

        print("train total reward {a}".format(a=sum_reward * 255))


def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    # print(image)
    plt.imshow(image, vmin=1, vmax=9)
    plt.colorbar()
    plt.pause(1)
    # plt.show()
    plt.close('all')

def patin_val(val_PSNR):
    plt.plot(val_PSNR)
    # plt.show()
    plt.pause(1)
    plt.close('all')

def paint_scatter(act, img):
    act = np.asanyarray(act.squeeze(), dtype=np.uint8)
    img = np.asanyarray(img.squeeze()*255, dtype=np.uint8)
    act = act.reshape(63*63)
    color = ['red', 'green', 'blue', 'yellow', 'black', 'purple', 'orange', 'pink', 'gray']
    img = img.reshape(63*63)
    for i in range(9):
        # plt.scatter(img[act==i],
        #             img[act==i], c=color[i], marker='.')
        plt.bar(img[act == i],
                img[act == i], color=color[i])
    # plt.pause(1)
    # plt.close('all')
    plt.show()

if __name__ == '__main__':
    main()
