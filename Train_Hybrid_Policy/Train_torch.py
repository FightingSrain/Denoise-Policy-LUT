# import matplotlib
# matplotlib.use("Agg")
import math
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
import torch.optim as optim
from tqdm import tqdm
import glob
import State as State
# import State_Bilateral as State
# import State_Gaussian as State
# from FCN import *
from FCN_sm_5 import *
from mini_batch_loader import MiniBatchLoader
from pixelwise_a3c import *
from config import config
from Train_Hybrid_Policy.utils import *
from genMask import masksamplingv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRAINING_DATA_PATH = "train.txt"
TESTING_DATA_PATH = "train.txt"
VAL_DATA_PATH = "val.txt"
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
        config.corp_size)

    current_state = State.State((config.BATCH_SIZE, 3, config.img_size, config.img_size), config.MOVE_RANGE)
    # current_state_ori = State.State((config.BATCH_SIZE, 3, config.img_size, config.img_size), config.MOVE_RANGE)
    agent = PixelWiseA3C_InnerState(model, optimizer, config.BATCH_SIZE, config.EPISODE_LEN, config.GAMMA)

    # train dataset
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)

    # val dataset
    val_data_size = MiniBatchLoader.count_paths(VAL_DATA_PATH)
    indices_val = np.random.permutation(val_data_size)
    r_val = indices_val[0: 12]
    raw_val = mini_batch_loader.load_val_data(r_val)

    ValData = []
    pre_pnsr = -math.inf
    for n_epi in tqdm(range(0, 100000), ncols=70, initial=0):

        r = indices[i_index: i_index + config.BATCH_SIZE]
        raw_x = mini_batch_loader.load_training_data(r)

        label = copy.deepcopy(raw_x)
        raw_n = np.random.normal(0, config.sigma, label.shape).astype(label.dtype) / 255.
        ori_ins_noisy = np.clip(label + raw_n, a_min=0., a_max=1.)
        # current_state_ori.reset(ori_ins_noisy.copy())

        if n_epi % 10 == 0:
            image = np.asanyarray(label[10].transpose(1, 2, 0) * 255, dtype=np.uint8)
            image = np.squeeze(image)
            cv2.imshow("label", image)
            cv2.waitKey(1)

        # 生成自监督图像对
        # patten = torch.randint(0, 4, (1,))
        # gen_mask = masksamplingv2()
        # ins_noisy, label = gen_mask(torch.Tensor(ori_ins_noisy), patten.item())


        current_state.reset(ori_ins_noisy.copy())
        reward = np.zeros((config.BATCH_SIZE*3, 1, config.img_size, config.img_size))
        sum_reward = 0

        # raw_ns = np.random.normal(0, config.sigma, label.shape).astype(label.dtype) / 255.
        # label = np.clip(label + raw_ns, a_min=0., a_max=1.)
        # label = labels.copy()

        for t in range(config.EPISODE_LEN):

            if n_epi % 10 == 0:
                #     # cv2.imwrite('./test_img/'+'ori%2d' % (t+c)+'.jpg', current_state.image[20].transpose(1, 2, 0) * 255)
                image = np.asanyarray(current_state.image[10].transpose(1, 2, 0) * 255, dtype=np.uint8)
                image = np.squeeze(image)
                cv2.imshow("temp", image)
                cv2.waitKey(1)

            previous_image = np.clip(current_state.image.copy(), a_min=0., a_max=1.)
            action, action_par, action_prob, tst_act = agent.act_and_train(current_state.tensor, reward)
            current_state.step(action, action_par)
            #-------------------
            # previous_image_ori = np.clip(current_state_ori.image.copy(), a_min=0., a_max=1.)
            # action_ori, action_par_ori = agent.get_action(current_state_ori.tensor)
            # current_state_ori.step(action_ori, action_par_ori)
            # pre_ori1, pre_ori2 = gen_mask(torch.Tensor(previous_image_ori), patten.item())
            # cur_ori1, cur_ori2 = gen_mask(torch.Tensor(current_state_ori.image), torch.randint(0, 4, (1,)).item())

            if n_epi % 150 == 0:
                # print(action[10])
                # # print(action_par[10])
                # print([current_state.hybrid_act(3, 10, action_par),
                #        current_state.hybrid_act(4, 10, action_par),
                #        current_state.hybrid_act(5, 10, action_par),
                #        current_state.hybrid_act(6, 10, action_par)])
                # print(action_prob[10])
                paint_amap(tst_act[10])
                # paint_amap(action[10])
                # paint_scatter(tst_act[10], current_state.image[10])

            # reward = (np.square(label - previous_image) + 8 * np.square(previous_image - label - (pre_ori1 - pre_ori2))) - \
            #          (np.square(label - current_state.image) + 8 * np.square(current_state.image - label - (cur_ori1 - cur_ori2)))
            # reward *= 255
            reward = np.square(label - previous_image) * 255 - \
                     np.square(label - current_state.image) * 255
            reward = reward.reshape(reward.shape[0]*3, 1, reward.shape[2], reward.shape[3])
            # reward = -np.square(current_state.image - label) * 255
            sum_reward += np.mean(reward) * np.power(config.GAMMA, t)
        print(reward.shape)
        print(current_state.tensor.shape)
        print("------------")
        agent.stop_episode_and_train(current_state.tensor, reward, True)

        torch.cuda.empty_cache()

        if n_epi % 100 == 0 and n_epi != 0:
            temp_psnr, temp_ssim = agent.val(agent, State, raw_val, config.EPISODE_LEN)
            if temp_psnr > pre_pnsr:
                pre_pnsr = temp_psnr
                for f in os.listdir("./GaussianFilterHybridMax_{}/".format(config.sigma)):
                    if os.path.splitext(f)[1] == ".pth":
                        os.remove("./GaussianFilterHybridMax_{}/{}".format(config.sigma, f))
                torch.save(model.state_dict(),
                       "./GaussianFilterHybridMax_{}/{}_{}_{}.pth".
                           format(config.sigma, n_epi, temp_psnr, temp_ssim))
                print("save model")
            ValData.append([n_epi, temp_psnr, temp_ssim])
            # savevaltocsv(ValData, "val.csv", config.sigma)  # 保存验证集数据
            patin_val(np.asarray(ValData)[:, 1])

        # if n_epi % 1000 == 0:
        #     # torch.save(model.state_dict(), "../GaussianFilterModel/GaussianModela{}_.pth".format(n_epi))
        #     torch.save(model.state_dict(), "../MixFilterModel/MixModela{}_.pth".format(n_epi))

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
    plt.imshow(image, vmin=1, vmax=9)
    plt.colorbar()
    plt.pause(1)
    # plt.show()
    plt.close('all')

def patin_val(val_PSNR):
    plt.plot(val_PSNR)
    plt.grid()
    # plt.show()
    plt.pause(2)
    plt.close('all')

def paint_scatter(act, img):
    act = np.asanyarray(act.squeeze(), dtype=np.uint8)
    img = np.asanyarray(img.squeeze()*255, dtype=np.uint8)
    act = act.reshape(63*63)
    color = ['red', 'green', 'blue', 'yellow', 'black', 'purple', 'orange', 'pink', 'gray']
    img = img.reshape(63*63)
    for i in range(9):
        plt.scatter(img[act==i],
                    img[act==i], c=color[i], marker='.')
        # plt.bar(img[act == i],
        #         img[act == i], color=color[i])
    # plt.pause(1)
    # plt.close('all')
    plt.show()

if __name__ == '__main__':
    main()
