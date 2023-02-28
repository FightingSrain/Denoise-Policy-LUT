# import matplotlib
# matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
import torch.optim as optim
from tqdm import tqdm
import glob
from skimage.metrics import structural_similarity as ssim_cal
from skimage.metrics import peak_signal_noise_ratio as psnr_cal

from Train_denoise_net.net3 import *
from Train_denoise_net.mini_batch_loader import MiniBatchLoader
from Train_denoise_net.config import config
from Train_denoise_net.utils import *
from Train_denoise_net.genMask import *
from Train_denoise_net.lab_loss import *
from Train_denoise_net.Myloss import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRAINING_DATA_PATH = "train.txt"
TRAINING_DATA_PATH_GT = "train.txt"
TESTING_DATA_PATH = "train.txt"
# TRAINING_DATA_PATH = "SIDDNOISY.txt"
# TRAINING_DATA_PATH_GT = "SIDDGT.txt"
# TESTING_DATA_PATH = "SIDDNOISY.txt"

VAL_DATA_PATH = "val.txt"
IMAGE_DIR_PATH = "..//"

labloss = TotalLoss()
L_color_rate = L_color_rate()
def paint_val(val_PSNR):
    plt.plot(val_PSNR)
    plt.grid()
    # plt.show()
    plt.pause(2)
    plt.close('all')
def main():

    model = Net().to(device)
    # model.load_state_dict(torch.load("../GaussianFilterModel/GaussianModela20000_.pth"))
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    i_index = 0

    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TRAINING_DATA_PATH_GT,
        TESTING_DATA_PATH,
        VAL_DATA_PATH,
        IMAGE_DIR_PATH,
        config.img_size)

    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)

    val_data_size = MiniBatchLoader.count_paths(VAL_DATA_PATH)
    indices_val = np.random.permutation(val_data_size)

    r_val = indices_val
    raw_val = mini_batch_loader.load_val_data(r_val)
    len_val = len(raw_val)
    ValData = []
    max_psnr = -math.inf
    sigma = 25

    for n_epi in tqdm(range(0, 100000), ncols=70, initial=0):

        r = indices[i_index: i_index + config.BATCH_SIZE]
        raw_x, gt = mini_batch_loader.load_training_data(r)


        label = copy.deepcopy(raw_x)
        raw_n = np.random.normal(0, sigma, label.shape).astype(label.dtype) / 255.
        ins_noisy = np.clip(label + raw_n, a_min=0., a_max=1.)
        ins_noisy = torch.Tensor(ins_noisy).float().cuda()
        label = torch.Tensor(label).float().cuda()

        # label = torch.Tensor(copy.deepcopy(gt)).float().cuda()
        # ins_noisy = torch.Tensor(copy.deepcopy(raw_x)).float().cuda()

        # patten = torch.randint(0, 4, (1,))
        # gen_mask = masksamplingv2()
        # ins_noisy1, label1 = gen_mask(torch.Tensor(ins_noisy), patten.item())



        # denoised1 = model(ins_noisy1)
        # with torch.no_grad():
        #     denoised2 = model(ins_noisy)
        #     a_ins, a_label = gen_mask(denoised2, patten.item())

        denoised = model(ins_noisy)
        optimizer.zero_grad()
        loss1 = F.mse_loss(denoised, label) * 255
        # loss, _, _ = labloss(denoised, label)
        loss2 = torch.mean(L_color_rate(denoised, label)) * 1

        # loss = F.mse_loss(denoised1, label1) + 8 * F.mse_loss(denoised1 - label1, a_ins - a_label)
        # loss, _, _ = labloss(denoised1, label1)
        # loss += (8 * F.mse_loss(denoised1 - label1,
        #                         a_ins - a_label))
        print("\nloss1:", loss1.data,
              "\nloss2:", loss2.data)
        loss = loss1
        loss.backward()
        optimizer.step()
        print("\niter: ", n_epi, "loss: ", loss.data)

        # if n_epi % 100 == 0:
        #     # print("loss: ", loss.item())
        #     plt.figure()
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(raw_x[0, 0, :, :])
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(ins_noisy[0, 0, :, :].cpu().detach().numpy())
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(denoised[0, 0, :, :].cpu().detach().numpy())
        #     # plt.savefig("result/{}.png".format(n_epi))
        #     plt.pause(1)
        #     plt.close()

        if n_epi % 100 == 0:
            image1 = np.asanyarray(np.clip(denoised[0, :, :, :].detach().cpu().numpy(), a_min=0., a_max=1.).transpose(1, 2, 0) * 255, dtype=np.uint8)
            image1 = np.squeeze(image1)
            cv2.imshow("denoise", image1)

            image2 = np.asanyarray(ins_noisy[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0) * 255,
                                   dtype=np.uint8)
            image2 = np.squeeze(image2)
            cv2.imshow("noisy", image2)
            image3 = np.asanyarray(label[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0) * 255,
                                   dtype=np.uint8)
            image3 = np.squeeze(image3)
            cv2.imshow("label", image3)

            # image1 = np.asanyarray(raw_x[0,:,:,:].transpose(1,2,0) * 255, dtype=np.uint8)
            # image1 = np.squeeze(image1)
            # cv2.imshow("label", image1)
            #
            # image2 = np.asanyarray(np.clip(denoised2[0, :, :, :].detach().cpu().numpy().transpose(1,2,0),
            #                                a_min=0., a_max=1.) * 255, dtype=np.uint8)
            # image2 = np.squeeze(image2)
            # cv2.imshow("denoise", image2)
            #
            # image3 = np.asanyarray(ins_noisy[0, :, :, :].detach().cpu().numpy().transpose(1,2,0) * 255,
            #                        dtype=np.uint8)
            # image3 = np.squeeze(image3)
            # cv2.imshow("ins", image3)

            cv2.waitKey(1)

        if n_epi % 100 == 0 and n_epi != 0:
            model.eval()
            val_pnsr = 0
            val_ssim = 0
            raw_n = np.random.normal(0, sigma, raw_val.shape).astype(raw_val.dtype) / 255.
            ins_noisy_val = np.clip(raw_val + raw_n, a_min=0., a_max=1.)
            val_res = model(torch.Tensor(ins_noisy_val).float().to(device))
            val_res = np.clip(val_res.detach().cpu().numpy(), a_min=0., a_max=1.)

            for i in range(0, len_val):
                psnr = psnr_cal(raw_val[i, :, :, :].transpose(1, 2, 0),
                                val_res[i, :, :, :].transpose(1, 2, 0))
                ssim = ssim_cal(raw_val[i, :, :, :].transpose(1, 2, 0),
                                val_res[i, :, :, :].transpose(1, 2, 0), multichannel=True)
                val_pnsr += psnr
                val_ssim += ssim
            val_pnsr /= len_val
            val_ssim /= len_val
            ValData.append([n_epi, val_pnsr, val_ssim])
            savevaltocsv(ValData, "val.csv", sigma) # 保存验证集数据
            # 绘图
            paint_val(np.asarray(ValData)[:, 1])
            model.train()
            # 保存验证集中PSNR最好的模型
            if val_pnsr > max_psnr:
                max_psnr = val_pnsr
                for f in os.listdir("./DenoiseNetModelMax_{}/".format(sigma)):
                    if os.path.splitext(f)[1] == ".pth":
                        os.remove("./DenoiseNetModelMax_{}/{}".format(sigma, f))
                torch.save(model.state_dict(),
                       "./DenoiseNetModelMax_{}/{}_{}_{}.pth".format(sigma, n_epi, val_pnsr, val_ssim))
                print("save model")

        if i_index + config.BATCH_SIZE >= train_data_size:
            i_index = 0
            indices = np.random.permutation(train_data_size)
        else:
            i_index += config.BATCH_SIZE

        if i_index + 2 * config.BATCH_SIZE >= train_data_size:
            i_index = train_data_size - config.BATCH_SIZE


if __name__ == '__main__':
    main()
