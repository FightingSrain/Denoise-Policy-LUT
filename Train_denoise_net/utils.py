import pandas as pd
import os

def savevaltocsv(records, filename, sigma):
    col_name = ['n_epi', 'val_psnr', 'val_ssim']
    # records = [['001', '小明', 18]]
    # 先转为DataFrame格式
    df = pd.DataFrame(columns=col_name, data=records)
    # index=False表示存储csv时没有默认的id索引
    # 如果文件不存在，则创建文件，如果文件存在，则追加内容
    df.to_csv("./Val_PSNR_and_SSIM_{}/{}".format(sigma, filename),
              encoding='utf-8', index=False)



# 定义一个函数，接受文件夹路径和文件名作为参数
def save_file(folder_path, file_name):
  # 拼接文件路径
  file_path = os.path.join(folder_path, file_name)
  # 判断文件是否存在
  if os.path.exists(file_path):
    # 如果存在，删除文件
    os.remove(file_path)