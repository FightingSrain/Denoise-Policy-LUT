import os

def readname():
    # filePath = './div2k'
    # filePath = './pristine_images'
    filePath = 'D://Dataset/exploration_database_and_code/pristine_images'
    # filePath = './flower_2000'
    name = os.listdir(filePath)
    return name, filePath

if __name__ == "__main__":
    name, filePath = readname()
    print(name)
    txt = open("waterloo.txt", 'w')
    # txt = open("BSD432_test.txt", 'w')
    for i in name:
        # print(filePath + "/" + i)
        image_dir = os.path.join(filePath + "/", str(i))
        # image_dir = os.path.join('./pristine_images/', str(i))
        # image_dir = os.path.join('./flower_2000', str(i))
        txt.write(image_dir + "\n")
