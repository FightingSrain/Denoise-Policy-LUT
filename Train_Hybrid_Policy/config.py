import time

class config:

    MOVE_RANGE = 3
    EPISODE_LEN = 5
    MAX_EPISODE = 100000
    GAMMA = 0.95
    N_ACTIONS = 9
    BATCH_SIZE = 11
    LR = 0.0001

    sigma = 25
    num_episodes = 1e8

    corp_size = 64
    img_size = 64
    img_tst_size = 64


    # --LUT--
    SAMPLING_INTERVAL = 4  # N bit uniform sampling
    SIGMA = 25  # Gaussian noise std


    # --plicy LUTs--

