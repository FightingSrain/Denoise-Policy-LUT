import time

class config:

    MOVE_RANGE = 3
    EPISODE_LEN = 8
    MAX_EPISODE = 100000
    GAMMA = 0.95
    N_ACTIONS = 9
    BATCH_SIZE = 32
    LR = 0.0003
    img_size = 63
    sigma = 15
    num_episodes = 1e8

    # --LUT--
    SAMPLING_INTERVAL = 4  # N bit uniform sampling
    SIGMA = 5  # Gaussian noise std


    # --plicy LUTs--

