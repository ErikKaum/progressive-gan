# printify specs
# Print file requirements
# JPG and PNG file types supported
# Maximum file size 50 MB
# Recommended size 1140?~@~J?~W?~@~J1982 px
# Maximum resolution 23000 x 23000 px

# Recommended size 1140 x 1982 px
# Halved: 571 x 991 px
# Halved: 285.5 x 495.5 px
# Halved: 142.75 x 247.75 px
# 
# Therefore, what we want is:
# 143 x 248 px
# 286 x 496 px
# 572 x 992 px
# 1144 x 1984 px, which is pretty close to the real deal


import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 1024
DATASET = './pictures'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 8, 8, 4, 4, 4]
CHANNELS_IMG = 3
Z_DIM = 512  # should be 512 in original paper
IN_CHANNELS = 512  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4
