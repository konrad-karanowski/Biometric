from test import *
from functools import partial
import pandas as pd
import os
import numpy as np
import pandas as pd
import os
import cv2
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel
from data.dataset import retinex_decomposition
from sklearn.metrics import f1_score
from PIL import Image

from logging import getLogger, basicConfig, INFO, FileHandler, StreamHandler





def add_gaussian_noise_with_psnr(image, psnr):
    # Calculate the mean squared error (MSE) from PSNR
    
    # Generate Gaussian noise with zero mean and calculated variance
    noise = np.random.normal(0, psnr, image.shape)
    
    # Add noise to the image
    noisy_image = image + noise
    
    # Clip values to [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image.astype(np.uint8)

def to_lab(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return img

def rescale(L):
    """
    Rescales the L channel of the input LAB image to the range [0, 255].
    """
    # Split LAB image into channels

    # Rescale L channel to the range [0, 255]
    L_rescaled = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
    return L_rescaled

def to_bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img

def rescale_luminance(img, method: str, param = None):
    img = to_lab(img)
    if method == 'square':
        temp_img = img.copy()
        temp_img[..., 0] = rescale((img[..., 0].astype(np.float32) ** 2))
        return to_bgr(temp_img)
    elif method == 'linear':
        temp_img = img.copy()
        temp_img[..., 0] = rescale(img[..., 0].astype(np.float32) * param)
        return to_bgr(temp_img)
    elif method == 'const':
        temp_img = img.copy()
        temp_img[..., 0] = np.clip((img[..., 0].astype(np.float32) + param), 0, 255).astype(np.uint8)
        return to_bgr(temp_img)
    raise NotImplementedError(f'No such method: {method}. Allowed options: [square, linear, const]')


def test_with_perturbation(model, test_list, test_root, test_batch_size, perturbation, retinex):
    identity_list = get_lfw_list(test_list)
    img_paths = [os.path.join(test_root, each) for each in identity_list]

    model.eval()
    acc, score = lfw_test(model, img_paths, identity_list, test_list, test_batch_size, perturbation, retinex=retinex)
    return acc, score




TEST_LISTS = [
    '/home/mzieba/Biometrics/arcface-pytorch/weak_test.txt',
    '/home/mzieba/Biometrics/arcface-pytorch/strong_test.txt'
]
TEST_ROOT = '/home/mzieba/S03-Biometrics-L-Konrad-Karanowski/storage/data/CelebAFRTriplets/CelebAFRTriplets/images/'
TEST_BATCH_SIZE = 10
RETINEX = True


PSNRS = [
0.1,
1,
10,
100,
1000
]

LUMINENCES_LINEAR = [
    1/2, 3/5, 3/4, 4/3, 2/3 
]

LUMINENCES_CONST = [
    -100, -20, -10, 30, -255
]


# MODEL_PATH = '/home/mzieba/Biometrics/arcface-pytorch/checkpoints/resnet18_retinex/best.pth'

# LOG_ROOT = '/home/mzieba/Biometrics/arcface-pytorch/noise/retinex'

# LOG_NAME = 'resnet_18_retinex'

# RETINEX = 'L'

MODEL_PATH = '/home/mzieba/Biometrics/arcface-pytorch/checkpoints/resnet18/resnet18_face.pth'

LOG_ROOT = '/home/mzieba/Biometrics/arcface-pytorch/noise/normal'

LOG_NAME = 'resnet_18'

RETINEX = None


# MODEL_PATH = '/home/mzieba/Biometrics/arcface-pytorch/checkpoints/resnet18_retinex_r/resnet18_40.pth'

# LOG_ROOT = '/home/mzieba/Biometrics/arcface-pytorch/test_results/retinex_r'

# LOG_NAME = 'resnet_18_retinex_r'

# RETINEX = 'R'

os.makedirs(LOG_ROOT, exist_ok=True)

logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        FileHandler(os.path.join(LOG_ROOT, f"{LOG_NAME}.log")),
        StreamHandler()
    ]
)


if __name__ == '__main__':
    # load model
    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))

    for test_file in TEST_LISTS:

        test_name = os.path.basename(test_file).split('.')[0]

        logger.info(f'Testing in setup {test_name}')

        # test normal
        logger.info('Test in normal setting')
        acc, score = test_with_perturbation(model, test_file, test_batch_size=TEST_BATCH_SIZE, test_root=TEST_ROOT, retinex=RETINEX, perturbation=None)
        logger.info(f'Accuracy: {acc}')
        score.to_csv(os.path.join(LOG_ROOT, f'{LOG_NAME}_{test_name}_normal.csv'))

        # test for psnr
        logger.info('Test with added noise')
        for psnr in PSNRS:
            perturbation = partial(add_gaussian_noise_with_psnr, psnr=psnr)
            logger.info(f'Testing with PSNR {psnr}')
            acc, score = test_with_perturbation(model, test_file, test_batch_size=TEST_BATCH_SIZE, test_root=TEST_ROOT, retinex=RETINEX, perturbation=perturbation)
            logger.info(f'Accuracy: {acc}')
            score.to_csv(os.path.join(LOG_ROOT, f'{LOG_NAME}_{test_name}_PSNR_{psnr}.csv'))


        # # test for luminence 

        # ## square
        # perturbation = partial(rescale_luminance, param=None, method='square')
        # logger.info('Testing with quadratical lumination')
        # acc, score = test_with_perturbation(model, test_file, test_batch_size=TEST_BATCH_SIZE, test_root=TEST_ROOT, retinex=RETINEX, perturbation=perturbation)
        # logger.info(f'Accuracy: {acc}')
        # score.to_csv(os.path.join(LOG_ROOT, f'{LOG_NAME}_{test_name}_LUM_square.csv'))


        # ## linear
        # for l in LUMINENCES_LINEAR:
        #     perturbation = partial(rescale_luminance, param=l, method='linear')
        #     logger.info(f'Testing with linear lumination with scale {l}')
        #     acc, score = test_with_perturbation(model, test_file, test_batch_size=TEST_BATCH_SIZE, test_root=TEST_ROOT, retinex=RETINEX, perturbation=perturbation)
        #     logger.info(f'Accuracy: {acc}')
        #     score.to_csv(os.path.join(LOG_ROOT, f'{LOG_NAME}_{test_name}_LUM_linear_{round(l, 2)}.csv'))

        # ## const
        # for c in LUMINENCES_CONST:
        #     perturbation = partial(rescale_luminance, param=c, method='linear')
        #     logger.info(f'Testing with const lumination with scale {c}')
        #     acc, score = test_with_perturbation(model, test_file, test_batch_size=TEST_BATCH_SIZE, test_root=TEST_ROOT, retinex=RETINEX, perturbation=perturbation)
        #     logger.info(f'Accuracy: {acc}')
        #     score.to_csv(os.path.join(LOG_ROOT, f'{LOG_NAME}_{test_name}_LUM_const_{c}.csv'))