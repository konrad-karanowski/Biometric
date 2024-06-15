import torch
from torch.nn import DataParallel
from data.dataset import retinex_decomposition
from sklearn.metrics import f1_score, confusion_matrix
from torchvision import transforms as T
from PIL import Image
from models import resnet_face18
import cv2
import numpy as np

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

path = '/home/mzieba/Biometrics/arcface-pytorch/checkpoints/resnet18_retinex/resnet18_30.pth'


MODEL = resnet_face18(False)
model = DataParallel(MODEL)
# load_model(model, opt.test_model_path)
model.load_state_dict(torch.load(path))
model.eval()
model.to(torch.device("cuda"))


normalize = T.Normalize(mean=[0.5], std=[0.5])
THRESHOLD = 0.5
RETINEX = 'L'  



def img2tensor(img, size=(64,64), retinex=None):
    img = cv2.resize(img, size)
    if retinex:
        R, L = retinex_decomposition(img)
        if retinex == 'R':
            img= cv2.cvtColor((R * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            img = L[:, :, 0]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = torch.from_numpy(img)
    img = normalize(img.float()[None, ...]).numpy()
    # img = np.vstack((img, np.fliplr(img)))[:, np.newaxis, :, :]
    # image = image.transpose((2, 0, 1))
    
    # image = image[:, np.newaxis, :, :]
    # image = image.astype(np.float32, copy=False)
    # image -= 127.5
    # image /= 127.5
    #raise Exception(image.shape)
    return img[:, np.newaxis, :, :]


def compare_face(x, y):
    x, y = img2tensor(x), img2tensor(y)
    images = np.concatenate([x, y], axis=0)
    data = torch.from_numpy(images)
    data = data.to(torch.device("cuda"))
    output = model(data)
    output = output.data.cpu().numpy()
    fe_1 = output[0]
    fe_2 = output[1]
    sim = cosin_metric(fe_1, fe_2)
    return sim > THRESHOLD, sim.item()
