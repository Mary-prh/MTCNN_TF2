#coding:utf-8

from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 128
config.LR_BASE = 0.001
# config.CLS_OHEM = True
# config.CLS_OHEM_RATIO = 0.7
# config.BBOX_OHEM = False
# config.BBOX_OHEM_RATIO = 0.7

# config.EPS = 1e-14
# config.LR_EPOCH = [6,14,20]
config.BASE_DIR = './DATA/imglists/'

def get_model_config(net_type):
    if net_type == 'PNet':
        image_size = 12
        loss_weights = {'classifier': 1.0, 'bbox_regress': 0.5, 'landmark_pred': 0.5}
    elif net_type == 'RNet':
        image_size = 24
        loss_weights = {'classifier': 1.0, 'bbox_regress': 0.5, 'landmark_pred': 0.5}
    else:  # ONet
        image_size = 48
        loss_weights = {'classifier': 1.0, 'bbox_regress': 0.5, 'landmark_pred': 1.0}
    return image_size, loss_weights

