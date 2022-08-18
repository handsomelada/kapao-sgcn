import torch
import os
import json
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import numpy as np
import logging
import time
from torch import optim
from torch.utils.data import Dataset
from models.sgcn.st_gcn import Model


def train_logging(file_path):
    logging.basicConfig(filename=file_path, level=logging.DEBUG,
                        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
                        datefmt="%a %d %b %Y %H:%M:%S")
    logging.debug('debug')
    logging.info('info')
    logging.warning('warning')
    logging.error('Error')
    logging.critical('critical')



# 1.load data

class MyDataset(Dataset):
    def __init__(self, root_path, transform=None, target_transform=None):
        self.root_path = root_path
        self.transform = transform
        self.target_transform = target_transform

        self.list = os.listdir(self.root_path)
        self.json_list = []
        for i, f in enumerate(list):
            if f[-4:] == 'json':
                self.json_list.append(f)
        total_list = []

        with open(self.annot_file, encoding='utf-8') as annot:
            self.result = json.load(annot)

    def ground_truth_parser(self, json_path):
        f = open(json_path, 'rb')
        infos = json.load(f)
        bbox_anno, kpts_anno = [], []
        for info in infos:
            xmin, ymin, width, height = info['bbox']  ###检测框的左上角坐标和高宽
            box_name = info['box_name']  ###检测框的名称
            bbox_anno.append({'name': box_name, 'xmin': xmin, 'ymin': ymin, 'width': width, 'height': height})

            anno = {'keypoints': info['keypoints'], 'num_keypoints': 17,
                    'category_id': 1, 'id': info['id'], 'bbox': info['bbox'], 'area': info['area'],
                    'iscrowd': int(info['iscrowd'])}  ###关键点的标注信息
            kpts_anno.append(anno)
        return bbox_anno, kpts_anno

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, item):
        json_name = self.json_list[item]
        json_path = os.path.join(self.root_path, json_name)
        bbox, kpts = self.ground_truth_parser(json_path)

        keypoints = np.array(kpts[item]['keypoints'])
        keypoints = keypoints.reshape(17, 3)
        action = np.array(bbox[item]['name'])

        return keypoints, action


def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # args define
    device = 0
    root_path = 'data/dataset/out11.json'
    log_path = 'four_test_result.log'
    total_epoch = 1000
    in_channels = 3
    num_class = 5
    edge_importance_weighting = True
    save_best = False
    graph_args = {'layout': 'coco', 'strategy': 'spatial'}

    # logging
    train_logging(log_path)

    # 1.load data
    mydata = MyDataset(root_path)
    trainset = torch.utils.data.DataLoader(
        dataset=mydata,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        worker_init_fn=init_seed
    )

    # 2.load model
    model = Model(in_channels, num_class, graph_args, edge_importance_weighting)
    devices = torch.device("cuda")
    model.to(devices)

    # 3.define loss
    CELloss = nn.CrossEntropyLoss().cuda(device)

    # 4.define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0004)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500, 750], gamma=0.1, last_epoch=-1)

    # 5.start train
    model.train()
    train_record = dict()
    least_loss = 100.
    for epo in range(total_epoch):
        start = time.time()
        total_loss = []
        print(f'epoch:{epo}')
        for step, (keypoints, label) in enumerate(trainset):
            with torch.no_grad():
                keypoints = keypoints.float().cuda(device)
                label = label.cuda(device)
            # forward
            output = model(keypoints)
            loss = CELloss(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.data.item())
        mean_epoch_loss = np.mean(total_loss)
        logging.info(
            f"\tEpoch: {epo + 1}/{total_epoch}\tcost: {time.time() - start:.4f}\tloss: {mean_epoch_loss:.4f}")
        if mean_epoch_loss < least_loss:
            logging.info(f'save epoch:{epo+1}')
            print(f'save epoch:{epo+1}')
            save_best = True
            least_loss = mean_epoch_loss
        once = '\tMean training loss: {:.4f}.'.format(mean_epoch_loss)
        print(once)

    # 7.save weights
        if save_best == True:
            state_dict = model.state_dict()
            torch.save(state_dict, 'weights/five_layer.pt')





