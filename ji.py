import json
import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import torch
import argparse
import yaml
import numpy as np
from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords
from utils.datasets import LoadImages
from models.experimental import attempt_load
from val import run_nms, post_process_batch
from models.sgcn.detect import Detect

model_class_names = ["others_unsure", "squatting_unsure", "sitting_unsure", "standing_unsure", "lying_unsure", "others",\
    "squatting", "sitting", "standing", "lying"]  ###模型榜，需要检测的类别名称
alert_class_names = ["sitting", "lying"]  ###实战榜，需要检测的类别名称


def kapao_sgcn(img_path):
    parser = argparse.ArgumentParser()

    # plotting options
    parser.add_argument('--bbox', default=True)
    parser.add_argument('--kp-bbox', action='store_true')
    parser.add_argument('--pose', default=True)
    parser.add_argument('--face', action='store_true')
    parser.add_argument('--color-pose', type=int, nargs='+', default=[255, 0, 255], help='pose object color')
    parser.add_argument('--color-kp', type=int, nargs='+', default=[0, 255, 255], help='keypoint object color')
    parser.add_argument('--line-thick', type=int, default=2, help='line thickness')
    parser.add_argument('--kp-size', type=int, default=1, help='keypoint circle size')
    parser.add_argument('--kp-thick', type=int, default=2, help='keypoint circle thickness')

    # model options
    parser.add_argument('--data', type=str, default='../data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--weights', default='../kapao_l_coco.pt')
    parser.add_argument('--action-weights', default='../weights/five_layer.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--overwrite-tol', type=int, default=25)
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])
    parser.add_argument('--label-map', type=str, default='../data/label_map.txt')

    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    # add inference settings to data dict
    data['imgsz'] = args.imgsz
    data['conf_thres'] = args.conf_thres
    data['iou_thres'] = args.iou_thres
    data['use_kp_dets'] = not args.no_kp_dets
    data['conf_thres_kp'] = args.conf_thres_kp
    data['iou_thres_kp'] = args.iou_thres_kp
    data['conf_thres_kp_person'] = args.conf_thres_kp_person
    data['overwrite_tol'] = args.overwrite_tol
    data['scales'] = args.scales
    data['flips'] = [None if f == -1 else f for f in args.flips]
    data['count_fused'] = False

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)
    detector = Detect(args.action_weights, args.conf_thres, args.label_map)

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(img_path, img_size=imgsz, stride=stride, auto=True)

    (_, img, im0, _) = next(iter(dataset))
    img = torch.from_numpy(img).to(device)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    out = model(img, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]
    person_dets, kp_dets = run_nms(data, out)

    bboxes = scale_coords(img.shape[2:], person_dets[0][:, :4], im0.shape[:2]).round().cpu().numpy()
    bbox_dict = []
    for i, k in enumerate(bboxes):
        bb = dict()
        bb['x'], bb['y'] = k[0], k[1]
        bb['width'] = k[2] - k[0]
        bb['height'] = k[3] - k[1]
        bbox_dict.append(bb)

    _, poses, scores, ids, n_fused = post_process_batch(data, img, [], [[im0.shape[:2]]], person_dets, kp_dets)
    # print(poses)
    total = []
    for i in poses:
        keypoints = dict()
        keypoints['keypoints'] = i.flatten().tolist()
        keypoints['score'] = 0.7
        total.append(keypoints)

    aaa = detector.inference(poses)
    zz = zip(bbox_dict, scores, total, aaa)

    dets = []
    for i, t in enumerate(list(zz)):
        objs = []
        objs.append(t[0]['x'])
        objs.append(t[0]['y'])
        objs.append(t[0]['width'])
        objs.append(t[0]['height'])
        objs.append(t[3]['action'])
        objs.append(t[1])
        objs.append(t[2])
        dets.append(objs)
    return dets


def process_image(net, input_image, args=None):
    dets = net(input_image)
    target_info, objects = [], []
    for obj in dets:
        x, y, width, height, name, conf, kpts = obj  ###检测框用x,y,width,height表示, 也可以
        keypoints = kpts['keypoints']
        '''
        keypoints是一个长度为17*3的列表，里面的内容示例如下
        keypoints = [x1,y1,v1, ......,x17,y17,v17]
        总共17个点，每个点有3个元素值x,y,v
        x,y表示点的坐标值,v是个标志位,v为0时表示这个关键点没有标注（这种情况下您可以忽略这个关键点的标注），
        v为1时表示这个关键点标注了但是不可见（被遮挡了），v为2时表示这个关键点标注了同时也可见
        17个点依次对应的人体骨骼关键点名称是:
        ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        开发者需要按照这个名称顺序来存储关键点信息到keypoints里, 不能乱序
        '''

        if name in model_class_names:
            obj = {'x': x, 'y': y, 'width': width, 'height': height, 'confidence': conf, 'name': name,
                   ###检测框用x,y,width,height表示, 也可以
                   'keypoints': {'keypoints': keypoints, 'score': kpts['score']}}
            objects.append(obj)

        if name in alert_class_names:
            alert_obj = {'x': x, 'y': y, 'width': width, 'height': height, 'confidence': conf, 'name': name,
                         ###检测框用x,y,width,height表示, 也可以
                         'keypoints': {'keypoints': keypoints, 'score': kpts['score']}}
            target_info.append(alert_obj)

    target_count = len(target_info)
    is_alert = True if target_count > 0 else False
    return json.dumps(
        {'algorithm_data': {'is_alert': is_alert, 'target_count': target_count, 'target_info': target_info},
         'model_data': {"objects": objects}})

'''
ev_sdk输出json样例
{"algorithm_data": {"is_alert": True, "target_count": 1, "target_info": [
    {"x": 1805, "y": 886, "width": 468, "height": 595, "confidence": 0.7937217950820923, "name": "lying",
     "keypoints": {"keypoints": [2161.423828125, 990.58984375, 1.0, 2161.423828125, 981.29296875, 1.0, 2161.423828125,
                                 981.29296875, 1.0, 2093.238525390625, 967.34765625, 1.0, 2124.231689453125,
                                 985.94140625, 1.0, 2031.251708984375, 995.23828125, 1.0, 2068.443603515625,
                                 1069.61328125, 1.0, 2093.238525390625, 1074.26171875, 1.0, 2124.231689453125,
                                 1190.47265625, 1.0, 2161.423828125, 1088.20703125, 1.0, 2198.615966796875,
                                 1130.04296875, 1.0, 1944.47021484375, 1185.82421875, 1.0, 2012.6556396484375,
                                 1236.95703125, 1.0, 2124.231689453125, 1106.80078125, 0.0, 2186.218505859375,
                                 1195.12109375, 1.0, 2130.430419921875, 1232.30859375, 0.0, 2173.8212890625,
                                 1246.25390625, 1.0], "score": 0.7535077333450317}}]}, 
                  "model_data": {"objects": [
    {"x": 1805, "y": 886, "width": 468, "height": 595, "confidence": 0.7937217950820923, "name": "standing",
     "keypoints": {"keypoints": [2161.423828125, 990.58984375, 1.0, 2161.423828125, 981.29296875, 1.0, 2161.423828125,
                                 981.29296875, 1.0, 2093.238525390625, 967.34765625, 1.0, 2124.231689453125,
                                 985.94140625, 1.0, 2031.251708984375, 995.23828125, 1.0, 2068.443603515625,
                                 1069.61328125, 1.0, 2093.238525390625, 1074.26171875, 1.0, 2124.231689453125,
                                 1190.47265625, 1.0, 2161.423828125, 1088.20703125, 1.0, 2198.615966796875,
                                 1130.04296875, 1.0, 1944.47021484375, 1185.82421875, 1.0, 2012.6556396484375,
                                 1236.95703125, 1.0, 2124.231689453125, 1106.80078125, 0.0, 2186.218505859375,
                                 1195.12109375, 1.0, 2130.430419921875, 1232.30859375, 0.0, 2173.8212890625,
                                 1246.25390625, 1.0], "score": 0.7535077333450317}}]}}  #score不影响测试分数
'''

'''
(1). 目标检测框使用f1‑score作为指标
(2). 关键点检测使用Average Precision (AP): AP at OKS=.50:.05:.95作为测试指标
测试配置权重参数w, 最终精度得分 = w * 第(1)项的精度分 + (1 - w) * 第(2)项的精度分
'''