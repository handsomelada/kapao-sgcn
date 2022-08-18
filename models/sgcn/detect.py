import torch
from models.sgcn.st_gcn import Model

# ----------------------------------- #
#   class Detect():
#        1.__init__(self, weights, class_thr, classmap_path)：初始化
#            weights：动作分类模型权重文件路径
#            class_thr： person bbox 阈值，小于此阈值不进行动作分类
#            classmap_path：动作分类class map文件路径
#        2.load_model(self)： 加载动作分类模型
#        3.get_skeleton(self, detection)： 输入：单帧yolopose输出  输出：单帧person列表，包含每个人keypoints,bbox
#            detection：yolopose输入，输入为(1, number_person, 57)
#               1: 单帧
#               number_person: 一帧中人数
#               57: 6+51 (6:bbox+conf+class)(51: 17 keypoints * 3)
#        4.inference(self, det)： 输入：单帧yolopose输出  输出：单帧action_result列表，包含每个人bbox,action,confidence
# ----------------------------------- #

# ----------------------------------- #
#   接口使用方法：
#       1.实例化类，并初始化加载模型，例：detector = Detect(action_weights, opt.conf_thres, label_name_path)
#       2.使用inference方法获得动作识别模型的输出，例：results = detector.inference(det)
# ----------------------------------- #

class Detect():
    def __init__(self, weights, class_thr, classmap_path):
        self.weights = weights
        self.class_thr = class_thr
        self.in_channels = 3
        self.num_class = 5
        self.edge_importance_weighting = True
        self.graph_args = {'layout': 'coco', 'strategy': 'spatial'}
        self.devices = torch.device("cuda")
        self.load_model()

        with open(classmap_path) as f:
            label_name = f.readlines()
            self.label_name = [line.rstrip() for line in label_name]

    def inference(self, person):
        # person = self.get_skeleton(detection)
        action_result = []
        for i, det in enumerate(person):
            one_person = dict()
            k = torch.from_numpy(det)
            k = k.view(17, 3)
            kpts = torch.zeros(1, 17, 3)
            kpts[:, :, :] = k
            with torch.no_grad():
                keypoints = kpts.float().cuda(0)
                output = self.model(keypoints)
            probability = torch.softmax(output, dim=1)

            pred_label = output.argmax()
            action_label = self.label_name[pred_label]
            # one_person['bbox'] = det.get('bbox')
            one_person['action'] = action_label
            one_person['score'] = probability[0][pred_label]
            action_result.append(one_person)
        return action_result

    def get_skeleton(self, detection):
        person = []
        for det_index, (*xyxy, conf, cls) in enumerate(detection[:, :6]):
            single = dict()
            if cls == 0 and conf >= self.class_thr:
                kpts = detection[det_index, 6:]
                single['keypoints'] = kpts
                coord = xyxy
                single['bbox'] = coord
            person.append(single)
        return person

    def load_model(self):
        pretrained_dict = torch.load(self.weights, map_location=lambda storage, loc: storage.cuda(self.devices))
        self.model = Model(self.in_channels, self.num_class, self.graph_args, self.edge_importance_weighting)
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.to(self.devices)
        self.model.eval()



