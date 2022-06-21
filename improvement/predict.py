import sys, os, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'env'))
import torch, cv2, numpy as np, rapidjson
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from utils import *
from time import time
from multiprocessing.pool import ThreadPool
from modules.cv import nvJPEGDataloader
from modules.cv.ops.boxes import resize as box_resize
from modules.cv.models.detection.utils import AnchorGenerator
from torch.cuda.amp import autocast_mode
from det_seg import det_seg_resnet
from scipy.stats import norm
warnings.filterwarnings('ignore')

pool = ThreadPool(4)

def boxcox(x, a):
    return (np.power(x, a) - 1) / a if abs(a) > 0.01 else np.log(x)

def find_seg(seg, image_id, result, seg_thresh, arsts):
    try:
        seg = (seg > seg_thresh[:, None, None]).cpu().numpy().astype(np.uint8)
        for v, mask in enumerate(seg):
            area_thresh = norm.ppf(1 - arsts['thresh'][v], loc = 0, scale = 1)
            pts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            npts = [c[:, 0] for c in pts if len(c) > 2]
            pts, area = [], []
            for c in npts:
                a = cv2.contourArea(c)
                if a >= 1e-6:
                    pts.append(c)
                    area.append(a)
            if not len(pts):
                continue
            area = (boxcox(np.array(area), arsts['lambda'][v]) - arsts['mean'][v]) / arsts['std'][v]
            for a, c in zip(area, pts):
                if a >= area_thresh:
                    result.append(rapidjson.RawJSON(
                        rapidjson.dumps({'image_id': image_id, 'type': v + 8, 'x': 0, 'y': 0,
                                        'width': 0, 'height': 0, 'segmentation': [c.ravel().tolist()]})))
    except:
        print('threads failed!')

class Rcnn_Deeplab_with_resize(nn.Module):
    def __init__(self, model):
        super(Rcnn_Deeplab_with_resize, self).__init__()
        self.model = model
    def forward(self, images, ori_sizes, seg_size):
        image_size = images.shape[-2:]
        det_preds, seg_preds = self.model(images, seg_size)
        ori_sizes = [[y.item() for y in x] for x in ori_sizes]
        det_boxes = [box_resize(pred['boxes'], image_size, ori_size)
                    for pred, ori_size in zip(det_preds, ori_sizes)]
        seg_preds = [F.interpolate(pred[None].float(), ori_size)[0]
                    for pred, ori_size in zip(seg_preds, ori_sizes)]
        return [{'boxes': boxes, 'labels': box['labels'], 'scores': box['scores'], 'seg': seg}
                for boxes, box, seg in zip(det_boxes, det_preds, seg_preds)]

class model_wrapper():
    def __init__(self, args, seg_size, pretrained = True):
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                            for x in [16, 32, 64, 128, 256])
        # anchor_sizes = tuple((x,) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2),) * len(anchor_sizes)
        model = Rcnn_Deeplab_with_resize(det_seg_resnet('resnet50',
                                    det_classes = args.det_classes, seg_classes = args.seg_classes,
                                    pretrained_backbone = pretrained,
                                    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios, True),
                                    # box_score_thresh = [0.8, 0.6, 0.55, 0.55, 0.4, 0.65, 0.4],
                                    box_score_thresh = [0.8, 0.6, 0.55, 0.5, 0.5, 0.65, 0.4],
                                    rpn_score_thresh = 0, rpn_nms_thresh = 0.7,
                                    box_detections_per_img = 50, box_nms_thresh = 0.5,
                                    returned_layers = [1, 2, 3, 4]))
        self.model, self.seg_size = model.cuda(), seg_size
        self.model.eval()
    def parameters(self):
        return self.model.parameters()
    def predict(self, images, ori_sizes):
        with autocast_mode.autocast():
            r = self.model(images, ori_sizes, self.seg_size)
        return r
    def load(self, model_path):
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                model_k = k.split('.')
                if model_k[0] == 'module':
                    model_k = '.'.join(model_k[1:]) #'module.xxx'
                else:
                    model_k = '.'.join(model_k)
                pv = model_state_dict.get(model_k)
                if pv is not None and pv.shape == v.shape:
                    model_state_dict[model_k].data = v.data
            self.model.load_state_dict(model_state_dict)

class testDataset(Dataset):
    def __init__(self, file_list):
        self.file_list, self.len, self.num_outs = file_list, len(file_list), 2
    def __getitem__(self, index):
        filepath = self.file_list[index]
        image_id, _ = os.path.splitext(os.path.split(filepath)[1])
        return filepath, torch.tensor(int(image_id))
    def __len__(self):
        return self.len

if __name__ == '__main__':
    st_time = time()
    torch.random.manual_seed(20);np.random.seed(20)
    args = argparse.ArgumentParser()
    args.add_argument("input_path", type = str)
    args.add_argument("output_path", type = str)
    args.add_argument('-dc', "--det_classes", type = int, help = 'number of detection classes',
                        default = 7)
    args.add_argument('-sc', "--seg_classes", type = int, help = 'number of segmentation classes',
                        default = 3)
    args = args.parse_args()
    with open(args.input_path, 'r') as f:
        input_list = f.read().split('\n')
    if input_list[-1] == '':
        del input_list[-1]
    model = model_wrapper(args, (1080, 1920), False)
    model.load('./model/model.pkl')
    testloader = nvJPEGDataloader(testDataset(input_list), 16, (1024, 1024), return_ori_sizes = True,
                                    num_threads = 4)
    result = []
    means = torch.tensor([0.485, 0.456, 0.406]).cuda()
    stds = torch.tensor([0.229, 0.224, 0.225]).cuda()
    seg_thresh = torch.tensor([0.3, 0.3, 0.3]).cuda()
    for data in testloader:
        images, image_id, ori_sizes = data
        images = ((images / 255.0 - means) / stds).permute([0, 3, 1, 2])
        ori_sizes = ori_sizes[:, :2]
        with torch.no_grad():
            r = model.predict(images, ori_sizes)
        for x, idx in zip(r, image_id):
            image_id = idx.item()
            area_stats = {'lambda': [0.161102802793638, 0.18215981444690393, 0],
                        'mean': [13.980377675873042, 15.038020724509117, 7.31037546484126],
                        'std': [5.175684862015474, 5.173897616185673, 2.4002681513204807],
                        'thresh': [0.85, 0.85, 0.85]}
            pool.apply_async(find_seg, (x['seg'], image_id, result, seg_thresh, area_stats))
            boxes, labels = x['boxes'].cpu().numpy(), x['labels'].cpu().numpy()
            for i, label in enumerate(labels):
                x1, y1, x2, y2 = boxes[i].tolist()
                result.append(rapidjson.RawJSON(
                    rapidjson.dumps({'image_id': int(image_id), 'type': int(label + 1),
                    'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1, 'segmentation': []})))
    pool.close();pool.join()
    with open(args.output_path, 'w', encoding = 'utf-8') as f:
        f.write(rapidjson.dumps({'result': result}))
    fps = 4000 / (time() - st_time)
    # if fps < 20:
    #     os.remove(args.output_path)
    #     raise ValueError('time limit exceeded!')
    print(fps)