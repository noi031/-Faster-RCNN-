import torch, os, shutil, json, warnings, argparse, numpy as np, random
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from PIL import Image
from modules.cv.process import Trainprocess
from modules.cv.preprocess import *
from modules.cv.ops.boxes import resize as box_resize
from modules.cv.models.detection.utils import AnchorGenerator
from det_seg import det_seg_resnet
warnings.filterwarnings('ignore')

class Rcnn_Deeplab_with_resize(nn.Module):
    def __init__(self, model):
        super(Rcnn_Deeplab_with_resize, self).__init__()
        self.model = model
    def forward(self, images, image_size, seg_size, targets):
        """
        This function wraps up the det_seg model with resizing.
        Parameters:
            images: list of Tensors each of which is of shape (C, H, W).
            image_size: Tuple (H, W), representing the size into which all images in a batch is
                transformed.
            seg_size: Tuple (H, W), representing the size feature maps are interpolated into after
                    the last conv layer.
            targets: list[Dict[str, Tensor]], each element in this list is a dict with keys
                'boxes', 'seg', 'labels', where 'boxes' is a tensor of shape (n, 4) representing
                bounding boxes coordinates for an image, the four numbers are x1, y1, x2, y2. 'seg'
                is a tensor of shape (H, W), each pixel represents the class it belongs to. 'labels'
                is a tensor of shape (n) representing the label that boxes have.
        Returns:
            This module returns a dict with keys 'det_loss' representing the
            detection loss and 'seg_loss' representing the segmentation loss. Both of them are
            of shape (batch_size). But if segmentaion is not needed, seg_loss = 0.
        """
        ori_sizes = [img.shape[-2:] for img in images]
        images = [F.interpolate(img[None], image_size)[0] for img in images]
        boxes = [box_resize(target['boxes'], ori_size, image_size)
                        for target, ori_size in zip(targets, ori_sizes)]
        gt_boxes = [{'boxes': box, 'labels': target['labels']} for box, target in zip(boxes, targets)]
        gt_segs = [F.interpolate(target['seg'][None, None].float(), seg_size)[0, 0].long()
                    for target in targets]
        gt_segs = torch.stack(gt_segs)
        return self.model(torch.stack(images), seg_size, gt_segs, gt_boxes)

class model_wrapper():
    def __init__(self, args, train_size, test_size, seg_size, pretrained = True):
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                            for x in [16, 32, 64, 128, 256])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        model = Rcnn_Deeplab_with_resize(det_seg_resnet('resnet50',
                                    det_classes = args.det_classes, seg_classes = args.seg_classes,
                                    pretrained_backbone = pretrained,
                                    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios),
                                    box_score_thresh = 0.3, box_detections_per_img = 50,
                                    box_nms_thresh = 0.5, returned_layers = [1, 2, 3, 4]))
        self.model = model.cuda()
        self.train_size, self.test_size = train_size, test_size
        self.image_size, self.args = test_size, args
        self.batch_id, self.seg_size = 0, seg_size
    def switch_size(self):
        if self.model.training:
            self.image_size = random.choice(self.train_size)
    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
        self.image_size = self.test_size
    def parameters(self):
        return self.model.parameters()
    def save(self, epoch, val_loss):
        val = 0 if val_loss is None else val_loss['loss']
        torch.save(self.model.state_dict(), './model/' + str(epoch) + '.pkl')
        with open('./model/' + str(epoch) + '.txt', 'w', encoding = 'utf-8') as f:
            f.write(str(val))
    def get_loss(self, inputs):
        if self.batch_id % 10 == 0:
            self.switch_size()
            self.batch_id = 0
        self.batch_id += 1
        images, targets = inputs
        img = [x.cuda() for x in images]
        new_targets = []
        for x in targets:
            y = {}
            for k, v in x.items():
                y[k] = v.cuda()
            new_targets.append(y)
        loss = torch.stack(list(self.model(img, self.image_size, self.seg_size, new_targets).values()))
        loss_show = {'r_c': loss[0].item(), 'r_r': loss[1].item(),
                    'b_c': loss[2].item(), 'b_r': loss[3].item(), 's': loss[4].item()}
        loss = loss.sum()
        if self.model.training:
            loss.backward()
        return loss_show
    def metric_eval(self, inputs):
        return {'loss': sum(self.get_loss(inputs).values())}
    def metric_agg(self, metric_list):
        return {'loss': sum(metric_batch['loss'] for metric_batch in metric_list) / len(metric_list)}
    def load(self, model_path):
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

class trainDataset(Dataset):
    def __init__(self, aug_method, rootpath):
        self.aug = data_augmentation(aug_method)
        self.rootpath, cur_index, self.name_list = rootpath, 0, []
        for filename in os.listdir(os.path.join(rootpath, 'images')):
            self.name_list.append(''.join(filename.split('.')[:-1]))
            cur_index += 1
        self.rootpath, self.len = rootpath, cur_index
    def __getitem__(self, index):
        filename = self.name_list[index]
        path = os.path.join(self.rootpath, 'images')
        img = np.array(Image.open(os.path.join(path, filename + '.jpg')))
        path = os.path.join(self.rootpath, 'labels')
        path = os.path.join(path, filename)
        with open(os.path.join(path, 'boxes.json'), 'r', encoding = 'utf-8') as f:
            all_box = json.load(f)
        all_label = [x['type'] - 1 for x in all_box]
        all_box = [torch.tensor([x['x1'], x['y1'], x['x2'], x['y2']]) for x in all_box]
        boxes = torch.stack(all_box).float() if all_box != [] else torch.zeros([0, 4], dtype = torch.float)
        seg = torch.from_numpy(np.array(Image.open(os.path.join(path, 'seg.png')))).long()
        labels = torch.tensor(all_label).long()
        return self.aug.augment(img), {'boxes': boxes, 'seg': seg, 'labels': labels}
    def __len__(self):
        return self.len

def collate_fn(batch):
    images, targets = [], []
    for x in batch:
        images.append(x[0])
        targets.append(x[1])
    return images, targets

if __name__ == '__main__':
    torch.random.manual_seed(20);np.random.seed(20);random.seed(20)
    args = argparse.ArgumentParser()
    args.add_argument('-dc', "--det_classes", type = int, help = 'number of detection classes',
                        default = 7)
    args.add_argument('-sc', "--seg_classes", type = int, help = 'number of segmentation classes',
                        default = 3)
    args = args.parse_args()
    model = model_wrapper(args, [(1024, 1024), (768, 768), (512, 512), (1024, 768), (768, 1024),
                                (512, 1024), (1024, 512), (768, 512), (512, 768), (832, 832),
                                (832, 1024), (832, 768)], (1024, 1024), (1080, 1920))
    model.load('./model.pkl')
    trainset = trainDataset(['noise', 'normalize'], '../data/train')
    devset = trainDataset(['normalize'], '../data/dev')
    trainloader = DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2,
                            collate_fn = collate_fn)
    devloader = DataLoader(devset, batch_size = 4, num_workers = 2, collate_fn = collate_fn)
    train_wrapper = Trainprocess(model, trainloader, devloader)
    if os.path.exists('model'):
        shutil.rmtree('model')
    os.mkdir('model')
    #train_wrapper.train(0)
    train_wrapper.train(30)