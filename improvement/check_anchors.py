import os, json, torch
from modules.cv.utils import anchor_matcher
from tqdm import tqdm
from PIL import Image
from modules.cv.ops.boxes import resize as box_resize

anchor_sizes = tuple((x, x * 2, x * 2 ** (1 / 3)) for x in [8, 16, 32, 64, 128])
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
all_box_num, matched_box_num, matched_class_num, all_class_num = 0, 0, [0] * 7, [0] * 7
matcher = anchor_matcher(anchor_sizes, aspect_ratios, 0.5, 0.4, True)
for pname in ['train', 'dev']:
    dirpath = '/mnt/nfs2/ikcest/data/' + pname + '/labels'
    for name in tqdm(os.listdir(dirpath), ncols = 100):
        filepath = os.path.join(os.path.join(dirpath, name), 'boxes.json')
        with open(filepath, 'r', encoding = 'utf-8') as f:
            det = json.load(f)
        det_labels = [label['type'] for label in det if label['type']]
        det = [[label['x1'], label['y1'], label['x2'], label['y2']] for label in det if label['type']]
        all_box_num += len(det)
        if len(det):
            det = torch.as_tensor(det, dtype = torch.float32).cuda()
            det_labels = torch.as_tensor(det_labels, dtype = torch.int64).cuda()
            for i in range(7):
                all_class_num[i] += torch.sum(det_labels == i + 1).item()
            filepath = os.path.join('/mnt/nfs2/ikcest/data/' + pname + '/images', name + '.jpg')
            image_size = Image.open(filepath).size
            det = box_resize(det, [image_size[1], image_size[0]], [1024, 1024])
            feature_sizes = [[1024 // (2 ** (i + 2)), 1024 // (2 ** (i + 2))] for i in range(4)]
            feature_sizes += [feature_sizes[-1]]
            matched_idxs = matcher.match(det, [1024, 1024], feature_sizes)
            matched_box_num += len(matched_idxs)
            det_labels = det_labels[matched_idxs]
            for i in range(7):
                matched_class_num[i] += torch.sum(det_labels == i + 1).item()
print('anchors matched rate:', matched_box_num / all_box_num)
print('anchors matched per class:', [a / b for a, b in zip(matched_class_num, all_class_num)])