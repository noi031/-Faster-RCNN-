import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'env'))
import json, cv2, torch, argparse, numpy as np, json
from PIL import Image
from utils import *
from tqdm import tqdm

args = argparse.ArgumentParser()
args.add_argument("-result_path", type = str)
args.add_argument('-target_dir', type = str)
args.add_argument('-image_dir', type = str)
args = args.parse_args()

with open(args.result_path, 'r', encoding = 'utf-8') as f:
    pred_result = json.load(f)['result']
f1, img_count, preds = 0, 0, {}
for item in pred_result:
    image_id = item['image_id']
    if preds.get(image_id) is None:
        preds[image_id] = []
    preds[image_id].append(item)
TP_all, FP_all, FN_all = [0] * 10, [0] * 10, [0] * 10
for filename in tqdm(os.listdir(args.image_dir), ncols = 100):
    dirname = filename.split('.')[0]
    img_count += 1
    pred = preds.get(int(dirname))
    if pred is None:
        pred = []
    boxes, masks, box_labels, mask_labels = [], [], [], []
    img_size = Image.open(os.path.join(args.image_dir, dirname + '.jpg')).size
    for p in pred:
        if p['type'] < 8:
            box_labels.append(p['type'])
            x1, y1, h, w = p['x'], p['y'], p['height'], p['width']
            x2, y2 = x1 + w, y1 + h
            boxes.append(torch.tensor([x1, y1, x2, y2]))
        else:
            mask_labels.append(p['type'])
            m = np.zeros([img_size[1], img_size[0]])
            contours = [np.array([c[i : i + 2] for i in range(0, len(c), 2)])
                        for c in p['segmentation']]
            cv2.fillPoly(m, contours, 1)
            masks.append(torch.from_numpy(m))
    box_pred = {'boxes': torch.stack(boxes).float().cuda() if boxes != [] else [],
                'labels': torch.tensor(box_labels).long().cuda() if box_labels != [] else []}
    mask_pred = {'masks': torch.stack(masks).to(torch.uint8).cuda() if masks != [] else [],
                'labels': torch.tensor(mask_labels).long().cuda() if mask_labels != [] else []}
    boxes, masks, box_labels, mask_labels = [], [], [], []
    with open(os.path.join(args.target_dir, dirname + '.json'), 'r', encoding = 'utf-8') as f:
        for r in json.load(f):
            if r['type'] < 8:
                x1, y1, h, w = r['x'], r['y'], r['height'], r['width']
                x2, y2 = x1 + w, y1 + h
                boxes.append(torch.tensor([x1, y1, x2, y2]))
                box_labels.append(r['type'])
            else:
                c, sf = r['segmentation'][0], 1
                contour = np.array([c[i : i + 2] for i in range(0, len(c), 2)])
                while sf <= 16:
                    m = np.zeros([img_size[1] * sf, img_size[0] * sf], dtype = np.uint8)
                    cv2.fillPoly(m, [(contour * sf).astype(np.int32)], 1)
                    m = cv2.resize(m, img_size)
                    if m.sum():
                        break
                    sf *= 2
                if not m.sum():
                    print('area error!')
                masks.append(torch.from_numpy(m))
                mask_labels.append(r['type'])
    box_target = {'boxes': torch.stack(boxes).float().cuda() if boxes != [] else [],
                'labels': torch.tensor(box_labels).long().cuda() if box_labels != [] else []}
    mask_target = {'masks': torch.stack(masks).to(torch.uint8).cuda() if masks != [] else [],
                'labels': torch.tensor(mask_labels).long().cuda() if mask_labels != [] else []}
    bTP, bFP, bFN = metric4box(box_pred, box_target, [1, 2, 3, 4, 5, 6, 7])
    mTP, mFP, mFN = metric4mask(mask_pred, mask_target, [8, 9, 10])
    for i in range(7):
        TP_all[i] += bTP[i]
        FP_all[i] += bFP[i]
        FN_all[i] += bFN[i]
    for i in range(7, 10):
        TP_all[i] += mTP[i - 7]
        FP_all[i] += mFP[i - 7]
        FN_all[i] += mFN[i - 7]
print('TP:', TP_all)
print('FP:', FP_all)
print('FN:', FN_all)
f1 = []
for i in range(10):
    P = TP_all[i] / (TP_all[i] + FP_all[i] + 1e-6)
    R = TP_all[i] / (TP_all[i] + FN_all[i] + 1e-6)
    f1.append(2 * P * R / (P + R + 1e-6))
print('F1:', f1)
print(sum(f1) / 10)