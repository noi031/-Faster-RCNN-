import os, shutil, json, cv2, numpy as np
from types import prepare_class
from tqdm import tqdm
from PIL import Image

if os.path.exists('train'):
    shutil.rmtree('train')
os.mkdir('train')
if os.path.exists('dev'):
    shutil.rmtree('dev')
os.mkdir('dev')
for i in ['train', 'dev']:
    for j in ['images', 'labels']:
        os.mkdir(i + '/' + j)
np.random.seed(20)
image_dir = os.listdir('images')
isdev = np.zeros([len(image_dir)])
isdev[np.random.choice(len(image_dir), 4000, False)] = 1
for idx, filename in enumerate(tqdm(image_dir, ncols = 100)):
    name = filename.split('.')[0]
    image_size = Image.open(os.path.join('images', filename)).size
    with open(os.path.join('labels', name + '.json'), 'r', encoding = 'utf-8') as f:
        labels = json.load(f)
    pname = 'dev' if isdev[idx] else 'train'
    shutil.copyfile(os.path.join('images', filename), pname + '/images/' + filename)
    labeldir = pname + '/labels/' + name
    os.mkdir(labeldir)
    box_labels, seg = [], np.zeros([image_size[1], image_size[0]], dtype = np.uint8)
    for label in labels:
        if label['type'] < 8:
            x, y, w, h = label['x'], label['y'], label['width'], label['height']
            if w < 1e-2 or h < 1e-2:
                continue
            box_labels.append({'type': label['type'], 'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h})
        else:
            contours = [np.asarray([c[i : i + 2] for i in range(0, len(c), 2)], dtype = np.int32)
                        for c in label['segmentation']]
            cv2.fillPoly(seg, contours, label['type'] - 7)
    if box_labels == []:
        box_labels.append({'type': 0, 'x1': 0, 'y1': 0, 'x2': image_size[0], 'y2': image_size[1]})
    with open(os.path.join(labeldir, 'boxes.json'), 'w', encoding = 'utf-8') as f:
        json.dump(box_labels, f)
    Image.fromarray(seg).save(os.path.join(labeldir, 'seg.png'))
print('done.')
