import os, json, matplotlib, math
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

alldet, area, ratio = [], [], []
for pname in ['train', 'dev']:
    dirpath = '/mnt/nfs2/ikcest/data/' + pname + '/labels'
    for name in tqdm(os.listdir(dirpath), ncols = 100):
        filepath = os.path.join(os.path.join(dirpath, name), 'boxes.json')
        with open(filepath, 'r', encoding = 'utf-8') as f:
            det = json.load(f)
        det = [label for label in det if label['type']]
        filepath = os.path.join('/mnt/nfs2/ikcest/data/' + pname + '/images', name + '.jpg')
        image_size = Image.open(filepath).size
        ws = [(label['x2'] - label['x1']) / image_size[0] for label in det]
        hs = [(label['y2'] - label['y1']) / image_size[1] for label in det]
        area += [w * h for w, h in zip(ws, hs)]
        ratio += [h / w for w, h in zip(ws, hs)]
        alldet += det
num_bins = int(math.sqrt(len(alldet)))
area = [x for x in area if x < 5e-4]
plt.figure()
plt.hist(area, num_bins)
plt.xlabel('area')
plt.ylabel('freq')
plt.savefig('area histogram.png')
plt.figure()
plt.hist(ratio, num_bins)
plt.xlabel('ratio(height / width)')
plt.ylabel('freq')
plt.savefig('height-width ratio histogram.png')
with open('all_boxes.json', 'w', encoding = 'utf-8') as f:
    json.dump(alldet, f)