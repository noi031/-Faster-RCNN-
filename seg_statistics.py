from tqdm import tqdm
from PIL import Image
import os, cv2, warnings, numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import boxcox_normmax
warnings.filterwarnings('ignore')

def boxcox(x, a):
    return (np.power(x, a) - 1) / a if abs(a) > 0.01 else np.log(x)

area_list = [[], [], []]
for pname in ['train', 'dev']:
    rootpath = os.path.join(pname, 'labels')
    for dirname in tqdm(os.listdir(rootpath), ncols = 100):
        dirpath = os.path.join(rootpath, dirname)
        filepath = os.path.join(dirpath, 'seg.png')
        seg = np.array(Image.open(filepath))
        for v in [1, 2, 3]:
            mask = (seg == v).astype(np.uint8)
            pts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in pts:
                if len(c) > 2:
                    a = cv2.contourArea(c[:, 0])
                    if a < 1e-6:
                        continue
                    area_list[v - 1].append(a)
for v in range(3):
    area = np.array(area_list[v])
    a, b = boxcox_normmax(area)
    area = boxcox(area, a)
    plt.figure()
    plt.hist(area, bins = 20)
    plt.savefig('seg_area_dist' + str(v) + '.jpg')
    print({'area_mean': area.mean(), 'area_std': area.std(), 'area_lambda': a})