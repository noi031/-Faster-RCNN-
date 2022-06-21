from PIL import Image, ImageDraw
import os, json, argparse, cv2, numpy as np

args = argparse.ArgumentParser()
args.add_argument("-result_path", type = str)
args.add_argument('-target_dir', type = str)
args.add_argument('-image_id', type = int)
args.add_argument('-image_dir', type = str)
args.add_argument('-output_path', type = str)
args.add_argument('-show_gt_segs', type = int, default = 1)
args.add_argument('-show_pred_segs', type = int, default = 1)
args = args.parse_args()
with open(args.result_path, 'r') as f:
    data = json.load(f)['result']
data = [x for x in data if x['image_id'] == args.image_id]
for p in os.listdir(args.target_dir):
    if int(p) == args.image_id:
        dirpath = os.path.join(args.target_dir, p)
        imgpath = os.path.join(args.image_dir, p + '.jpg')
        break
img = Image.open(imgpath)
a = ImageDraw.ImageDraw(img)
intcolor = {1 : 'yellow', 2: 'green', 3: 'pink'}
boxes = [[p['x'], p['y'], p['x'] + p['width'], p['y'] + p['height']] for p in data]
for item in boxes:
    a.rectangle(((item[0], item[1]), (item[2], item[3])), fill=None, outline='red', width=3)
with open(os.path.join(dirpath, 'boxes.json'), 'r', encoding = 'utf-8') as f:
    boxes = json.load(f)
boxes = [[box['x1'], box['y1'], box['x2'], box['y2']] for box in boxes]
for item in boxes:
    a.rectangle(((item[0], item[1]), (item[2], item[3])), fill=None, outline='blue', width=3)
img = np.array(img)
if args.show_pred_segs:
    for p in data:
        if p['type'] >= 8:
            pts = [np.array([c[i : i + 2]
                        for i in range(0, len(c), 2)]) for c in p['segmentation']]
            cv2.fillPoly(img, pts,
                        (102, 204, 153)) #green
seg = np.array(Image.open(os.path.join(dirpath, 'seg.png')))
if args.show_gt_segs:
    for v in [1, 2, 3]:
        mask = (seg == v).astype(np.uint8)
        pts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(img, pts, (255, 255, 153)) #yellow
Image.fromarray(img).save(args.output_path)