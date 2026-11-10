"""The numbers of Part 7: object detection (Lecture 26) on the lecture's courtyard photo.

    ~/base/bin/python scripts/sci/p7.py      writes data/p7.json

Recomputed here and asserted against the lecture: the IoU of the dog's box and the 60-pixel guess, the target
records of the 3 x 2 grid, the squared-error loss of the untrained network's table, class-aware NMS over the
detector's 31 boxes at every score threshold the live figure offers, and precision, recall and AP over the eight
ranked boxes of the dog-park example.

Recorded (they needed a trained model), read at build time from the lecture's own data files, read only:
  /Users/nipun/git/object-detection/data/science.json   YOLOv8n's 31 boxes with score >= 0.25 on the courtyard photo
                                                        (nms_courtyard), the photo's size, the dog and cat boxes
  /Users/nipun/git/object-detection/data/models.json    YOLOv8n's record counts (12,600), the ResNet-50 answer, and the
                                                        small grid CNN's table before training and its loss over 300 steps
  /Users/nipun/git/object-detection/data/dog-park.json  five annotated dogs and eight made-up ranked boxes (teaching example)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'data' / 'p7.json'
SRC = Path('/Users/nipun/git/object-detection/data')
SCI = json.loads((SRC / 'science.json').read_text())
MOD = json.loads((SRC / 'models.json').read_text())
PARK = json.loads((SRC / 'dog-park.json').read_text())
r = lambda v, n=4: round(float(v), n)


def iou_parts(a, b):
    """Intersection, union and IoU of two boxes given by corners (x1, y1, x2, y2)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return {'w': w, 'h': h, 'inter': inter, 'union': union, 'iou': inter / union}


# ---- the photo and its annotation (Lecture 26, data/science.json: photo, dog, cat)
W, H = SCI['photo']['W'], SCI['photo']['H']
DOG, CAT = SCI['dog']['xyxy'], SCI['cat']['xyxy']
assert (W, H) == (960, 640) and DOG == [407, 83, 768, 588] and CAT == [112, 296, 321, 607]

# ---- 7.1: the classifier's one label, and the detector's list (recorded: ResNet-50 and YOLOv8n were run by the lecture)
top = MOD['classifier']['top5'][0]
classifier = {'label': top['label'], 'p': top['p']}
assert classifier['label'] == 'golden retriever' and abs(classifier['p'] - 0.19) < 0.005

# ---- 7.2: IoU of the dog's box and the guess, moved 60 px right and 40 px down
GUESS = [467, 123, 828, 628]
ip = iou_parts(DOG, GUESS)
iou = {'dog': DOG, 'guess': GUESS, 'shift': [GUESS[0] - DOG[0], GUESS[1] - DOG[1]],
       'w': ip['w'], 'h': ip['h'], 'inter': ip['inter'], 'area': (DOG[2] - DOG[0]) * (DOG[3] - DOG[1]),
       'union': ip['union'], 'iou': r(ip['iou']), 'match': 0.5,
       # the live figure's two sliders: [min, max, step] of the shift in pixels, chosen so the box stays in the photo
       'dx': [-400, 190, 10], 'dy': [-80, 50, 10]}
assert iou['inter'] == 139965 and iou['union'] == 224645 and abs(ip['iou'] - 0.623) < 5e-4
assert DOG[0] + iou['dx'][0] >= 0 and DOG[2] + iou['dx'][1] <= W and DOG[1] + iou['dy'][0] >= 0 and DOG[3] + iou['dy'][1] <= H
assert iou_parts(DOG, [DOG[0] + iou['dx'][0], DOG[1], DOG[2] + iou['dx'][0], DOG[3]])['iou'] == 0     # the far left end: apart

# ---- 7.3: one target record per cell of a 3 x 2 grid of 320-pixel cells, and the squared-error loss
S = 320
KEYS = ['object?', 'b_x', 'b_y', 'b_w', 'b_h', 'cat', 'dog']


def build_target(boxes, classes):
    y = np.zeros((H // S, W // S, 7))
    for (x1, y1, x2, y2), k in zip(boxes, classes):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        row, col = int(cy // S), int(cx // S)
        y[row, col] = [1, cx / S - col, cy / S - row, (x2 - x1) / S, (y2 - y1) / S, k == 0, k == 1]
    return y


target = build_target([CAT, DOG], [0, 1])
toy = MOD['toy_yolo']                                       # the lecture's small CNN, trained on this photo alone
pred0 = np.array(toy['snaps']['0']['table'])                # its table before training (recorded)
assert np.allclose(target, np.array(toy['target']), atol=1e-3)
obj = target[..., 0] == 1
cell_loss = (pred0[..., 0] - target[..., 0]) ** 2 + obj * ((pred0[..., 1:] - target[..., 1:]) ** 2).sum(-1)
dog_cell = [int(SCI['dog']['cy'] // S), int(SCI['dog']['cx'] // S)]
sq = (pred0[dog_cell[0], dog_cell[1]] - target[dog_cell[0], dog_cell[1]]) ** 2
assert dog_cell == [1, 1] and abs(cell_loss[1, 1] - 1.35) < 0.01 and abs(cell_loss.sum() - 3.38) < 0.01
assert np.allclose(cell_loss, np.array(toy['snaps']['0']['cell_loss']), atol=2e-3)
grid = {'s': S, 'rows': H // S, 'cols': W // S, 'keys': KEYS, 'dog_cell': dog_cell, 'dog_centre': [SCI['dog']['cx'], SCI['dog']['cy']],
        'target': [r(v, 2) for v in target[1, 1]], 'pred': [r(v, 2) for v in pred0[1, 1]], 'sq': [r(v, 3) for v in sq],
        'cell_loss': [[r(v, 2) for v in row] for row in cell_loss], 'dog_loss': r(cell_loss[1, 1], 2), 'loss': r(cell_loss.sum(), 2),
        'steps': toy['steps'], 'lr': toy['lr'],
        'loss_end': toy['loss'][-1],                       # recorded: the loss after 300 Adam steps on this photo
        'pred_end': [r(v, 2) for v in toy['snaps'][str(toy['steps'])]['table'][1][1]]}
assert abs(grid['loss_end'] - 0.002) < 0.001 and grid['steps'] == 300

# ---- 7.4: score threshold, then class-aware NMS at IoU 0.7, over YOLOv8n's 31 boxes (recorded)
NC = SCI['nms_courtyard']
cands = sorted(NC['candidates'], key=lambda k: -k['score'])
yolo = MOD['yolo']['photos']['cat-dog']
assert yolo['n'] == 12600 and sum(yolo['per_level']) == 12600 and len(cands) == 31 and min(k['score'] for k in cands) >= 0.25


def nms(idx, thr):
    """Keep the best box, delete boxes of its class with IoU above thr, repeat."""
    order, keep = list(idx), []
    while order:
        i, order = order[0], order[1:]
        keep.append(i)
        order = [j for j in order if cands[j]['name'] != cands[i]['name'] or iou_parts(cands[i]['box'], cands[j]['box'])['iou'] <= thr]
    return keep


NMS_IOU = 0.7
cuts = [r(0.25 + 0.05 * i, 2) for i in range(15)]          # the slider: 0.25 to 0.95
by_cut = {}
for c in cuts:
    passed = [i for i, k in enumerate(cands) if k['score'] >= c]
    by_cut[str(c)] = {'pass': passed, 'keep': nms(passed, NMS_IOU)}
k25 = by_cut['0.25']
kept = [cands[i] for i in k25['keep']]
assert len(k25['pass']) == 31 and len(kept) == 4
assert [(k['name'], k['score']) for k in kept] == [(k['name'], k['score']) for k in NC['kept']]
dog_keep = next(k for k in kept if k['name'] == 'dog')
dog_ious = [iou_parts(dog_keep['box'], k['box'])['iou'] for k in cands if k['name'] == 'dog' and k is not dog_keep]
assert len(dog_ious) == 9 and min(dog_ious) > NMS_IOU
det = {'records': yolo['n'], 'per_level': yolo['per_level'], 'iou_thr': NMS_IOU, 'cuts': cuts, 'default': 0.25,
       'boxes': [{'name': k['name'], 'score': k['score'], 'box': k['box']} for k in cands], 'by_cut': by_cut,
       'final': [{'name': k['name'], 'score': k['score'], 'box': k['box']} for k in kept],
       'dog_iou_min': r(min(dog_ious), 3), 'n_dog': 1 + len(dog_ious)}

# ---- the check: the dog's box and the 60-px guess, both reported as dog boxes
check = {'iou': iou['iou'], 'nms': NMS_IOU, 'match': 0.5,
         'score_a': 0.89,      # made up for the question: the box on the dog, scored like YOLOv8n's dog box
         'score_b': 0.80}      # made up for the question: the 60-px guess
check['survive'] = 2 if check['iou'] <= check['nms'] else 1
check['precision'] = r(1 / check['survive'], 2)
assert check['survive'] == 2 and check['precision'] == 0.5

# ---- 7.5: precision, recall and AP on the dog park (eight ranked boxes, five dogs; made up for teaching by the lecture)
gt = [g[2] for g in PARK['gt']]
dets = sorted(PARK['dets'], key=lambda d: -d[0])
found, hit, best = set(), [], []
for s_, box, _ in dets:
    ious = [iou_parts(box, g)['iou'] for g in gt]
    j = int(np.argmax(ious))
    ok = ious[j] >= 0.5 and j not in found
    if ok:
        found.add(j)
    hit.append(ok)
    best.append(r(ious[j], 2))
tp = np.cumsum(hit)
fp = np.cumsum([not h for h in hit])
prec, rec = tp / (tp + fp), tp / len(gt)
env = [max(prec[i:]) for i in range(len(prec))]
ap = sum((rec[i] - (rec[i - 1] if i else 0)) * env[i] for i in range(len(prec)))
assert hit == [True, True, False, True, False, False, True, False]
assert (prec[-1], rec[-1]) == (0.5, 0.8) and abs(ap - 0.664) < 1e-3
assert (prec[1], rec[1]) == (1, 0.4) and (prec[4], rec[4]) == (0.6, 0.6)        # the lecture's cuts at 0.85 and 0.5
pr = {'n_gt': len(gt), 'scores': [d[0] for d in dets], 'what': [d[2] for d in dets], 'hit': hit, 'best_iou': best,
      'tp': tp.tolist(), 'fp': fp.tolist(), 'fn': [len(gt) - int(t) for t in tp],
      'precision': [r(v, 3) for v in prec], 'recall': [r(v, 3) for v in rec], 'envelope': [r(v, 3) for v in env],
      'ap': r(ap, 3), 'match': 0.5}

OUT.write_text(json.dumps({'source': 'scripts/sci/p7.py', 'photo': {'W': W, 'H': H, 'dog': DOG, 'cat': CAT},
                           'classifier': classifier, 'iou': iou, 'grid': grid, 'det': det, 'check': check, 'pr': pr}, indent=1) + '\n')
print('IoU', iou['inter'], iou['union'], iou['iou'])
print('grid: dog cell', grid['dog_loss'], 'all', grid['loss'], 'after', grid['steps'], grid['loss_end'])
print('NMS:', {c: (len(v['pass']), len(v['keep'])) for c, v in by_cut.items()})
print('PR:', pr['precision'], pr['recall'], 'AP', pr['ap'])
