"""The numbers of Part 5: vision and language (Lectures 16 to 22).

    ~/base/bin/python scripts/sci/p5.py      writes data/p5.json

What is recomputed here, and checked against the lectures:
  - the ViT's token arithmetic for patch sizes 32, 16 and 8 (Lecture 16), and where the model's 224 x 224 crop sits in
    the Newfoundland photo (from the checkpoint's preprocessing);
  - CLIP's toy batch (Lecture 17): the cosine matrix from the six authored 3-D vectors, its row and column softmax at
    the lecture's temperature and the symmetric loss. The part file recomputes the softmax and the loss for other
    temperatures from the matrix stored here;
  - the VLM's position counts and the mean answer loss on "A black dog." (Lectures 18 and 19);
  - the zebra, horse and giraffe worksheet of the concept bottleneck lecture (Lecture 20), from its weights.
What is recorded (a trained model or a dataset run; no training and no downloads here): the ViT's prediction, CLIP's
measured cosines for the photo, the toy CLIP loss after 100 steps, the prompt's token count, and the CUB accuracies.
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'data' / 'p5.json'
CBM_DATA = Path('/Users/nipun/git/cbm/src/data.json')      # read only, to check the constants below against the lecture
r = lambda v, n=4: round(float(v), n)


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


# ---------------------------------------------------------------------------------------- Lecture 16: the ViT
IMG = 224                         # the checkpoint's input: 3 x 224 x 224
PHOTO = (500, 334)                # figures/p5-newfoundland.jpg, newfoundland_31 of Oxford-IIIT Pet
CROP_PCT = 0.9                    # timm preprocessing of vit_tiny_patch16_224.augreg_in21k_ft_in1k (real-inference.json)
# timm resizes the short side to floor(224 / 0.9) = 248, then takes the centre 224 x 224. In the photo's own pixels:
short = math.floor(IMG / CROP_PCT)
scale = PHOTO[1] / short
resized_w = round(PHOTO[0] / scale)
left, top = round((resized_w - IMG) / 2), round((short - IMG) / 2)
crop = [r(left * PHOTO[0] / resized_w, 2), r(top * scale, 2), r(IMG * scale, 2)]   # x, y, side of the square, in photo pixels
assert (short, resized_w, left, top) == (248, 371, 74, 12)     # checked pixel for pixel against the lecture's model-input.png

D_MODEL, HEADS, BLOCKS, CLASSES, CHANNELS = 192, 3, 12, 1000, 3     # ViT-Tiny, as Lecture 16 used it
sizes = {}
for P in (32, 16, 8):
    side = IMG // P
    n = side * side
    sizes[str(P)] = {'side': side, 'tokens': n, 'with_cls': n + 1, 'scores': (n + 1) ** 2, 'numbers': P * P * CHANNELS}
assert sizes['16']['tokens'] == 196 and sizes['16']['numbers'] == 768 and sizes['16']['with_cls'] == 197
assert sizes['32']['tokens'] == 49 and sizes['8']['tokens'] == 784
assert sizes['16']['scores'] == 38809 and sizes['8']['scores'] == 616225
# Recorded: the checkpoint's top three on the photo (attention repo, figures/vision1/real-inference.json).
TOP3 = [['Newfoundland', 0.9572675228118896], ['Tibetan mastiff', 0.01625724323093891], ['briard', 0.006742686033248901]]
vit = {'img': IMG, 'patch': 16, 'patches': [32, 16, 8], 'sizes': sizes, 'd': D_MODEL, 'heads': HEADS, 'd_head': D_MODEL // HEADS,
       'blocks': BLOCKS, 'classes': CLASSES, 'channels': CHANNELS, 'pixels': IMG * IMG, 'photo': list(PHOTO), 'crop': crop,
       'top3': [[k, r(v, 6)] for k, v in TOP3],
       'follow_px': [6 * 16 + 8, 4 * 16 + 8]}        # the centre of P63, the patch Lecture 16 followed (row 4, column 6 from zero)
assert vit['d_head'] == 64 and abs(TOP3[0][1] - 0.9573) < 1e-4

# ---------------------------------------------------------------------------------------- Lecture 17: CLIP
# The authored batch of the lecture: three photos and their captions, after each encoder's projection (3-D).
NAMES = ['cat', 'dog', 'car']
IMAGES = [[2.0, 0.5, 0.2], [0.4, 2.0, 0.1], [0.2, 0.3, 2.0]]
TEXTS = [[1.6, 0.8, 0.4], [0.7, 1.4, 0.3], [0.4, 0.5, 1.5]]
TAU = 0.5
unit = lambda v: [a / math.sqrt(sum(b * b for b in v)) for a in v]
U, V = [unit(v) for v in IMAGES], [unit(v) for v in TEXTS]
C = [[sum(a * b for a, b in zip(u, v)) for v in V] for u in U]


def clip_loss(C, tau):
    S = [[c / tau for c in row] for row in C]
    P = [softmax(row) for row in S]                                   # each photo: which caption?
    cols = [softmax([S[i][j] for i in range(len(S))]) for j in range(len(S))]
    Q = [[cols[j][i] for j in range(len(S))] for i in range(len(S))]  # each caption: which photo? (rows are still photos)
    li = -sum(math.log(P[i][i]) for i in range(len(S))) / len(S)
    lt = -sum(math.log(Q[i][i]) for i in range(len(S))) / len(S)
    return S, P, Q, li, lt, (li + lt) / 2


S, P, Q, LI, LT, LOSS = clip_loss(C, TAU)
lecture_C = [[0.969, 0.654, 0.400], [0.609, 0.956, 0.393], [0.365, 0.358, 0.975]]
assert all(abs(C[i][j] - lecture_C[i][j]) < 6e-4 for i in range(3) for j in range(3))
assert abs(C[0][0] - 0.969281) < 1e-6
assert all(abs(a - b) < 1e-4 for a, b in zip(P[0], [0.5398, 0.2875, 0.1727])) and abs(Q[0][0] - 0.560141) < 1e-6
assert abs(LI - 0.560020) < 1e-6 and abs(LT - 0.561033) < 1e-6 and abs(LOSS - 0.560526) < 1e-6
TAU_RANGE = [0.1, 2.0, 0.05]      # the slider: min, max, step (Lecture 17's slider ran on 1 / tau from 0.1 to 10)
sweep = {str(t): r(clip_loss(C, t)[5]) for t in (0.1, 0.25, 0.5, 1.0, 2.0)}
clip = {'names': NAMES, 'C': [[r(c, 6) for c in row] for row in C], 'tau': TAU, 'tau_range': TAU_RANGE,
        'P': [[r(v) for v in row] for row in P], 'Q': [[r(v) for v in row] for row in Q],
        'pen_rows': [r(-math.log(P[i][i])) for i in range(3)], 'loss_rows': r(LI, 6), 'loss_cols': r(LT, 6), 'loss': r(LOSS, 6),
        'ln_n': r(math.log(len(NAMES)), 6), 'n': len(NAMES), 'dim': len(IMAGES[0]), 'sweep': sweep}
# Recorded: the lecture's toy run on the same batch (clip/index.html, closing-summary-losses): one SGD step, 100 steps.
clip['recorded'] = {'one_step': 0.539743, 'eta': 0.25, 'steps': 100, 'after_steps': 0.124937}
# Recorded: frozen CLIP ViT-B/32 on the photo, raw cosines (clip/index.html, mot-new-labels-result, mot-wrong-menu-result).
clip['menu'] = [['a photo of a dog', 0.2723], ['a photo of a cat', 0.2152], ['a photo of a bird', 0.2184]]
clip['wrong_menu'] = [['a photo of a cat', 0.2152], ['a photo of a bird', 0.2184], ['a photo of a car', 0.2179]]
assert max(clip['menu'], key=lambda x: x[1])[0].endswith('dog') and max(clip['wrong_menu'], key=lambda x: x[1])[0].endswith('bird')

# ---------------------------------------------------------------------------------------- Lectures 18 and 19: the VLM
PROMPT_TOKENS = 13      # recorded: "Question: What animal is shown?\nAnswer: It is a" in DistilGPT2's tokenizer (cost-full)
SLOTS, D_LM, VOCAB = 8, 768, 50257
rows = sizes['16']['tokens']                     # CLS is dropped: every patch row goes to the language model
ANSWER = [' A', ' black', ' dog', '.', 'end']   # the reference "A black dog." and the end-of-text token
PROBS = [0.20, 0.40, 0.50, 0.80, 0.90]           # authored in Lecture 19 (teacher-loss)
pen = [-math.log(p) for p in PROBS]
vlm = {'rows': rows, 'd_v': D_MODEL, 'd_lm': D_LM, 'vocab': VOCAB, 'prompt': PROMPT_TOKENS, 'slots': SLOTS,
       'full': rows + PROMPT_TOKENS, 'short': SLOTS + PROMPT_TOKENS, 'saved': rows - SLOTS,
       'answer': ANSWER, 'probs': PROBS, 'pen': [r(v) for v in pen], 'total': r(sum(pen)), 'mean': r(sum(pen) / len(pen), 6)}
assert vlm['full'] == 209 and vlm['short'] == 21 and vlm['saved'] == 188
assert abs(sum(pen) / len(pen) - 0.7094759783680473) < 1e-12

# ---------------------------------------------------------------------------------------- Lecture 20: the concept bottleneck
CLASSES3 = ['zebra', 'horse', 'giraffe']
CONCEPTS = ['stripes', 'horse-like shape', 'long neck']
W = [[3.0, 1.0, 0.0], [-1.0, 2.0, -1.0], [-1.0, 0.0, 3.0]]     # the head: one row per class (cbm, src/data.json, "zoo")
c_hat = [0.1, 0.9, 0.1]                                      # the predicted concepts
c_fix = [1.0, 0.9, 0.1]                                      # a person corrects stripes to 1
s0 = [sum(w * c for w, c in zip(row, c_hat)) for row in W]
s1 = [sum(w * c for w, c in zip(row, c_fix)) for row in W]
p0, p1 = softmax(s0), softmax(s1)
assert all(abs(a - b) < 5e-4 for a, b in zip(p0, [0.350, 0.522, 0.129])) and abs(p1[0] - 0.95) < 5e-3
assert all(abs(a - b) < 1e-9 for a, b in zip(s0, [1.2, 1.6, 0.2])) and all(abs(a - b) < 1e-9 for a, b in zip(s1, [3.9, 0.7, -0.7]))
cbm = {'classes': CLASSES3, 'concepts': CONCEPTS, 'W': W, 'c': c_hat, 'c_fix': c_fix, 'edit': 0,
       's': [r(v) for v in s0], 's_fix': [r(v) for v in s1], 'p': [r(v) for v in p0], 'p_fix': [r(v) for v in p1],
       'delta': [r(v) for v in (a - b for a, b in zip(s1, s0))],
       'concept_term': {'label': 1, 'pred': 0.8, 'loss': r(-math.log(0.8))}}           # one concept-loss term, s05.4

# Recorded: CUB, frozen CLIP ViT-B/32 features, 112 concepts in 28 groups, 200 species; one run
# (cbm, src/data.json "cub"; slides s23.4 and s23.6). Accuracy after correcting the first k concept groups, k = 0 to 28.
CUB_ACC = {'standard': 0.7327, 'joint': 0.7346, 'sequential': 0.6886, 'independent': 0.4993}
CUB_CURVES = {
    'independent': [0.4993, 0.5667, 0.6353, 0.6874, 0.7317, 0.7888, 0.8306, 0.8701, 0.8964, 0.9171, 0.9381, 0.9456, 0.9552, 0.962, 0.9675,
                    0.975, 0.9787, 0.9818, 0.9853, 0.9879, 0.9924, 0.9939, 0.9952, 0.996, 0.9967, 0.9974, 0.9983, 0.9993, 1.0],
    'sequential': [0.6886, 0.7299, 0.7653, 0.7938, 0.8186, 0.8505, 0.8744, 0.8947, 0.9098, 0.9218, 0.9331, 0.9377, 0.9449, 0.9516, 0.9567,
                   0.9618, 0.9658, 0.9688, 0.9742, 0.9755, 0.9808, 0.9822, 0.9828, 0.9828, 0.9833, 0.9829, 0.9839, 0.9846, 0.9846],
    'joint': [0.7346, 0.7499, 0.7614, 0.7761, 0.7851, 0.7986, 0.8101, 0.8253, 0.8365, 0.8438, 0.8525, 0.8559, 0.8647, 0.8696, 0.8758,
              0.8807, 0.8853, 0.889, 0.8973, 0.9001, 0.9047, 0.9096, 0.9133, 0.9091, 0.9095, 0.9076, 0.9072, 0.9085, 0.902]}
cub = {'images': 11788, 'species': 200, 'concepts': 112, 'groups': 28, 'acc': CUB_ACC, 'curves': CUB_CURVES}
assert all(len(v) == cub['groups'] + 1 for v in CUB_CURVES.values())
assert [round(100 * CUB_ACC[k], 1) for k in ('standard', 'joint', 'sequential', 'independent')] == [73.3, 73.5, 68.9, 49.9]
assert [round(100 * CUB_CURVES[k][-1], 1) for k in ('independent', 'sequential', 'joint')] == [100.0, 98.5, 90.2]
if CBM_DATA.exists():            # the lecture's own data file, read only: the constants above must match it
    lec = json.loads(CBM_DATA.read_text())
    assert lec['zoo']['W'] == W and lec['zoo']['c'] == c_hat and lec['zoo']['c_new'] == c_fix
    assert all(lec['cub']['acc'][k] == v for k, v in CUB_ACC.items())
    assert all(lec['cub']['intervention'][k] == v for k, v in CUB_CURVES.items())
    assert (lec['cub']['n_images'], lec['cub']['n_concepts'], lec['cub']['n_groups']) == (11788, 112, 28)

OUT.write_text(json.dumps({'source': 'scripts/sci/p5.py', 'vit': vit, 'clip': clip, 'vlm': vlm, 'cbm': cbm, 'cub': cub}, indent=1) + '\n')
print('crop', crop, 'sizes', sizes)
print('C', clip['C'], '\nP', clip['P'], '\nloss', clip['loss_rows'], clip['loss_cols'], clip['loss'], 'sweep', sweep)
print('vlm', vlm['full'], vlm['short'], vlm['pen'], vlm['mean'])
print('cbm', cbm['s'], cbm['p'], cbm['s_fix'], cbm['p_fix'], cbm['delta'])
