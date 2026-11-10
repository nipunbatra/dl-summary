"""The numbers of Part 3 (generalization, Lecture 10): the seeded ten-point polynomial lab, the four regularisers
on their worked examples, and the lecture's CIFAR-10 ablation.

    ~/base/bin/python scripts/sci/p3.py      writes data/p3.json

The polynomial lab is a port of lecture7/L7-regularization.typ on origin/master of dl-teaching (the schedule's
Lecture 10). That deck draws its noise with the chalkdust `rand` package: Threefry-2x32 with 20 rounds, keyed by
the seed, then Box-Muller. The port below reproduces it bit for bit, so the fits, the lambda sweep and the
early-stopping checkpoints match the lecture's numbers (asserted at the end).

    truth f(x) = sin(pi x), noise sd 0.25, seed 21
    train: 10 inputs evenly spaced on [-1, 1], noise stream seed 21
    validation: the 9 midpoints, noise stream seed 121
    test: 101 inputs evenly spaced on [-1, 1], noise stream seed 321
    features (1, x, ..., x^9); ridge J(w) = MSE/2 + (lambda/2)|w|^2, so (X'X + n lambda I) w = X'y

The CIFAR-10 accuracies are a recorded result of lecture7/diagrams/cifar_regularization_experiment.py (one seed);
they are not retrained here.
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'data' / 'p3.json'

# ---------------------------------------------------------------------------------------- the lecture's PRNG
M32 = 2 ** 32
ROT = (13, 15, 26, 6, 17, 29, 16, 24)


def rotl(x, k):
    k %= 32
    return ((x << k) | (x >> (32 - k))) % M32 if k else x


def threefry2x32(c0, c1, k0, k1):
    ks = (k0, k1, 466688986 ^ k0 ^ k1)
    x0, x1 = (c0 + k0) % M32, (c1 + k1) % M32
    for r in range(20):
        x0 = (x0 + x1) % M32
        x1 = rotl(x1, ROT[r % 8]) ^ x0
        if r % 4 == 3:
            j = r // 4 + 1
            x0 = (x0 + ks[j % 3]) % M32
            x1 = (x1 + ks[(j + 1) % 3] + j) % M32
    return x0, x1


def rand(seed, i):
    return threefry2x32(i, 0, seed, 0)[0] / M32


def randn(seed, k):                     # Box-Muller, as chalkdust rand.randn
    u1, u2 = max(rand(seed, 2 * k), 1e-9), rand(seed, 2 * k + 1)
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


# ---------------------------------------------------------------------------------------- the ten-point lab
SEED, NTR, DEG, NOISE = 21, 10, 9, 0.25
truth = lambda x: np.sin(np.pi * x)
xs = np.array([-1 + 2 * i / (NTR - 1) for i in range(NTR)])
ys = truth(xs) + NOISE * np.array([randn(SEED, i) for i in range(NTR)])
xv = (xs[:-1] + xs[1:]) / 2
yv = truth(xv) + NOISE * np.array([randn(SEED + 100, i) for i in range(NTR - 1)])
xt = np.array([-1 + 2 * i / 100 for i in range(101)])
yt = truth(xt) + NOISE * np.array([randn(SEED + 300, i) for i in range(101)])

V = lambda x: np.vander(np.atleast_1d(x), DEG + 1, increasing=True)
mse = lambda w, x, y: float(np.mean((V(x) @ w - y) ** 2))
A = V(xs)
w9 = np.linalg.solve(A, ys)                                            # the interpolant: 10 equations, 10 weights
ridge = lambda lam: np.linalg.solve(A.T @ A + NTR * lam * np.eye(DEG + 1), A.T @ ys)

GRID = np.linspace(-1, 1, 121)                                         # where the slides draw the curves
r = lambda v, n=4: round(float(v), n)
rl = lambda a, n=4: [r(v, n) for v in a]
curve = lambda w: rl(V(GRID) @ w, 3)


def row(lam, w):
    return {'lam': lam, 'train': r(mse(w, xs, ys)), 'val': r(mse(w, xv, yv)), 'test': r(mse(w, xt, yt)),
            'norm': r(np.linalg.norm(w), 1), 'curve': curve(w)}


LAMS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]                     # the lecture's lam-grid
sweep = [row(0, w9)] + [row(l, ridge(l)) for l in LAMS]                # lambda = 0 is the interpolant
best = min(range(1, len(sweep)), key=lambda i: sweep[i]['val'])

# early stopping: gradient descent from w = 0 on MSE/2, eta = 0.25 / largest eigenvalue, in closed form
G = A.T @ A / NTR
ev, U = np.linalg.eigh(G)
ETA_GD = 0.25 / ev.max()
coef = U.T @ w9
wt = lambda t: U @ ((1 - (1 - ETA_GD * ev) ** t) * coef)
CKPT = [1, 3, 10, 30, 100, 300, 1000, 2000, 5000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000]
early = [{'t': t, 'train': r(mse(wt(t), xs, ys)), 'val': r(mse(wt(t), xv, yv)), 'test': r(mse(wt(t), xt, yt))} for t in CKPT]
t_best = min(range(len(early)), key=lambda i: early[i]['val'])

poly = {
    'seed': SEED, 'n_train': NTR, 'n_val': NTR - 1, 'n_test': len(xt), 'degree': DEG, 'n_weights': DEG + 1,
    'noise': NOISE, 'sigma2': NOISE ** 2,
    'x_train': rl(xs), 'y_train': rl(ys), 'x_val': rl(xv), 'y_val': rl(yv),
    'grid': rl(GRID), 'truth': rl(truth(GRID), 3),
    'interp': {'train': r(mse(w9, xs, ys), 6), 'val': r(mse(w9, xv, yv)), 'test': r(mse(w9, xt, yt)),
               'norm': r(np.linalg.norm(w9), 1), 'curve': curve(w9),
               'times_noise': r(mse(w9, xv, yv) / NOISE ** 2, 1)},
}

# ---------------------------------------------------------------------------------------- the four regularisers
th, g, eta, lam = 5.0, 3.0, 0.1, 0.2                                     # one L2-SGD step (the lecture's numbers)
wd = {'theta': th, 'g': g, 'eta': eta, 'lam': lam, 'shrink': r(1 - eta * lam), 'shrunk': r((1 - eta * lam) * th),
      'fit': r(eta * g), 'new': r((1 - eta * lam) * th - eta * g)}

h, mask, keep = [2, 4, 6, 8], [1, 0, 1, 0], 0.5                        # inverted dropout
dropout = {'h': h, 'mask': mask, 'keep': keep, 'masked': [a * b for a, b in zip(h, mask)],
           'out': [r(a * b / keep, 2) for a, b in zip(h, mask)], 'eval': h,
           'mean3': r(keep * h[2] / keep + (1 - keep) * 0, 2)}         # E[h~_3] = h_3

K, EPS, Y = 5, 0.1, 1                                                  # label smoothing: classes a to e, the right one is b
target = [r((1 - EPS) * (k == Y) + EPS / K) for k in range(K)]
smooth = {'K': K, 'eps': EPS, 'classes': list('abcde'), 'y': Y, 'target': target,
          'gap': r(math.log(target[Y] / target[0]), 2)}                # the logit gap where the push stops

SIZE, PAD = 32, 4                                                      # v2.RandomCrop(32, padding=4) + RandomHorizontalFlip
augment = {'size': SIZE, 'pad': PAD, 'positions': (2 * PAD + 1) ** 2, 'flips': 2, 'views': (2 * PAD + 1) ** 2 * 2}

# ---------------------------------------------------------------------------------------- recorded: CIFAR-10
# lecture7/L7-regularization.typ, slides "Controlled CIFAR-10 ablations" and lecture7/figures/cifar_regularization_summary.svg,
# made by lecture7/diagrams/cifar_regularization_experiment.py: test accuracy (%) at the validation-chosen checkpoint, one seed.
cifar = {'n_train': 2000, 'n_val': 5000, 'n_test': 10000, 'params': 666538, 'epochs': 35, 'wd_lam': 0.01, 'eps': 0.1,
         'runs': [{'name': 'baseline', 'acc': 43.0}, {'name': 'weight decay', 'acc': 43.0}, {'name': 'dropout', 'acc': 46.8},
                  {'name': 'label smoothing', 'acc': 44.8}, {'name': 'crop and flip', 'acc': 54.6}]}
cifar['best'] = max(range(len(cifar['runs'])), key=lambda i: cifar['runs'][i]['acc'])
cifar['gain'] = r(cifar['runs'][cifar['best']]['acc'] - cifar['runs'][0]['acc'], 1)

out = {'source': 'scripts/sci/p3.py', 'poly': poly, 'sweep': sweep, 'best': best,
       'early': {'eta': r(ETA_GD, 6), 'rows': early, 'best': t_best}, 'wd': wd, 'dropout': dropout, 'smooth': smooth,
       'augment': augment, 'recorded': {'cifar': cifar}}

# ---------------------------------------------------------------------------------------- the lecture's numbers
I = poly['interp']
assert I['train'] < 1e-6 and abs(I['val'] - 1.70) < 5e-3 and round(I['norm']) == 465 and round(I['times_noise']) == 27
assert abs(I['test'] - 1.187) < 5e-4
lecture_sweep = {1e-6: (0.014, 0.151, 35.1), 1e-4: (0.018, 0.077, 5.6), 1e-3: (0.023, 0.065, 3.9),
                 1e-2: (0.062, 0.111, 1.8), 1.0: (0.286, 0.405, 0.2)}
for s in sweep[1:]:
    if s['lam'] in lecture_sweep:
        a, b, c = lecture_sweep[s['lam']]
        assert abs(s['train'] - a) < 5e-4 and abs(s['val'] - b) < 5e-4 and abs(s['norm'] - c) < 0.05, s['lam']
assert sweep[best]['lam'] == 1e-3 and abs(sweep[best]['test'] - 0.124) < 5e-4
E = early[t_best]
assert E['t'] == 5000 and abs(E['train'] - 0.020) < 5e-4 and abs(E['val'] - 0.061) < 5e-4 and abs(E['test'] - 0.125) < 5e-4
assert wd['shrunk'] == 4.9 and wd['new'] == 4.6 and dropout['out'] == [4, 0, 12, 0] and dropout['mean3'] == 6
assert target == [0.02, 0.92, 0.02, 0.02, 0.02] and abs(smooth['gap'] - 3.83) < 5e-3
assert [c['acc'] for c in cifar['runs']] == [43.0, 43.0, 46.8, 44.8, 54.6] and cifar['best'] == 4

OUT.write_text(json.dumps(out, indent=1) + '\n')
print('wrote', OUT)
print('interpolant: train %.6f  val %.4f  norm %.1f  test %.4f' % (I['train'], I['val'], I['norm'], I['test']))
for s in sweep:
    print('lambda %-7g train %.3f val %.3f test %.3f norm %.1f  curve %.2f..%.2f' % (s['lam'], s['train'], s['val'], s['test'], s['norm'], min(s['curve']), max(s['curve'])))
print('early stop t*', E)
