"""The numbers of Part 2 (Lectures 5 to 9): gradients and optimization.

    /Users/nipun/git/clip-lab/.venv/bin/python scripts/sci/p2.py      writes data/p2.json

Every number is recomputed with the lecture's own example and settings, and checked against the value the lecture
printed (plan/digests/02-gradients-optimization-generalization.md):

- bp:       Lecture 6's graph y_hat = w x + b, L = (y_hat - y)^2 at (w, x, b, y) = (2, 3, 1, 10), and one step of 0.01.
- sgd:      Lecture 5's Example A, the four-point line fit from theta_0 = (b, w) = (2, -1) with eta = 0.1: the
            full-batch step and the four one-point steps, with the contours of the full loss for the figure.
- opt:      Lecture 7's Example B ravine, L = 1/2 (t1 - 1)^2 + 50 (t2 + 0.5)^2 from (-1.4, 0.35), 55 updates of
            gradient descent, momentum, RMSProp and Adam at the lecture's method-specific settings.
- vjp:      Lecture 8's J = [[1, 1], [2, -1]] and upstream gradient (3, 4), run through autograd.
- softmax:  Lecture 9's logits (2, 1, 0) and (1000, 999, 998) in float32, directly and after subtracting the max.
- sched:    Lecture 9's cosine schedule, eta_0 = 0.1, eta_min = 0.001, T = 100, and the bounce of a fixed noise.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parents[2] / 'data' / 'p2.json'
rd = lambda v, n=4: round(float(v), n)
rl = lambda a, n=4: [rd(v, n) for v in np.asarray(a, dtype=float).ravel()]
close = lambda a, b, tol=1e-3: abs(float(a) - float(b)) < tol

# ---------------------------------------------------------------------------------------- Lecture 6: one graph, backward
torch.set_default_dtype(torch.float64)
w, b = torch.tensor(2., requires_grad=True), torch.tensor(1., requires_grad=True)
x, y = 3., 10.
yhat = w * x + b
yhat.retain_grad()
L = (yhat - y) ** 2
L.backward()
ETA_BP = 0.01
w1, b1 = w.item() - ETA_BP * w.grad.item(), b.item() - ETA_BP * b.grad.item()
yhat1 = w1 * x + b1
ETA_BIG = 0.1                                   # ten times the lecture's step: it overshoots the target
w10, b10 = w.item() - ETA_BIG * w.grad.item(), b.item() - ETA_BIG * b.grad.item()
bp = {'w': 2, 'x': 3, 'b': 1, 'y': 10, 'm': rd(w.item() * x), 'yhat': rd(yhat.item()), 'loss': rd(L.item()),
      'g_yhat': rd(yhat.grad), 'g_b': rd(b.grad), 'g_w': rd(w.grad), 'g_x': rd(w.item() * yhat.grad.item()),
      'eta': ETA_BP, 'after': {'w': rd(w1), 'b': rd(b1), 'yhat': rd(yhat1), 'loss': rd((yhat1 - y) ** 2)},
      'big': {'eta': ETA_BIG, 'w': rd(w10), 'b': rd(b10), 'yhat': rd(w10 * x + b10), 'loss': rd((w10 * x + b10 - y) ** 2)}}
assert (bp['yhat'], bp['loss'], bp['g_yhat'], bp['g_b'], bp['g_w']) == (7, 9, -6, -6, -18)
assert close(bp['after']['w'], 2.18) and close(bp['after']['b'], 1.06)
assert close(bp['after']['yhat'], 7.60) and close(bp['after']['loss'], 5.76) and bp['g_x'] == -12
assert close(bp['big']['yhat'], 13) and close(bp['big']['loss'], bp['loss'])     # past the target, and back to the same loss

# ---------------------------------------------------------------------------------------- Lecture 5: Example A, SGD
XA = np.array([-1., 0., 1., 2.])
YA = np.array([-1., 1., 3., 6.])
TH0 = np.array([2., -1.])                      # (b, w)
ETA_A = 0.1


def lossA(th, idx=range(4)):
    r = th[0] + th[1] * XA[list(idx)] - YA[list(idx)]
    return float(np.mean(0.5 * r ** 2))


def gradA(th, idx):
    idx = list(idx)
    r = th[0] + th[1] * XA[idx] - YA[idx]
    return np.array([np.mean(r), np.mean(r * XA[idx])])


per_point = [gradA(TH0, [i]) for i in range(4)]
assert np.allclose(per_point, [[4, -4], [1, 0], [-2, -2], [-6, -12]])
choices = {}
for key, idx in [('all', range(4))] + [(str(i + 1), [i]) for i in range(4)]:
    g = gradA(TH0, idx)
    th1 = TH0 - ETA_A * g
    choices[key] = {'points': [i + 1 for i in idx], 'grad': rl(g), 'theta1': rl(th1),
                    'loss1': rd(lossA(th1)),                                        # the full loss after the step
                    'own0': rd(lossA(TH0, idx)), 'own1': rd(lossA(th1, idx))}      # the loss on the points used
# the least-squares optimum and the curvature of the full loss (it is quadratic in (b, w))
A = np.stack([np.ones(4), XA], 1)
th_star = np.linalg.lstsq(A, YA, rcond=None)[0]
H = A.T @ A / 4
L_min = lossA(th_star)


def ellipse(level, n=180):
    """The contour L(theta) = level of the quadratic loss, as a closed polyline in (b, w)."""
    vals, vecs = np.linalg.eigh(H)
    t = np.linspace(0, 2 * math.pi, n + 1)
    circ = np.stack([np.cos(t), np.sin(t)])
    pts = th_star[:, None] + vecs @ (np.sqrt(2 * (level - L_min) / vals)[:, None] * circ)
    return [[rd(a, 3), rd(c, 3)] for a, c in pts.T]


LEVELS_A = [2.0, 3.0, 4.0, 5.0, 6.0, 7.125, 8.5, 10.0, 12.0]
sgd = {'x': XA.tolist(), 'y': YA.tolist(), 'theta0': TH0.tolist(), 'eta': ETA_A, 'loss0': rd(lossA(TH0)),
       'per_point': [rl(g) for g in per_point], 'choices': choices, 'order': ['all', '1', '2', '3', '4'],
       'theta_star': rl(th_star), 'L_min': rd(L_min), 'levels': LEVELS_A,
       'contours': {str(c): ellipse(c) for c in LEVELS_A},
       'mean_single_step': rl(np.mean([-ETA_A * g for g in per_point], 0)),
       'full_step': rl(-ETA_A * gradA(TH0, range(4)))}
assert close(sgd['loss0'], 7.125, 1e-9)
assert np.allclose(choices['all']['grad'], [-0.75, -4.5]) and np.allclose(choices['all']['theta1'], [2.075, -0.55])
assert close(choices['all']['loss1'], 5.215)
assert np.allclose(choices['2']['theta1'], [1.9, -1]) and close(choices['2']['loss1'], 7.205)
assert close(choices['2']['own0'], 0.5) and close(choices['2']['own1'], 0.405)
assert np.allclose(th_star, [1.1, 2.3]) and np.allclose(sgd['mean_single_step'], sgd['full_step'])
assert np.allclose(sgd['full_step'], [0.075, 0.45])

# ---------------------------------------------------------------------------------------- Lecture 7: the ravine
TH0B = np.array([-1.4, 0.35])
STEPS = 55
EPS = 1e-8
lossB = lambda t: 0.5 * (t[0] - 1) ** 2 + 50 * (t[1] + 0.5) ** 2
gradB = lambda t: np.array([t[0] - 1, 100 * (t[1] + 0.5)])


def run(method, eta, b1=None, b2=None):
    th, m, v, path = TH0B.copy(), np.zeros(2), np.zeros(2), [TH0B.copy()]
    for t in range(STEPS):
        g = gradB(th)
        if method == 'gd':
            th = th - eta * g
        elif method == 'momentum':                                  # the lecture's EWMA form
            m = b1 * m + (1 - b1) * g
            th = th - eta * m
        elif method == 'rmsprop':
            v = b2 * v + (1 - b2) * g ** 2
            th = th - eta * g / (np.sqrt(v) + EPS)
        elif method == 'adam':
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g ** 2
            mh, vh = m / (1 - b1 ** (t + 1)), v / (1 - b2 ** (t + 1))
            th = th - eta * mh / (np.sqrt(vh) + EPS)
        path.append(th.copy())
    return np.array(path)


METHODS = [('gd', 'gradient descent', {'eta': 0.0195}),
           ('momentum', 'momentum', {'eta': 0.0195, 'b1': 0.25}),
           ('rmsprop', 'RMSProp', {'eta': 0.08, 'b2': 0.9}),
           ('adam', 'Adam', {'eta': 0.10, 'b1': 0.9, 'b2': 0.999})]
paths = {k: run(k, **s) for k, _, s in METHODS}
loss_paths = {k: [lossB(t) for t in p] for k, p in paths.items()}
assert close(lossB(TH0B), 39.005)
assert close(loss_paths['gd'][1], 35.372) and close(loss_paths['momentum'][1], 10.524)
assert np.allclose(paths['momentum'][1], [-1.3649, -0.893125], atol=1e-5)
assert close(loss_paths['momentum'][6], 2.303) and close(loss_paths['gd'][6], 21.794)
assert close(loss_paths['momentum'][55], 0.330) and close(loss_paths['gd'][55], 0.458)
assert np.allclose(paths['rmsprop'][1], [-1.147, 0.097], atol=1e-3) and np.allclose(paths['rmsprop'][2], [-0.973, -0.054], atol=1e-3)
assert close(loss_paths['rmsprop'][2], 11.915)
assert np.allclose(paths['adam'][1], [-1.30, 0.25], atol=1e-3) and np.allclose(paths['adam'][2], [-1.200, 0.151], atol=1e-3)
assert np.allclose(paths['gd'][1], [-1.3532, -1.3075], atol=1e-4)               # one correct step crosses the valley


def ravine_contour(level, n=240):
    t = np.linspace(0, 2 * math.pi, n + 1)
    return [[rd(1 + math.sqrt(2 * level) * math.cos(a), 3), rd(-0.5 + math.sqrt(level / 50) * math.sin(a), 3)] for a in t]


LEVELS_B = [0.5, 2, 5, 10, 20, 39.005]
opt = {'theta0': TH0B.tolist(), 'theta_star': [1, -0.5], 'steps': STEPS, 'loss0': rd(lossB(TH0B), 3),
       'methods': [{'key': k, 'name': n, **s} for k, n, s in METHODS],
       'paths': {k: [rl(t, 4) for t in p] for k, p in paths.items()},
       'loss': {k: rl(v, 4) for k, v in loss_paths.items()},
       'levels': LEVELS_B, 'contours': {str(c): ravine_contour(c) for c in LEVELS_B},
       'steeper': 100}                                     # the curvature in theta_2 over the curvature in theta_1

# ---------------------------------------------------------------------------------------- Lecture 8: a vector-Jacobian product
xv = torch.tensor([1., 2.], requires_grad=True)
z = torch.stack([xv[0] + xv[1], 2 * xv[0] - xv[1]])
gz = torch.tensor([3., 4.])
z.backward(gz)
J = torch.autograd.functional.jacobian(lambda u: torch.stack([u[0] + u[1], 2 * u[0] - u[1]]), xv.detach())
vjp = {'x': rl(xv.detach(), 0), 'J': [rl(r, 0) for r in J], 'g': rl(gz, 0), 'JTg': rl(xv.grad, 0),
       'units': 4096, 'entries': 4096 * 4096}
assert vjp['JTg'] == [11, -1] and vjp['J'] == [[1, 1], [2, -1]] and np.allclose(J.T @ gz, xv.grad)
assert close(vjp['entries'] / 1e7, 1.68, 0.01)            # "about 1.7 x 10^7 entries"

# ---------------------------------------------------------------------------------------- Lecture 9: softmax and logits
def naive(zs):
    zt = torch.tensor(zs, dtype=torch.float32)
    e = zt.exp()
    return e, e / e.sum()


e1, p1 = naive([2., 1., 0.])
e2, p2 = naive([1000., 999., 998.])
zs3 = torch.tensor([1000., 999., 998.], dtype=torch.float32)
shift = zs3 - zs3.max()
e3, p3 = naive(shift.tolist())
fmt = lambda t: [('inf' if math.isinf(v) else 'nan' if math.isnan(v) else rd(v, 4)) for v in t.tolist()]
ce = torch.nn.functional.cross_entropy(zs3[None], torch.tensor([0]))
softmax = {'rows': [{'z': [2, 1, 0], 'exp': fmt(e1), 'sum': rd(e1.sum(), 3), 'p': fmt(p1)},
                    {'z': [1000, 999, 998], 'exp': fmt(e2), 'sum': 'inf', 'p': fmt(p2)},
                    {'z': [int(v) for v in shift.tolist()], 'exp': fmt(e3), 'sum': rd(e3.sum(), 3), 'p': fmt(p3)}],
           'max': 1000, 'ce': rd(ce, 3), 'fp32_max_exp': rd(math.log(torch.finfo(torch.float32).max), 1)}
assert np.allclose(p1.tolist(), [0.665, 0.245, 0.090], atol=5e-4) and np.allclose(p3.tolist(), p1.tolist(), atol=1e-6)
assert all(v == 'nan' for v in softmax['rows'][1]['p']) and all(v == 'inf' for v in softmax['rows'][1]['exp'])
assert close(softmax['rows'][0]['sum'], 11.107) and close(softmax['rows'][2]['sum'], 1.503) and close(ce, 0.408)

# ---------------------------------------------------------------------------------------- Lecture 9: a cosine schedule
E0, EMIN, T = 0.1, 0.001, 100
cosine = [EMIN + (E0 - EMIN) / 2 * (1 + math.cos(math.pi * t / T)) for t in range(T + 1)]
NOISE = 0.5                                                  # Lecture 9 (L5c): the same minibatch noise |eps_B| = 0.5
bounce = [{'eta': e, 'move': rd(e * NOISE, 3)} for e in (0.20, 0.04)]
sched = {'eta0': E0, 'eta_min': EMIN, 'T': T, 'eta': rl(cosine, 5), 'mid': T // 2, 'noise': NOISE, 'bounce': bounce}
assert close(cosine[50], 0.0505, 1e-9) and close(cosine[100], 0.001, 1e-12) and close(cosine[0], 0.1, 1e-12)
assert [x['move'] for x in bounce] == [0.1, 0.02]

OUT.write_text(json.dumps({'source': 'scripts/sci/p2.py', 'bp': bp, 'sgd': sgd, 'opt': opt, 'vjp': vjp,
                           'softmax': softmax, 'sched': sched}, indent=1) + '\n')
print('bp', bp)
print('sgd', {k: (v['grad'], v['theta1'], v['loss1'], v['own0'], v['own1']) for k, v in choices.items()})
print('opt loss at 1, 2, 6, 55:', {k: [rd(v[i], 3) for i in (1, 2, 6, 55)] for k, v in loss_paths.items()})
print('opt ranges', {k: (p.min(0).round(3).tolist(), p.max(0).round(3).tolist()) for k, p in paths.items()})
print('vjp', vjp)
print('softmax', softmax)
print('sched', cosine[50], bounce)
