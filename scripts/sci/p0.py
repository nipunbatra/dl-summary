"""The numbers of the opening: the recipe, run once on Lecture 4's XOR network.

    /Users/nipun/git/clip-lab/.venv/bin/python scripts/sci/p0.py      writes data/p0.json

The network is the one Lecture 4 built by hand: h1 = ReLU(x1 + x2), h2 = ReLU(x1 + x2 - 1), z = 2 h1 - 4 h2 - 1,
p = sigmoid(z). One example, x = (1, 0) with y = 1, goes through the six steps of the recipe; then the step is
repeated on all four XOR points to show the loop at work.
"""
import json
import math
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parents[2] / 'data' / 'p0.json'
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([0., 1., 1., 0.])
ETA = 0.1


def fresh():
    W1 = torch.tensor([[1., 1.], [1., 1.]], requires_grad=True)   # rows: the two hidden units
    b1 = torch.tensor([0., -1.], requires_grad=True)
    w2 = torch.tensor([2., -4.], requires_grad=True)
    b2 = torch.tensor(-1., requires_grad=True)
    return [W1, b1, w2, b2]


def forward(params, x):
    W1, b1, w2, b2 = params
    h = torch.relu(x @ W1.T + b1)
    z = h @ w2 + b2
    return h, z


def bce(z, y):                      # -log p(y | x) for a Bernoulli with p = sigmoid(z)
    return torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction='none')


r = lambda t, n=4: [round(float(v), n) for v in t.detach().flatten()] if t.ndim else round(float(t.detach()), n)

# ---- one example through the recipe
params = fresh()
theta0 = {'W1': r(params[0], 0), 'b1': r(params[1], 0), 'w2': r(params[2], 0), 'b2': int(params[3])}
x, y = X[2], Y[2]
h, z = forward(params, x)
p = torch.sigmoid(z)
loss = bce(z, y)
loss.backward()
grads = [g.grad.clone() for g in params]
with torch.no_grad():
    for q, g in zip(params, grads):
        q -= ETA * g
h2, z2 = forward(params, x)
p2, loss2 = torch.sigmoid(z2), bce(z2, y)

one = {
    'theta0': theta0, 'x': [int(v) for v in x], 'y': int(y), 'h': r(h), 'z': r(z), 'p': r(p), 'loss': r(loss),
    'dz': r(p - y),                                   # d loss / d z = p - y
    'grad': {'W1': r(grads[0]), 'b1': r(grads[1]), 'w2': r(grads[2]), 'b2': r(grads[3])},
    'eta': ETA,
    'after': {'W1': r(params[0]), 'b1': r(params[1]), 'w2': r(params[2]), 'b2': r(params[3]),
              'h': r(h2), 'z': r(z2), 'p': r(p2), 'loss': r(loss2)},
    'n_params': sum(q.numel() for q in params),
}
assert abs(one['p'] - 0.7311) < 1e-3 and abs(one['loss'] - 0.3133) < 1e-3 and one['n_params'] == 9

# ---- the loop: the same step on all four points, many times, at three step sizes
def run(eta, steps=300):
    """Loss and the four predictions before every step, and after the last one."""
    params, curve, probs, h2 = fresh(), [], [], []
    for t in range(steps + 1):
        hs, zs = forward(params, X)
        L = bce(zs, Y).mean()
        curve.append(round(L.item(), 4))
        probs.append(r(torch.sigmoid(zs), 3))
        h2.append(hs[:, 1].max().item())             # the second hidden unit's largest value on the four inputs
        if t == steps:
            alive = [i for i, v in enumerate(h2) if v > 0]
            return curve, probs, (alive[-1] + 1 if len(alive) < len(h2) and h2[-1] == 0 else None)
        for q in params:
            q.grad = None
        L.backward()
        with torch.no_grad():
            for q in params:
                q -= eta * q.grad


ETAS = [0.01, 0.1, 1.0]
runs = {str(e): run(e) for e in ETAS}
loop = {'steps': 300, 'etas': ETAS, 'default': ETA,
        'loss': {k: v[0] for k, v in runs.items()},          # mean loss on the four points, before step t
        'probs': {k: v[1] for k, v in runs.items()},         # the four predictions, before step t
        'peak': {k: {'loss': max(v[0]), 'step': v[0].index(max(v[0]))} for k, v in runs.items()},
        'h2_off_from': {k: v[2] for k, v in runs.items()},   # the step from which the second hidden unit is 0 on all four inputs, for good
        'ln2': round(math.log(2), 4),
        'inputs': [[int(a), int(b)] for a, b in X.tolist()], 'targets': [int(v) for v in Y]}
assert abs(loop['loss']['0.1'][0] - one['loss']) < 1e-9      # all four points start at the same loss
# the large step switches the second hidden unit off for good; one unit cannot draw XOR, so every prediction ends at 0.5
assert loop['h2_off_from']['1.0'] == 2 and loop['h2_off_from']['0.1'] is None and abs(loop['loss']['1.0'][-1] - loop['ln2']) < 1e-4

OUT.write_text(json.dumps({'source': 'scripts/sci/p0.py', 'xor': one, 'loop': loop}, indent=1) + '\n')
print(json.dumps(one, indent=1))
for k, (c, pr, off) in runs.items():
    print('eta', k, 'loss', c[0], c[1], c[10], c[100], c[-1], 'peak', loop['peak'][k], 'p', pr[0], '->', pr[-1], 'h2 off from', off)
