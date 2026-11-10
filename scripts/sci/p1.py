"""The numbers of Part 1 (Foundations, Lectures 1 to 4): every loss is a negative log-likelihood, a prior adds a
penalty, and layers of ReLUs make the model bend.

    ~/base/bin/python scripts/sci/p1.py      writes data/p1.json

Everything here is cheap, so everything is recomputed on the lectures' own examples (plan/digests/01-foundations.md):
the ten tosses HHTTTHHTTT (Lecture 1), the binary and three-class cross-entropy (Lecture 2), the four-point MAP fit
with sigma = 0.5 and a Gaussian prior of width tau (Lecture 3), the hand-built XOR network and the hinge fits of
sin x on [-pi, pi] (Lecture 4). The asserts check that each matches the number the lecture printed.
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'data' / 'p1.json'
r = lambda v, n=4: round(float(v), n)
rl = lambda a, n=4: [r(v, n) for v in a]

# ---------------------------------------------------------------- 1.1 the coin (Lecture 1)
SEQ = 'HHTTTHHTTT'
nH, nT = SEQ.count('H'), SEQ.count('T')
grid = np.round(np.arange(1, 100) / 100, 2)                      # theta = 0.01 ... 0.99, the slider's stops
lik = grid ** nH * (1 - grid) ** nT
nll = -(nH * np.log(grid) + nT * np.log(1 - grid))
L = lambda t: t ** nH * (1 - t) ** nT
NLL = lambda t: -(nH * math.log(t) + nT * math.log(1 - t))
mle = nH / (nH + nT)
coin = {
    'seq': list(SEQ), 'nH': nH, 'nT': nT, 'n': len(SEQ),
    'theta': [float(t) for t in grid],
    'lik_milli': rl(lik * 1e3, 5),                               # L(theta) in thousandths, for the plot
    'lik': rl(lik, 7), 'nll': rl(nll, 4),
    'cands': [0.4, 0.7], 'start': 0.7,
    'L_cand': [r(L(t), 7) for t in (0.4, 0.7)], 'nll_cand': [r(NLL(t), 4) for t in (0.4, 0.7)],
    'ratio': r(L(0.4) / L(0.7), 2),
    'mle': mle, 'nll_min': r(NLL(mle), 4),
}
assert abs(coin['nll_cand'][0] - 6.73) < 5e-3 and abs(coin['nll_cand'][1] - 8.65) < 5e-3     # Lecture 1: 6.73 and 8.65
assert abs(coin['L_cand'][0] - 0.00119) < 1e-5 and abs(coin['L_cand'][1] - 0.000175) < 1e-6
assert abs(coin['ratio'] - 6.8) < 0.05 and mle == 0.4 and grid[int(np.argmin(nll))] == 0.4

# ---------------------------------------------------------------- 1.2 cross-entropy (Lecture 2)
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
z = np.array([2., 1., -0.5])
p = softmax(z)
y = np.array([1., 0., 0.])                                      # the right class is class 1
ce = {
    'bin': {'y': 1, 'p': 0.8, 'loss': r(-math.log(0.8)), 'p_bad': 0.2, 'loss_bad': r(-math.log(0.2))},
    'z': list(z), 'y': [int(v) for v in y], 'p': rl(p, 3), 'lse': r(math.log(np.exp(z).sum()), 3),
    'loss': r(-math.log(p[0]), 3), 'grad': rl(p - y, 3),
}
assert abs(ce['bin']['loss'] - 0.223) < 1e-3 and abs(ce['bin']['loss_bad'] - 1.609) < 1e-3   # Lecture 2: 0.22, 1.61
assert abs(ce['p'][0] - 0.690) < 1e-3 and abs(ce['loss'] - 0.372) < 1e-3 and abs(ce['lse'] - 2.372) < 1e-3

# ---------------------------------------------------------------- 1.3 squared error is a Gaussian NLL (S03 cheat sheet)
res, sig = 2.0, 1.0
gauss = {'r': res, 'sigma': sig, 'sq': r(res ** 2 / (2 * sig ** 2)), 'const': r(0.5 * math.log(2 * math.pi * sig ** 2)),
         'total': r(res ** 2 / (2 * sig ** 2) + 0.5 * math.log(2 * math.pi * sig ** 2))}
assert gauss['sq'] == 2.0

# ---------------------------------------------------------------- 1.4 the four-point MAP fit (Lecture 3)
xs = np.array([-1., 0., 1., 2.])
ys = np.array([-0.8, 1.8, 5.3, 7.9])
SIGMA = 0.5
Xt = np.column_stack([np.ones_like(xs), xs])                    # theta = (b, w)


def fit(tau):
    """MAP for a Gaussian likelihood of width sigma and a prior N(0, tau^2 I): ridge with sigma^2 / tau^2."""
    A = Xt.T @ Xt + (0 if tau is None else SIGMA ** 2 / tau ** 2) * np.eye(2)
    return np.linalg.solve(A, Xt.T @ ys)


TAUS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
ml = fit(None)
mp = {
    'x': list(xs), 'y': list(ys), 'sigma': SIGMA, 'true': [2, 3],
    'ml': rl(ml, 2), 'ml_sse': r(((ys - Xt @ ml) ** 2).sum(), 3),
    'taus': TAUS, 'default': 0.5,
    'fits': [rl(fit(t), 2) for t in TAUS],
    'lam': [r(1 / (2 * t ** 2), 3) for t in TAUS],              # lambda = 1 / (2 tau^2), the weight of the L2 penalty
}
assert mp['ml'] == [2.07, 2.96]                                                     # Lecture 3: (2.07, 2.96)
assert mp['fits'][TAUS.index(0.5)] == [1.79, 2.62] and mp['fits'][TAUS.index(0.1)] == [0.44, 0.68]
assert mp['lam'][TAUS.index(0.5)] == 2.0
assert abs(mp['fits'][-1][0] - ml[0]) < 0.01 and abs(mp['fits'][-1][1] - ml[1]) < 0.01   # a wide prior gives ML back

# ---------------------------------------------------------------- 1.5 the XOR network (Lecture 4)
XS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
YS = [0, 1, 1, 0]
W1, b1, w2, b2 = np.array([[1, 1], [1, 1]], float), np.array([0., -1.]), np.array([2., -4.]), -1.
H = np.maximum(XS @ W1.T + b1, 0)
Z = H @ w2 + b2
xor = {'x': XS.astype(int).tolist(), 'y': YS, 'W1': W1.astype(int).tolist(), 'b1': b1.astype(int).tolist(),
       'w2': w2.astype(int).tolist(), 'b2': int(b2), 'h': H.astype(int).tolist(), 'z': Z.astype(int).tolist(),
       'pred': [int(v >= 0) for v in Z]}
assert xor['pred'] == YS and xor['h'] == [[0, 0], [1, 0], [1, 0], [2, 1]] and xor['z'] == [-1, 1, 1, -1]

# ---------------------------------------------------------------- 1.5 hinges approximate sin x (Lecture 4)
dense = np.linspace(-math.pi, math.pi, 20001)


def hinge_fit(m):
    """m hinges: the piecewise-linear interpolant of sin x with m + 2 equally spaced knots on [-pi, pi]."""
    knots = np.linspace(-math.pi, math.pi, m + 2)
    g = np.interp(dense, knots, np.sin(knots))
    gap = np.abs(np.sin(dense) - g)
    i = int(np.argmax(gap))
    return {'m': m, 'kx': rl(knots, 4), 'ky': rl(np.sin(knots), 4), 'err': r(gap[i], 3),
            'at': r(dense[i], 4), 'f_at': r(math.sin(dense[i]), 4), 'g_at': r(g[i], 4)}


HINGES = [2, 5, 15]
curve = np.linspace(-math.pi, math.pi, 181)
hinges = {'ms': HINGES, 'fits': {str(m): hinge_fit(m) for m in HINGES}, 'eps': 0.05,
          'sweep': {str(m): hinge_fit(m)['err'] for m in range(1, 16)},
          'x': rl(curve, 4), 'sin': rl(np.sin(curve), 4), 'pi': r(math.pi, 4)}
errs = [hinges['fits'][str(m)]['err'] for m in HINGES]
assert errs == [0.437, 0.134, 0.019]                                                # Lecture 4: 0.437, 0.134, 0.019
first_under = min(int(m) for m, e in hinges['sweep'].items() if e < hinges['eps'])
hinges['first_under'] = first_under
hinges['first_under_err'] = hinges['sweep'][str(first_under)]

# ---------------------------------------------------------------- the check: ten equal guesses
K = 10
check = {'K': K, 'p': r(1 / K, 2), 'loss': r(math.log(K), 4)}
assert abs(check['loss'] - 2.3026) < 1e-4

OUT.write_text(json.dumps({'source': 'scripts/sci/p1.py', 'coin': coin, 'ce': ce, 'gauss': gauss, 'map': mp,
                           'xor': xor, 'hinges': hinges, 'check': check}, indent=1) + '\n')
print('coin', coin['L_cand'], coin['nll_cand'], coin['ratio'], coin['nll_min'])
print('ce', ce['bin'], ce['p'], ce['loss'], ce['grad'], ce['lse'])
print('gauss', gauss)
print('map ml', mp['ml'], 'sse', mp['ml_sse'], {t: f for t, f in zip(TAUS, mp['fits'])})
print('xor', xor['h'], xor['z'])
print('hinges', errs, 'sweep', hinges['sweep'], 'first under', first_under)
print('check', check)
