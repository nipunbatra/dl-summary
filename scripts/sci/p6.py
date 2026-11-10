"""The numbers of Part 6: KL divergence, the ELBO, the reparameterisation trick, the Bayesian network and the VAE.

    ~/base/bin/python scripts/sci/p6.py      writes data/p6.json and figures/p6-*.png

Recomputed here (closed forms, cheap): the weather example's entropies and KL (Lecture 23), -ln 0.6 for the
classifier, the coin's evidence and the ELBO of a Beta q (Lecture 24), the reparameterised sample and its gradients,
the KL of the VAE's q for our 3 from its mu and sigma, the pixel-noise weight 1/(2 sigma^2) and the constant of
log p(x | z) (Lecture 25).

Recorded (from trained models; not retrained): the Bayesian network's ELBO and predictive bands, the autoencoder's
and the VAE's numbers for MNIST test digit #90, and the beta sweep. Each recorded constant names the file it came
from, and the script checks it against that file. Curves, codes and images for the figures are read from the same
files.
"""
import base64
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.special import betaln, digamma

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'p6.json'
FIG = ROOT / 'figures'
GEN = Path('/Users/nipun/git/generative/lectures')
L1 = json.loads((GEN / 'lecture1/data/science.json').read_text())     # the KL and VI deck (schedule Lectures 23, 24)
L2 = json.loads((GEN / 'lecture2/data/science.json').read_text())     # the trained AE and VAEs that Lecture 25 reuses
NETS = json.loads((GEN / 'lecture2/data/nets.json').read_text())
MNIST = Path('/Users/nipun/git/psdv-teaching/datasets/MNIST/raw/t10k-images-idx3-ubyte')

r = lambda v, n=4: round(float(v), n)
close = lambda a, b, tol=5e-3: abs(a - b) < tol

# ---------------------------------------------------------------- 6.1 KL divergence: the weather of Lecture 23
p = np.array([0.5, 0.25, 0.25])            # our town: sunny, cloudy, rainy
q = np.array([0.25, 0.25, 0.5])            # the other town, whose code we borrow
H = lambda a: float(-(a * np.log2(a)).sum())
CE = lambda a, b: float(-(a * np.log2(b)).sum())
weather = {'names': ['sunny', 'cloudy', 'rainy'], 'p': p.tolist(), 'q': q.tolist(),
           'H_p': r(H(p)), 'H_pq': r(CE(p, q)), 'kl': r(CE(p, q) - H(p))}
assert weather['H_p'] == 1.5 and weather['H_pq'] == 1.75 and weather['kl'] == 0.25
S = L1['story']['bits']
assert S['entropy'] == 1.5 and S['cross_entropy'] == 1.75 and S['kl'] == 0.25 and S['p'] == p.tolist() and S['q'] == q.tolist()
# the live figure: q slides in a straight line towards p, q_t = (1 - t) q + t p
path_kl = []
for t in np.linspace(0, 1, 21):
    qt = (1 - t) * q + t * p
    path_kl.append({'t': r(t, 2), 'q': [r(v, 4) for v in qt], 'H_pq': r(CE(p, qt)), 'kl': r(CE(p, qt) - H(p))})
assert path_kl[0]['kl'] == 0.25 and path_kl[-1]['kl'] == 0
weather['path'] = path_kl

# the classifier of Lecture 23: true class cat, the network gives it 0.60; the label is one-hot, so H(label) = 0
C = L1['story']['classify']
pred = C['pred']
classify = {'classes': C['classes'], 'pred': pred, 'label': C['label'], 'p_true': pred[C['label']],
            'loss_nats': r(-math.log(pred[C['label']])), 'loss_bits': r(-math.log2(pred[C['label']]))}
assert close(classify['loss_nats'], 0.5108, 1e-4) and close(classify['loss_nats'], C['kl_onehot'], 1e-4)
assert close(-torch.nn.functional.cross_entropy(torch.log(torch.tensor([pred])), torch.tensor([C['label']])).item(), -0.5108, 1e-4)

# ---------------------------------------------------------------- 6.2 the ELBO: the coin of Lecture 24
A0, B0, HEADS, TAILS = 5.0, 3.0, 17, 13     # prior Beta(5, 3); 17 heads in 30 ordered tosses
log_ev = float(betaln(A0 + HEADS, B0 + TAILS) - betaln(A0, B0))   # ln p(D) for the ordered sequence


def kl_beta(a, b, c, d):
    """KL(Beta(a, b) || Beta(c, d))."""
    return float(betaln(c, d) - betaln(a, b) + (a - c) * digamma(a) + (b - d) * digamma(b)
                 + (c - a + d - b) * digamma(a + b))


def elbo(a, b):
    """ELBO of q = Beta(a, b): the expected log-likelihood minus KL(q || prior)."""
    fit = HEADS * (digamma(a) - digamma(a + b)) + TAILS * (digamma(b) - digamma(a + b))
    kp = kl_beta(a, b, A0, B0)
    return float(fit), kp, float(fit - kp)


coin = {'prior': [A0, B0], 'heads': HEADS, 'tosses': HEADS + TAILS, 'post': [A0 + HEADS, B0 + TAILS], 'log_ev': r(log_ev)}
for name, (a, b) in {'start': (10, 10), 'posterior': (22, 16), 'prior_q': (5, 3)}.items():
    fit, kp, e = elbo(a, b)
    coin[name] = {'a': a, 'b': b, 'fit': r(fit), 'kl_prior': r(kp), 'elbo': r(e), 'gap': r(log_ev - e),
                  'kl_post': r(kl_beta(a, b, A0 + HEADS, B0 + TAILS))}
    assert close(coin[name]['gap'], coin[name]['kl_post'], 1e-6)      # ln p(D) = ELBO + KL(q || posterior)
C1 = L1['coin']
assert close(log_ev, C1['log_evidence'], 1e-9) and close(log_ev, -21.40, 5e-3)
assert close(coin['start']['elbo'], C1['candidates'][0]['elbo'], 1e-4) and close(coin['start']['gap'], 0.62, 5e-3)
assert close(coin['posterior']['fit'], C1['drift'][1]['fit'], 1e-4) and close(coin['posterior']['kl_prior'], 0.46, 5e-3)
assert close(coin['posterior']['gap'], 0, 1e-9)
# the live figure: q walks in a straight line from Beta(10, 10) through the posterior Beta(22, 16) and beyond
rs = np.linspace(0.005, 0.995, 100)
beta_pdf = lambda a, b: np.exp((a - 1) * np.log(rs) + (b - 1) * np.log(1 - rs) - betaln(a, b))
walk = []
for s in np.round(np.arange(0, 1.501, 0.05), 2):
    a, b = 10 + 12 * s, 10 + 6 * s
    fit, kp, e = elbo(a, b)
    walk.append({'s': float(s), 'a': r(a, 1), 'b': r(b, 1), 'fit': r(fit), 'kl_prior': r(kp), 'elbo': r(e), 'gap': r(log_ev - e),
                 'pdf': [r(v, 3) for v in beta_pdf(a, b)]})
assert walk[20]['a'] == 22 and walk[20]['b'] == 16 and close(walk[20]['gap'], 0, 1e-6) and close(walk[0]['gap'], 0.6229, 1e-4)
coin['walk'] = walk
coin['r'] = [r(v, 4) for v in rs]
coin['post_pdf'] = [r(v, 3) for v in beta_pdf(A0 + HEADS, B0 + TAILS)]
coin['walk_default'] = 0

# ---------------------------------------------------------------- 6.3 the reparameterisation trick, by hand (Lecture 24)
mu, sigma = torch.tensor(0., requires_grad=True), torch.tensor(2., requires_grad=True)
eps = torch.tensor(-0.75)
theta = mu + sigma * eps
theta.backward()
reparam = {'mu': mu.item(), 'sigma': sigma.item(), 'eps': eps.item(), 'theta': theta.item(),
           'dmu': mu.grad.item(), 'dsigma': sigma.grad.item()}
assert reparam['theta'] == -1.5 and reparam['dmu'] == 1 and reparam['dsigma'] == -0.75
# the mixture fit of Lecture 24 after 400 steps, sample() against rsample(): rows [step, mu, sigma, KL, ...]
MF = L1['story']['mixfit']
reparam['mix'] = {'steps': MF['steps'], 'start': MF['start'],
                  'sample': {'mu': r(MF['sample'][-1][1], 2), 'sigma': r(MF['sample'][-1][2], 2), 'kl': r(MF['sample'][-1][4], 1)},
                  'rsample': {'mu': r(MF['rsample'][-1][1], 2), 'sigma': r(MF['rsample'][-1][2], 2), 'kl': r(MF['rsample'][-1][4], 3)}}
# recorded from lecture1/data/science.json, story.mixfit (the deck's 'q stays at N(6, 0.1^2)' and 'KL 0.056')
assert MF['steps'] == 400 and close(MF['rsample'][-1][4], 0.056, 1e-3) and close(MF['sample'][-1][1], 6.08, 1e-2)

# ---------------------------------------------------------------- 6.4 the tiny Bayesian network (Lectures 23, 24): recorded
B = L1['story']['bnn']
# recorded from lecture1/data/science.json, story.bnn.vi (mean-field VI, 16 draws a step, Adam lr 0.02, 4,000 steps)
BNN_ELBO_START, BNN_ELBO, BNN_FIT, BNN_KL = -1234.5874, -7.1809, 8.0967, 15.2775
assert close(B['vi']['frames'][0][1], BNN_ELBO_START, 1e-4) and close(B['vi']['elbo'], BNN_ELBO, 1e-4)
assert close(B['vi']['expected_log_lik'], BNN_FIT, 1e-4) and close(B['vi']['kl_prior'], BNN_KL, 1e-4)
assert close(BNN_FIT - BNN_KL, BNN_ELBO, 1e-3)
grid = np.array(B['grid'])
i0 = int(np.argmin(abs(grid)))                               # x = 0, in the middle of the gap
iL = 0                                                       # x = -2.5, the left edge of the plot
bnn = {'n_weights': len(B['names']), 'n_points': len(B['x']), 'n_q': 2 * len(B['names']), 'draws': B['vi']['draws'],
       'steps': B['vi']['steps'], 'noise': B['noise_sd'],
       'elbo_start': r(BNN_ELBO_START, 2), 'elbo': r(BNN_ELBO, 2), 'fit': r(BNN_FIT, 2), 'kl': r(BNN_KL, 2),
       'x': [r(v, 3) for v in B['x']], 'y': [r(v, 3) for v in B['y']], 'grid': [r(v, 3) for v in grid],
       'vi_mean': [r(v, 3) for v in B['vi']['mean']], 'vi_sd': [r(v, 3) for v in B['vi']['sd']],
       'hmc_mean': [r(v, 3) for v in B['hmc']['mean']], 'hmc_sd': [r(v, 3) for v in B['hmc']['sd']],
       'at_gap': {'x': r(grid[i0], 2), 'vi': r(B['vi']['sd'][i0], 2), 'hmc': r(B['hmc']['sd'][i0], 2)},
       'at_edge': {'x': r(grid[iL], 2), 'vi': r(B['vi']['sd'][iL], 2), 'hmc': r(B['hmc']['sd'][iL], 2)},
       'near_data': {'x': -1.0, 'vi': r(B['vi']['sd'][int(np.argmin(abs(grid + 1)))], 2)}}
assert bnn['n_weights'] == 7 and bnn['n_points'] == 17 and bnn['n_q'] == 14
assert bnn['at_gap']['vi'] > bnn['near_data']['vi'] and bnn['at_edge']['hmc'] > bnn['at_edge']['vi']

# ---------------------------------------------------------------- 6.5, 6.6 the autoencoder and the VAE on our 3 (Lecture 25)
V, AE = L2['vae'], L2['ae']
shapes = [w['shape'] for w in NETS['ae2']['enc'] if len(w['shape']) == 2]
arch = [shapes[0][1], shapes[0][0], shapes[1][0]]           # 784 -> 400 -> 2
assert arch == [784, 400, 2]
SIGMA_PIX = L2['meta']['sigma']                               # pixel noise of the VAE's decoder, 0.2
# recorded from lecture2/data/science.json (the trained networks Lecture 25 reuses): ae.ae2_sse3, vae.q3, vae.grid.log_px
AE_SSE3 = 26.31
Q3_MU, Q3_SD = [-0.2106, 0.0702], [0.013, 0.0104]
Q3_FIT, Q3_KL, Q3_ELBO, LOG_PX = 251.309, 7.9403, 243.369, 244.07
assert AE['ae2_sse3'] == AE_SSE3 and V['q3']['mu'] == Q3_MU and V['q3']['sd'] == Q3_SD
assert V['q3']['exp_loglik'] == Q3_FIT and V['q3']['kl_prior'] == Q3_KL and V['q3']['elbo'] == Q3_ELBO and V['grid']['log_px'] == LOG_PX
# recomputed: the closed-form KL of the encoder's q to N(0, I), from the rounded mu and sigma
kl3 = 0.5 * sum(m_ ** 2 + s_ ** 2 - math.log(s_ ** 2) - 1 for m_, s_ in zip(Q3_MU, Q3_SD))
assert close(kl3, Q3_KL, 0.02)
weight = 1 / (2 * SIGMA_PIX ** 2)                             # 12.5
const = -784 * math.log(SIGMA_PIX * math.sqrt(2 * math.pi))   # +541.35, the constant in ln p(x | z)
assert close(weight, 12.5, 1e-9) and close(const, -V['loglik_const'], 1e-2)
assert close(Q3_FIT, const - weight * V['q3']['exp_sse'], 0.01)
vae = {'arch': arch, 'sigma_pix': SIGMA_PIX, 'weight': r(weight, 2), 'const': r(const, 2), 'index': L2['meta']['example_index'],
       'label': L2['mnist']['display_labels'][0], 'ae_sse': AE_SSE3, 'ae_z': AE['ae2_z3'],
       'mu': [r(v, 3) for v in Q3_MU], 'sd': [r(v, 3) for v in Q3_SD], 'exp_sse': V['q3']['exp_sse'],
       'fit': r(Q3_FIT, 2), 'kl': r(Q3_KL, 2), 'kl_recomputed': r(kl3, 2), 'elbo': r(Q3_FIT - Q3_KL, 2), 'log_px': r(LOG_PX, 2),
       'gap': r(LOG_PX - (Q3_FIT - Q3_KL), 2)}
assert vae['elbo'] == 243.37 and vae['gap'] == 0.70 and vae['label'] == 3 and vae['index'] == 90

# the beta sweep, recorded from lecture2/data/science.json, vae.beta: KL and squared error per test image
BETA_KL = {'0': 68.323, '0.1': 20.811, '1': 9.019, '4': 6.316, '16': 4.498, '64': 1.535, '256': 0.0}
BETA_SSE = {'0': 30.547, '0.1': 30.423, '1': 30.773, '4': 31.297, '16': 32.352, '64': 37.789, '256': 52.903}
labels = L2['mnist']['display_labels']
beta = {'values': list(BETA_KL), 'default': '1', 'kl': BETA_KL, 'sse': BETA_SSE, 'codes': {}, 'sd_mean': {}}
for k in BETA_KL:
    E = V['beta'][k]
    assert E['test_kl'] == BETA_KL[k] and E['test_sse'] == BETA_SSE[k]
    beta['codes'][k] = [[r(z[0], 2), r(z[1], 2)] for z in E['codes'][:400]]
    beta['sd_mean'][k] = E['sd_mean']
beta['labels'] = labels[:400]
assert beta['codes']['1'][0] == [-0.21, 0.07]                 # code 0 is our 3, the mean mu of its q

# ---------------------------------------------------------------- images: our 3, its rebuilds, and samples per beta
IMG = L2['img']
dec = lambda key: np.frombuffer(base64.b64decode(IMG[key]['data']), dtype=np.uint8).reshape(IMG[key]['n'], 28, 28)
x3 = dec('example')[0]
raw = np.frombuffer(MNIST.read_bytes()[16:], dtype=np.uint8).reshape(-1, 28, 28)
assert np.array_equal(raw[vae['index']], x3)                   # the image is MNIST test digit #90
ae3 = dec('ae2_recon3')[0]
sse_png = float((((x3 / 255.) - (ae3 / 255.)) ** 2).sum())
assert close(sse_png, AE_SSE3, 0.3), sse_png                 # the stored 8-bit rebuild gives the same squared error
INK, PAPER = np.array([20, 23, 31]), np.array([250, 250, 251])


def png(arrs, name, cols, scale=6, gap=12):
    rows_ = math.ceil(len(arrs) / cols)
    W, Hh = cols * 28 * scale + (cols - 1) * gap, rows_ * 28 * scale + (rows_ - 1) * gap
    canvas = np.ones((Hh, W, 3)) * PAPER
    for i, a in enumerate(arrs):
        v = np.kron(a / 255., np.ones((scale, scale)))[..., None]
        y0, x0 = (i // cols) * (28 * scale + gap), (i % cols) * (28 * scale + gap)
        canvas[y0:y0 + 28 * scale, x0:x0 + 28 * scale] = PAPER * (1 - v) + INK * v
    Image.fromarray(canvas.round().astype(np.uint8)).save(FIG / f'p6-{name}.png', optimize=True)
    return [W, Hh]


FIG.mkdir(exist_ok=True)
images = {'x3': png([x3], 'x3', 1, 8), 'ae3': png([ae3], 'ae3', 1, 8), 'vae3': png([dec('vae2_mean_recon3')[0]], 'vae3', 1, 8)}
for k in BETA_KL:
    images['beta-' + k.replace('.', '_')] = png(list(dec(f'beta{k}_prior_samples')[:8]), 'beta-' + k.replace('.', '_'), 4)
images['samples'] = 8

out = {'source': 'scripts/sci/p6.py', 'weather': weather, 'classify': classify, 'coin': coin, 'reparam': reparam,
       'bnn': bnn, 'vae': vae, 'beta': beta, 'images': images,
       'recorded': {'bnn': 'lecture1/data/science.json, story.bnn.vi and story.bnn.hmc',
                    'mixfit': 'lecture1/data/science.json, story.mixfit',
                    'vae': 'lecture2/data/science.json, ae.ae2_sse3, vae.q3, vae.grid.log_px (the networks Lecture 25 reuses)',
                    'beta': 'lecture2/data/science.json, vae.beta'}}
OUT.write_text(json.dumps(out, separators=(',', ':')) + '\n')
print('weather', {k: weather[k] for k in ('H_p', 'H_pq', 'kl')}, 'classify', classify['loss_nats'], classify['loss_bits'])
print('coin', coin['log_ev'], coin['start'], coin['posterior'], coin['prior_q'])
print('reparam', {k: v for k, v in reparam.items() if k != 'mix'}, reparam['mix'])
print('bnn', {k: bnn[k] for k in ('elbo_start', 'elbo', 'fit', 'kl', 'at_gap', 'at_edge', 'near_data')})
print('vae', vae, 'sse from png', round(sse_png, 2))
print('beta kl', BETA_KL, 'images', images)
