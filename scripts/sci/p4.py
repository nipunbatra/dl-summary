"""The numbers of Part 4: attention and language (Lectures 11 to 15).

    ~/base/bin/python scripts/sci/p4.py      writes data/p4.json

Lectures 11 to 13 embed their trained (L11) and hand-built (L12, L13) models in the pages as `window.__TOY__`, so the
probabilities and attention weights are recomputed here from those weights, read from the attention repo's published
branch (read-only: `git -C /Users/nipun/git/attention show origin/main:<file>`), and checked against the numbers the
lectures print. The small arithmetic (scaling, the masked softmax, -ln p, perplexity of a uniform guess) is computed.
The TinyStories results of Lecture 14 come from trained models, so they are recorded, not recomputed.
"""
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'data' / 'p4.json'
REPO = '/Users/nipun/git/attention'


def toy(page):
    """The model a lecture page embeds as window.__TOY__ = {...};"""
    html = subprocess.run(['git', '-C', REPO, 'show', f'origin/main:{page}'], capture_output=True, text=True, check=True).stdout
    line = re.search(r'window\.__TOY__ = (\{.*)', html).group(1).rstrip().rstrip(';')
    return json.loads(line)


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z[np.isfinite(z)].max())
    return e / e.sum()


r = lambda v, n=4: [round(float(x), n) for x in v] if np.ndim(v) else round(float(v), n)

# ---------------------------------------------------------------- Lecture 11: the character model (part1.html)
C = toy('part1.html')
VOC, E = C['vocab'], np.array(C['E'])
W1, b1, W2, b2 = (np.array(C[k]) for k in ('W1', 'b1', 'W2', 'b2'))


def logits(ctx):
    a0 = np.concatenate([E[VOC.index(c)] for c in ctx])          # look up three rows and join them: 6 numbers
    return np.maximum(0, a0 @ W1 + b1) @ W2 + b2                  # 32 ReLU units, then 27 logits


NAME, W = 'aabid', C['w']
padded = '-' * W + NAME + '-'
pairs = [[padded[i:i + W], padded[i + W]] for i in range(len(NAME) + 1)]
FOCUS = 3                                                         # aab -> i
ctx, tgt = pairs[FOCUS]
p = softmax(logits(ctx))
order = np.argsort(-p)
SHOW = 8                                                          # bars on the slide: the eight largest of 27
aab = {'context': ctx, 'target': tgt, 'p_target': r(p[VOC.index(tgt)], 4), 'loss': r(-math.log(p[VOC.index(tgt)]), 4),
       'top': [[VOC[i], r(p[i], 4)] for i in order[:SHOW]], 'rest': r(1 - p[order[:SHOW]].sum(), 4),
       'n_vocab': len(VOC), 'rank_target': int(np.where(order == VOC.index(tgt))[0][0]) + 1,
       'p_target_untrained': None}
# the same pair before training (the lecture's "before" checkpoint)
Bf = C['before']
a0 = np.concatenate([np.array(Bf['E'])[VOC.index(c)] for c in ctx])
p0 = softmax(np.maximum(0, a0 @ np.array(Bf['W1']) + np.array(Bf['b1'])) @ np.array(Bf['W2']) + np.array(Bf['b2']))
aab['p_target_untrained'] = r(p0[VOC.index(tgt)], 4)
assert pairs[0] == ['---', 'a'] and pairs[-1] == ['bid', '-'] and len(pairs) == 6
assert round(aab['p_target'], 3) == 0.117 and round(aab['loss'], 2) == 2.15 and aab['top'][0][0] == '-' and abs(aab['top'][0][1] - 0.464) < 1e-3
assert abs(aab['p_target'] - C['aabid_rows'][3]['probability']) < 1e-4 and round(aab['p_target_untrained'], 3) == 0.052

# temperature on the start of a new name, "- s a" (Lecture 11, s10). The JS applies softmax(z / T) to these logits.
TCTX = '-sa'
z = logits(TCTX)
zt = r(z, 5)
pt = lambda T: softmax(np.array(zt) / T)
tord = [int(i) for i in np.argsort(-pt(1.0))]
MARKS = [0.5, 1.0, 1.5]
temp = {'context': TCTX, 'vocab': VOC, 'logits': zt, 'order': tord[:SHOW], 'top': VOC[tord[0]],
        'p_top': {str(T): r(pt(T)[tord[0]], 4) for T in MARKS}, 'marks': MARKS,
        'T_min': 0.3, 'T_max': 2.0, 'T_step': 0.05, 'T_print': 0.5,
        'y_max': None}
# the bar chart's fixed height: the tallest bar over the slider's range (the top character, or the other 19 together)
tallest = max(max(pt(T)[tord[0]], 1 - pt(T)[tord[:SHOW]].sum()) for T in np.arange(temp['T_min'], temp['T_max'] + 1e-9, temp['T_step']))
temp['y_max'] = r(math.ceil(tallest * 10) / 10, 1)
assert temp['top'] == 'n' and [round(temp['p_top'][str(T)], 3) for T in MARKS] == [0.342, 0.161, 0.112]
assert pt(2.0)[tord[0]] < pt(1.0)[tord[0]] < pt(0.5)[tord[0]] and temp['y_max'] == 0.6

# ---------------------------------------------------------------- Lectures 12 and 13: the hand-built bank model (attention.html)
A = toy('attention.html')
POS = np.array(A['pos_emb'])
WQ, WK, WV, WO = (np.array(A[k]) for k in ('W_Q', 'W_K', 'W_V', 'W_O'))
DK = A['d_k']


def rows(words):
    E0 = np.array([A['tok_emb'][w.lower()] for w in words]) + POS[:len(words)]
    return E0, E0 @ WQ, E0 @ WK, E0 @ WV


bank = {}
for key, name in (('A', 'river'), ('B', 'cheque')):
    words = A['sentences'][name]
    E0, Q, K, V = rows(words)
    i = [w.lower() for w in words].index('bank')
    alpha = softmax(Q[i] @ K[:i + 1].T / math.sqrt(DK))
    m = alpha @ V[:i + 1]
    d = m @ WO
    srt = np.argsort(-alpha)
    bank[key] = {'words': words, 'receiver': i, 'weights': r(alpha, 3), 'first': [words[srt[0]], r(alpha[srt[0]], 3)],
                 'second': [words[srt[1]], r(alpha[srt[1]], 3)], 'e': r(E0[i], 2), 'm': r(m, 2), 'delta': r(d, 2), 'e_new': r(E0[i] + d, 2),
                 'next': A['candidates'][name][0]}
assert bank['A']['first'] == ['river', 0.414] and bank['A']['second'] == ['fisherman', 0.226]
assert bank['B']['first'] == ['cheque', 0.4] and bank['B']['second'] == ['deposited', 0.261]
assert bank['A']['m'] == [1.89, 0.07] and bank['A']['e_new'] == [2.21, 0.4, 0.68, 0.98] and bank['B']['e_new'] == [0.6, 2.72, 0.5, 0.44]
assert bank['A']['receiver'] == 6 and bank['A']['e'] == [0.7, 0.7, 0.1, 0.8]
model = {'d_model': A['d_model'], 'd_k': DK, 'd_v': A['d_v'], 'axes': A['axes']['e'], 'bank_row': A['tok_emb']['bank']}

# the forgotten mask (Lecture 13, s13): row 5 of sentence A ("the", whose target is river) with the mask off
words = A['sentences']['river']
E0, Q, K, V = rows(words)
ROW = 4
leak_w = softmax(Q[ROW] @ K.T / math.sqrt(DK))
leak = {'words': words, 'row': ROW, 'target': words[ROW + 1], 'weights': r(leak_w, 3),
        'future': r(leak_w[ROW + 1:].sum(), 3), 'on_target': r(leak_w[ROW + 1], 3)}
assert round(leak['future'], 2) == 0.67 and leak['on_target'] == 0.477 and leak['target'] == 'river'

# scaling (Lecture 13, s12): d_k = 64, raw scores [-8, 8]
SC_DK, RAW = 64, [-8.0, 8.0]
scale = {'d_k': SC_DK, 'sqrt': math.sqrt(SC_DK), 'raw': RAW, 'scaled': [v / math.sqrt(SC_DK) for v in RAW],
         'w_raw': [float(f'{x:.3g}') for x in softmax(RAW)], 'w_scaled': r(softmax(np.array(RAW) / math.sqrt(SC_DK)), 4)}
assert abs(scale['w_raw'][0] - 1.13e-7) < 1e-9 and scale['w_scaled'][0] == 0.1192

# the masked softmax (Lecture 13, the P02 cheat sheet): [1, 2, 3, 4] + [0, 0, -inf, -inf]
SCORES, MASK = [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, -math.inf, -math.inf]
mask = {'scores': SCORES, 'masked': [i for i, v in enumerate(MASK) if v < 0], 'weights': r(softmax(np.add(SCORES, MASK)), 3)}
assert mask['weights'] == [0.269, 0.731, 0.0, 0.0]

# ---------------------------------------------------------------- Lecture 14: TinyStories (recorded)
# Recorded from figures/multihead/capstone-evidence.json ("scores" and "costs") in the attention repo, origin/main;
# the same table is part3.html#s05-cap-scores. Mean and SD of three seeds (11, 29, 47), 6,000 updates each.
RECORDED_TINYSTORIES = [
    # name, short name, test loss (nats), its SD, perplexity, its SD, parameters
    ['Fixed-window MLP', 'MLP', 3.943, 0.008, 51.59, 0.40, 2332832],
    ['One-head attention', 'one head', 3.445, 0.011, 31.34, 0.33, 1321120],
    ['Four-head attention', 'four heads', 3.360, 0.021, 28.78, 0.61, 1321120],
]
tiny = [dict(zip(['name', 'short', 'loss', 'loss_sd', 'ppl', 'ppl_sd', 'params'], row)) for row in RECORDED_TINYSTORIES]
for t in tiny:            # perplexity is exp(test loss); the table averages per-seed perplexities, so allow a little slack
    assert abs(math.exp(t['loss']) - t['ppl']) / t['ppl'] < 0.01
assert tiny[2]['ppl'] < tiny[1]['ppl'] < tiny[0]['ppl'] and tiny[2]['params'] < tiny[0]['params']
ppl = {'models': tiny, 'gain_four_vs_one': r(100 * (1 - tiny[2]['ppl'] / tiny[1]['ppl']), 1),
       'quarter': {'p': 0.25, 'ppl': r(math.exp(-math.log(0.25)), 4)},
       'context': 64, 'd_model': 64, 'heads': 4, 'head_width': 16, 'vocab': 4000, 'seeds': 3}
assert ppl['quarter']['ppl'] == 4.0 and round(ppl['gain_four_vs_one'], 0) == 8

# ---------------------------------------------------------------- Lecture 15: three families, one sentence
families = {'source': ['Raghav', 'goes', 'to', 'school', 'in', 'Delhi'],
            'target': ['राघव', 'दिल्ली', 'में', 'स्कूल', 'जाता', 'है।']}
assert len(families['source']) == len(families['target']) == 6

OUT.write_text(json.dumps({'source': 'scripts/sci/p4.py',
                           'names': {'name': NAME, 'w': W, 'pairs': pairs, 'focus': FOCUS},
                           'aab': aab, 'temp': temp, 'bank': bank, 'model': model, 'leak': leak, 'scale': scale,
                           'mask': mask, 'recorded': {'tinystories': ppl}, 'families': families},
                          ensure_ascii=False, indent=1) + '\n')
print('aab', aab)
print('temp', temp['p_top'], [VOC[i] for i in temp['order']])
for k, b in bank.items():
    print('bank', k, b['first'], b['second'], b['m'], b['e_new'])
print('leak', leak['future'], leak['on_target'])
print('scale', scale)
print('mask', mask)
print('ppl', ppl['gain_four_vs_one'])
