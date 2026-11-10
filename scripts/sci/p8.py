"""The wrap-up of the summary lecture: five losses the parts showed, placed on one curve of -ln p.

    ~/base/bin/python scripts/sci/p8.py      writes data/p8.json

Nothing new is computed here. The script reads the numbers the parts already wrote (data/p1.json, p4.json,
p5.json, p6.json), checks that each loss is -ln of its probability, and writes them beside the curve.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'p8.json'


def load(name):
    f = ROOT / 'data' / f'{name}.json'
    if not f.exists():
        raise SystemExit(f'{f} is missing: run scripts/sci/{name}.py first')
    return json.loads(f.read_text())


def get(d, name, *keys):
    for k in keys:
        try:
            d = d[k]
        except (KeyError, IndexError, TypeError):
            raise SystemExit(f'data/{name}.json has no {"/".join(map(str, keys))}')
    return d


p1, p4, p5, p6 = (load(n) for n in ('p1', 'p4', 'p5', 'p6'))
g1, g4, g5, g6 = (lambda *k: get(p1, 'p1', *k)), (lambda *k: get(p4, 'p4', *k)), (lambda *k: get(p5, 'p5', *k)), (lambda *k: get(p6, 'p6', *k))

# lectures, the model (an id in kit.js MODELS, or None), what was observed, its probability, its loss, this deck's slide
points = [
    dict(lec=[2], model=None, name='Logistic classifier', what='the label y = 1', p=g1('ce', 'bin', 'p'), loss=g1('ce', 'bin', 'loss'), slide='nll-loss/2'),
    dict(lec=[11], model='charlm', name='Character language model', what=f'the letter {g4("aab", "target")} after "{g4("aab", "context")}"',
         p=g4('aab', 'p_target'), loss=g4('aab', 'loss'), slide='next-token/3'),
    dict(lec=[17], model='clip', name='CLIP', what=f'the {g5("clip", "names")[0]} photo\'s own caption', p=g5('clip', 'P')[0][0], loss=g5('clip', 'pen_rows')[0], slide='clip/3'),
    dict(lec=[19], model='vlm', name='Vision-language model', what=f'the first answer token, "{g5("vlm", "answer")[0].strip()}"',
         p=g5('vlm', 'probs')[0], loss=g5('vlm', 'pen')[0], slide='vlm/3'),
    dict(lec=[23, 24], model=None, name='Classifier, read as a KL', what=f'the class {g6("classify", "classes")[g6("classify", "label")]}',
         p=g6('classify', 'p_true'), loss=g6('classify', 'loss_nats'), slide='kl/3'),
]
for q in points:
    # the parts store p rounded, so the stored loss can differ from -ln p in the fourth decimal
    if abs(-math.log(q['p']) - q['loss']) > 2e-3:
        raise SystemExit(f'{q["name"]}: loss {q["loss"]} is not -ln {q["p"]} = {-math.log(q["p"]):.4f}')
    q['p'], q['loss'] = round(q['p'], 4), round(q['loss'], 4)

worst = max(range(len(points)), key=lambda i: points[i]['loss'])
curve = [[round(i / 100, 2), round(-math.log(i / 100), 4)] for i in range(5, 101)]
v16 = g5('vit', 'sizes', str(g5('vit', 'patch')))
out = {
    'source': 'scripts/sci/p8.py',
    'points': points, 'worst': worst, 'curve': curve, 'y_max': 3.0,
    'vit': {'img': g5('vit', 'img'), 'patch': g5('vit', 'patch'), 'side': v16['side'], 'patches': v16['tokens'], 'tokens': v16['with_cls']},
    'vlm': {'rows': g5('vlm', 'rows'), 'd_v': g5('vlm', 'd_v'), 'd_lm': g5('vlm', 'd_lm')},
}
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
print(f'wrote {OUT.relative_to(ROOT)}: {len(points)} points, worst = {points[worst]["name"]} ({points[worst]["loss"]})')
