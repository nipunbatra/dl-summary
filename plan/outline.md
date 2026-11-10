# Outline: the summary lecture (Lecture 27)

Read `AUTHORING.md` first. This file says what each part contains. Slide ids in **bold** are fixed. The builds are a
plan: you may move a click from one slide to another, but stay inside the part's click budget.

The opening (`parts/00-opening.js`, done) set up the recipe on Lecture 4's XOR network: Data, Model, Output, Loss,
Update, At test time. Every part answers: which rows did these lectures change, and how?

| Part | Section id | File | Data | Lectures | Rows changed | Clicks (max) |
|---|---|---|---|---|---|---|
| 1 Foundations | `found` | `parts/01-foundations.js` | `p1` | 1 to 4 | Model, Output, Loss | 22 |
| 2 Gradients and optimization | `grad` | `parts/02-gradients.js` | `p2` | 5 to 9 | Update | 26 |
| 3 Generalization | `gen` | `parts/03-generalization.js` | `p3` | 10 | Data, Loss, At test time | 18 |
| 4 Attention and language | `attn` | `parts/04-attention.js` | `p4` | 11 to 15 | Data, Model, Output, At test time | 24 |
| 5 Vision and language | `vision` | `parts/05-vision.js` | `p5` | 16 to 22 | all but Update | 26 |
| 6 Generative modelling | `genm` | `parts/06-generative.js` | `p6` | 23 to 25 | all but Update | 24 |
| 7 Object detection | `det` | `parts/07-detection.js` | `p7` | 26 | Data, Output, Loss, At test time | 18 |

The numbers below are the lectures' own, copied from the digests so you know what to aim at. They still have to come
from `data/pN.json` on the slide (AUTHORING.md, section 5).

---

## Part 1: Foundations (Lectures 1 to 4)

Digest: `plan/digests/01-foundations.md` (conventions at line 15; L1 26; L2 98; L3 169; L4 247).

`partBreak(1, 'Where the loss comes from', …)`. Lead: every loss in this course was read off a probability model; this
part recalls how, and builds the first network. Steps: pick a distribution for the output and write the likelihood of
the data; take the negative log, and that is the loss; a prior on the weights adds a penalty; stack linear layers and
ReLUs to get a network. Aim (`line`): by the end you can say where cross-entropy and squared error come from.

1. **`nll-coin`** (3 clicks), `chap('1.1','Loss: from a likelihood')`, `from: 1`. The ten tosses HHTTTHHTTT. Live
   figure `p1-coin`: a slider on θ, the negative log-likelihood curve with a moving dot and a readout; the minimum is
   at 0.4 (NLL 6.73), against 8.65 at θ = 0.7. Click 1 asks which θ the class would bet on; later clicks give the
   likelihood, its negative log, and the minimum.
2. **`nll-loss`** (4), `chap('1.2','Loss: cross-entropy')`, `from: 2`. Binary: y = 1, p = 0.8 gives −ln 0.8 = 0.22.
   Three classes: logits (2, 1, −0.5) → softmax → p₁ = 0.690 → cross-entropy 0.37. Last click: the gradient at the
   logits is p − y, with the numbers.
3. **`dist-loss`** (3), `chap('1.3','Output: a distribution')`, `from: 3`. One table, a row per click or two: kind of
   target → distribution → output layer → loss (Bernoulli/sigmoid/binary cross-entropy; categorical/softmax/
   cross-entropy; Gaussian/identity/squared error). Squared error is the Gaussian's negative log-likelihood.
4. **`map-prior`** (3), `chap('1.4','Loss: a prior adds a penalty')`, `from: 2` (or the lecture the digest says).
   The four-point fit: maximum likelihood (2.07, 2.96) against the prior of width τ = 0.5, (1.79, 2.62). Live figure
   `p1-map`: a slider on τ moves the fitted line and the estimate. The Gaussian prior is an L2 penalty.
5. **`mlp-xor`** (3), `chap('1.5','Model: layers and ReLUs')`, `from: 4`. Why one line fails on XOR and two ReLU units
   succeed. Then sin x fitted with 2, 5 and 15 hinges (errors 0.437, 0.134, 0.019): buttons `p1-hinges` choose the
   number of hinges and redraw the fit.
6. **`check-1`** (2). "A 10-class classifier reports a loss of 2.30 at its first step. Is something broken?" Answer:
   ln 10 = 2.30 is the cross-entropy of ten equal guesses, so the network starts where it should.
7. `recap(1, …, {card: RECIPES.mlp, title: 'MLP classifier'})`. Next: how the gradient of that loss is computed.

## Part 2: Gradients and optimization (Lectures 5 to 9)

Digest: `plan/digests/02-gradients-optimization-generalization.md` (L5 13; L6 82; L7 165; L8 245; L9 325; stop at 410).

`partBreak(2, 'How the weights move', …)`. Lead: the Update row needs a gradient and a step size; five lectures were
about getting both. Steps: backpropagation gives the gradient; a mini-batch gives a cheap, noisy one; momentum and
Adam shape the step; a schedule shrinks it.

1. **`bp-graph`** (5), `chap('2.1','Update: the gradient by backprop')`, `from: 6`. ŷ = wx + b with
   (w, x, b, y) = (2, 3, 1, 10). An SVG graph, one node per click (`gat`): forward ŷ = 7, loss (ŷ − y)² = 9; backward
   dL/dŷ = −6, g_b = −6, g_w = −18.
2. **`bp-update`** (4), `chap('2.2','Update: one step')`, `from: 6`. Minimal PyTorch, a line per click beside the
   numbers each line produces: η = 0.01 moves w 2 → 2.18 and b 1 → 1.06; the prediction goes 7 → 7.60, the loss
   9 → 5.76.
3. **`sgd`** (3), `chap('2.3','Update: a noisy gradient')`, `from: 5`. The four-point fit (−1, −1), (0, 1), (1, 3),
   (2, 6) from θ₀ = (b, w) = (2, −1), η = 0.1: the full gradient is (−0.75, −4.5) and the loss falls 7.125 → 5.215;
   one unlucky point raises it to 7.205. Live figure `p2-sgd` (buttons: all four points, or each single point) shows
   the gradient arrow and the loss after the step.
4. **`opt-play`** (2), `chap('2.4','Update: momentum and Adam')`, `from: 7`. Live figure `p2-opt`: the ravine
   L = ½(θ₁ − 1)² + 50(θ₂ + 0.5)² from θ₀ = (−1.4, 0.35); buttons gradient descent, momentum, RMSProp, Adam; a slider
   on the number of steps; a loss readout. Use the lecture's settings (digest L7). Click 2 says in words what each
   method changed.
5. **`vjp`** (3), `chap('2.5','Update: autograd on vectors')`, `from: 8`. Backprop through a layer multiplies by Jᵀ:
   J = [[1, 1], [2, −1]], g = (3, 4) gives Jᵀg = (11, −1). Autograd never builds J.
6. **`logits`** (3), `chap('2.6','Output: keep the logits')`, `from: 9`. softmax(2, 1, 0) = (0.665, 0.245, 0.090);
   (1000, 999, 998) overflows to NaN when computed directly; subtracting the maximum gives the same three
   probabilities. So the loss takes logits.
7. **`schedule`** (2), `chap('2.7','Update: a schedule for the step size')`, `from: 9`. Live figure `p2-sched`: cosine
   schedule with η₀ = 0.1, η_min = 0.001, T = 100; a slider on the step shows η (0.0505 at step 50).
8. **`check-2`** (2). "The loss went up after one SGD step. Is the code broken?" Answer: a mini-batch gradient points
   downhill for its own points only (the unlucky draw on `sgd`); judge the trend over many steps, and lower η if it
   keeps rising.
9. `recap(2, …, {card: {...RECIPES.mlp, step: '<backprop on a mini-batch; Adam and a schedule set the step>'}})`.
   Next: a low training loss is the wrong thing to celebrate.

## Part 3: Generalization (Lecture 10)

Digest: `plan/digests/02-gradients-optimization-generalization.md`, from line 410.

`partBreak(3, 'Doing well on data you have not seen', …)`. Steps: measure on held-out data; penalise large weights;
add noise to training (dropout, augmentation, label smoothing); choose with validation, report on test once.

1. **`overfit`** (3), `chap('3.1','At test time: the gap')`, `from: 10`. Ten points from y = sin(πx) + noise (the
   lecture's seed 21 and noise 0.25); a degree 9 polynomial: training error 0, validation error 1.70, ‖w‖ = 465.
   Port the lecture's data generation (`git -C /Users/nipun/git/dl-teaching show origin/master:<path>`) so the
   numbers match.
2. **`reg-play`** (2), `chap('3.2','Loss: a penalty on the weights')`. Live figure `p3-lambda`: a slider over the λ
   values of the lecture's sweep redraws the fit and reports training error, validation error and ‖w‖
   (best at λ = 10⁻³: validation 0.065).
3. **`regs`** (4), `chap('3.3','Data and Loss: four regularisers')`. One per click, each with its numbers: weight
   decay; dropout h = (2, 4, 6, 8), mask (1, 0, 1, 0), keep 0.5 → (4, 0, 12, 0); label smoothing K = 5, ε = 0.1 →
   (0.02, 0.92, 0.02, 0.02, 0.02); augmentation (crop and flip).
4. **`regs-cifar`** (2), `chap('3.4','At test time: what helped')`. Bars of the lecture's CIFAR-10 run (2,000
   training images): baseline 43.0 %, weight decay 43.0 %, dropout 46.8 %, label smoothing 44.8 %, crop and flip
   54.6 %. Click 1 asks which the class expects to help most.
5. **`splits`** (3), `chap('3.5','Data: three splits')`. A flowchart: train fits θ, validation chooses λ and when to
   stop, test is read once.
6. **`check-3`** (2). Training loss keeps falling while validation loss has started to rise: what do you do? Use
   only remedies the lecture taught (check the digest; early stopping only if it is there).
7. `recap(3, …, {card: {...RECIPES.mlp, data: '…splits and augmentation…', loss: '…plus a penalty…', test: '…dropout off; test read once…'}})`.
   Next: the model has been an MLP so far; language needs a model that can look back.

## Part 4: Attention and language (Lectures 11 to 15)

Digest: `plan/digests/03-attention-language.md` (L11 26; L12 123; L13 203; L14 286; L15 379).

`partBreak(4, 'Predicting the next token', …)`. Steps: turn text into tokens and predict the next one; let each token
look at the others (self-attention); mask the future, add heads, stack blocks; generate by sampling one token at a
time.

1. **`next-token`** (4), `chap('4.1','Data and Output: the next token')`, `from: 11`. The name "aabid": context "aab",
   target "i", p(i | aab) = 0.117, loss −ln 0.117 = 2.15. Live figure `p4-temp`: a temperature slider reshapes the
   bar chart of next-character probabilities (p(n) = 0.342, 0.161, 0.112 at T = 0.5, 1, 1.5). Sampling is the
   At test time row.
2. **`attn-bank`** (3), `chap('4.2','Model: attention')`, `from: 12`. The two "bank" sentences. Live figure `p4-bank`:
   buttons for sentence A and B show bank's attention weights as bars over the words (A: river 0.414, fisherman
   0.226; B: cheque 0.400, deposited 0.261). Same word, different mixture.
3. **`attn-eq`** (5), `chap('4.3','Model: one attention layer')`, `from: 14`. The lecture's two lines of code, built a
   piece per click beside what each piece does: scores QKᵀ; divide by √d_k (scores [−8, 8] give 1.13 × 10⁻⁷ unscaled,
   0.1192 scaled); add the mask (→ [0.269, 0.731, 0, 0]); softmax and mix V; W_O and the residual.
4. **`attn-ppl`** (2), `chap('4.4','At test time: perplexity')`, `from: 14`. Bars: TinyStories perplexity for the MLP
   (51.59, 2,332,832 parameters), one head (31.34), four heads (28.78, 1,321,120 parameters). Fewer parameters,
   lower perplexity.
5. **`families`** (3), `chap('4.5','Model: three families')`, `from: 15`. Decoder-only, encoder, encoder-decoder:
   what each sees, what it is trained to predict, one use each.
6. **`check-4`** (2). "You forget the mask. The training loss falls to almost zero. Why is that bad news?" Answer:
   every position can read the token it is asked to predict; at test time that token does not exist yet.
7. `recap(4, …, {card: RECIPES.gpt, title: 'Transformer language model'})`. Next: the same block, fed image patches.

## Part 5: Vision and language (Lectures 16 to 22)

Digests: `plan/digests/04-vision-transformers.md` (L16 12; L17 112; L18 196; L19 291; note on L22 395) and
`plan/digests/05-cbm-kl-vi.md` (L20 13; L21 80; stop at 145).

`partBreak(5, 'Images as tokens, labels as language', …)`. Steps: cut the image into patches and reuse the
transformer; train an image encoder and a text encoder to agree; hand the image tokens to a language model; make the
model reason through concepts a person can check.

1. **`vit`** (4), `chap('5.1','Data and Model: patches as tokens')`, `from: 16`. The Newfoundland photo
   (`figures/p5-newfoundland.jpg`, copied with
   `git -C /Users/nipun/git/attention show origin/main:figures/vision1/newfoundland_31.jpg`; Oxford-IIIT Pet,
   CC BY-SA 4.0, credit in `source`). Live figure `p5-patch`: buttons for patch size 32, 16, 8 draw the grid and
   give 49, 196, 784 tokens. Then 768 → 192, plus [CLS] makes 197 × 192, 12 blocks, 1,000 logits, 95.73 %.
   Mention in notes that Lecture 22 built a ViT live with AI tools (recording on the schedule).
2. **`clip`** (4), `chap('5.2','Loss: match photos to captions')`, `from: 17`. The cat, dog, car batch: the 3 × 3
   cosine matrix, a softmax along each row, the loss 0.56 against ln 3 = 1.10 for a model that cannot tell them
   apart; 0.12 after 100 steps. Live figure `p5-tau`: a slider on the temperature τ recomputes the row
   probabilities and the loss from the matrix.
3. **`clip-use`** (2), `chap('5.3','At test time: classes written as captions')`, `from: 17`. Write each class as a
   caption, encode, pick the best match. Never write "zero-shot".
4. **`vlm`** (3), `chap('5.4','Model: an image in front of a language model')`, `from: 18`. Flowchart: vision encoder →
   projector → language model. 196 + 13 = 209 tokens against 8 + 13 = 21 after compression. Training on
   "A black dog.": probabilities 0.20, 0.40, 0.50, 0.80, 0.90 give a mean loss of 0.709.
5. **`cbm`** (3), `chap('5.5','Output: concepts you can correct')`, `from: 20`. x → concepts ĉ → class ŷ. Live figure
   `p5-cbm`: toggle "stripes" between the predicted value and the corrected one and watch the class probabilities
   move from (0.350, 0.522, 0.129) to zebra 0.95. Then the CUB accuracies (standard 73.3 %, joint 73.5 %,
   sequential 68.9 %, independent 49.9 %; with concepts corrected: see the digest).
6. **`check-5`** (2). "Which of ViT, CLIP, a VLM and a concept bottleneck model need class labels to train?" Answer:
   the ViT and the concept bottleneck model (which also needs concept labels); CLIP needs captions; a VLM needs text
   about images.
7. `recap(5, …, {card: RECIPES.clip, title: 'CLIP'})`. Next: every model so far predicted a label or a token; the next
   ones model the data itself.

## Part 6: Generative modelling (Lectures 23 to 25)

Digests: `plan/digests/05-cbm-kl-vi.md` (L23 145; L24 230; odd things 350) and `plan/digests/06-vae-detection.md`
(L25, lines 9 to 130). The source decks are in `/Users/nipun/git/generative/lectures/lecture1` (KL and VI; schedule
Lectures 23 and 24) and `lecture5` (VAE; schedule Lecture 25); their `data/*.json` hold many of the numbers, and you
may read them (not write). Deep links: base `https://nipunbatra.github.io/generative/lectures/lecture1/` with
`#kl-lesson/2`, `#elbo-goal/3`, `#elbo-drift/3`, `#reparam/3`, `#now-optimise/2`, `#bnn-vs-ref/2`; base
`https://nipunbatra.github.io/generative/lectures/lecture5/` with `#ae-loss/3`, `#fix-loss/4`, `#vi-kl/5`,
`#maths-chain/5`, `#summary/2`. Use them as `from: [23, '#kl-lesson/2']` if `from()` resolves Lecture 23's slides
link to that base (check `data/course.json`), otherwise pass `{href, label: 'From Lecture 23'}`.

`partBreak(6, 'Learning a distribution', …)`. Steps: measure how far one distribution is from another (KL); when the
posterior is out of reach, fit a simpler q by raising the ELBO; sample through the reparameterisation so gradients
flow; put an encoder and a decoder around a latent code (VAE).

1. **`kl`** (3), `chap('6.1','Loss: KL divergence')`. The weather example: p = (½, ¼, ¼), q = (¼, ¼, ½); H(p) = 1.5,
   H(p, q) = 1.75, KL = 0.25 bits. Then the tie to Part 1: a classifier that gives the true class 0.60 pays
   −ln 0.6 = 0.51 nats, and cross-entropy is a KL to the one-hot target. A live figure `p6-kl` (a slider that moves
   q towards p) is welcome if it fits the budget.
2. **`elbo`** (4), `chap('6.2','Loss: the ELBO')`. The coin: Beta(5, 3) prior, 17 heads in 30 tosses,
   ln p(D) = −21.40. Live figure `p6-elbo`: move q from Beta(10, 10) (ELBO −22.02, gap 0.62) to Beta(22, 16)
   (ELBO −21.40, gap 0): the gap is KL(q ‖ posterior). ln p(D) = ELBO + KL.
3. **`reparam`** (3), `chap('6.3','Update: gradients through a sample')`. θ = μ + σε with ε = −0.75, μ = 0, σ = 2
   gives θ = −1.5; the randomness sits in ε, so μ and σ get gradients.
4. **`bnn`** (2), `chap('6.4','At test time: uncertainty')`. The tiny Bayesian network (7 weights, 17 points with a
   gap): ELBO −1,235 → −7.18 = 8.10 − 15.28; the predictive band widens in the gap.
5. **`ae-vae`** (3), `chap('6.5','Model: encoder, code, decoder')`. The autoencoder on MNIST test digit 3 (#90):
   784 → 400 → 2 → 400 → 784, squared error 26.3; the VAE's encoder gives μ = (−0.211, 0.070), σ = (0.013, 0.010).
6. **`vae-loss`** (4), `chap('6.6','Loss: reconstruction plus KL')`. The lecture's code lines, one per click, beside
   251.31 − 7.94 = 243.37 (ELBO), ln p(x) = 244.07, gap 0.70. Live figure `p6-beta`: the β sweep (KL per image
   68.3 at β = 0, 9.0 at β = 1, 0.00 at β = 256) with the decoded digits or the latent picture if the lecture's
   data has them.
7. **`check-6`** (2). "With β = 256 the KL term is 0.00. What do the generated digits look like, and why?" Check the
   answer against the digest (the encoder ignores x, so the decoder draws the same average digit).
8. `recap(6, …, {card: RECIPES.vae, title: 'Variational autoencoder'})`. Next: one more change of output, from one
   label to a list of objects.

## Part 7: Object detection (Lecture 26)

Digest: `plan/digests/06-vae-detection.md` (L26, lines 131 to 264; reusable engineering from 265). The source deck is
`/Users/nipun/git/object-detection` (read only): its `figures/cat-dog.jpg` (copy to `figures/p7-courtyard.jpg`, credit
in `source` as the deck credits it) and its `data/*.json` hold the boxes, scores and curves. Deep links: base
`https://nipunbatra.github.io/object-detection/` with `#iou-union/3`, `#target-photo/3`, `#nms-dog/4`, `#ap-area/3`,
`#pipeline/3`.

`partBreak(7, 'From one label to a list of objects', …)`. Steps: the output is a list of boxes with a class and a
score; IoU says how well two boxes agree; every grid cell predicts, and the loss compares cell by cell; at test time
keep the confident boxes and drop the duplicates.

1. **`det-list`** (3), `chap('7.1','Output: a list of boxes')`. The courtyard photo with the detector's four boxes
   drawn on it: cat 0.92, potted plant 0.89, dog 0.89, potted plant 0.29. Click 1 shows the photo and asks what a
   classifier would say.
2. **`iou`** (2), `chap('7.2','At test time: do two boxes agree?')`. Live figure `p7-iou`: a slider slides a guess box
   across the dog's box (407, 83, 768, 588) and reports intersection, union and IoU; the lecture's guess
   (467, 123, 828, 628) gives 0.62.
3. **`det-loss`** (3), `chap('7.3','Loss: one target per grid cell')`. The 3 × 2 grid of 320-pixel cells over the
   photo; the dog's cell costs 1.35, all six cells 3.38, and 0.002 after 300 Adam steps. The deck's loss is squared
   error; say in notes that real detectors use cross-entropy terms.
4. **`nms`** (3), `chap('7.4','At test time: threshold, then NMS')`. Live figure `p7-nms`: 12,600 predictions, 31
   pass a score of 0.25, 4 survive NMS at IoU 0.7. Controls: the score threshold and NMS on or off, drawn on the
   photo with the count.
5. **`det-eval`** (3), `chap('7.5','At test time: precision, recall, AP')`. Live figure `p7-pr`: a slider on the
   score threshold moves a point along the precision and recall curve (precision 0.5, recall 0.8 at the lecture's
   point; AP ≈ 0.66 is the area).
6. **`check-7`** (2). A question that needs IoU and NMS together, with numbers from the data, for example: two boxes
   on the dog overlap with IoU 0.65 and NMS removes pairs above 0.7; how many dogs are reported, and how does that
   show up in precision?
7. `recap(7, …, {card: RECIPES.det, title: 'Object detector'})`. Next: all seven parts side by side.

---

## Wrap (Part 8, section `wrap`, `parts/08-wrap.js`, data `p8`)

Written by the lead author, not by a part author. It links back to the fixed slide ids above.

- `threads-ce`: one −ln p curve with a dot from every part (0.8 → 0.22, 0.690 → 0.37, 0.6 → 0.51, 0.117 → 2.15,
  0.20 → 1.61).
- `threads-attn`: the one attention block, reused by the language model, the ViT, CLIP and the VLM.
- `parts-rows`: the seven parts against the six rows (from `PARTS`).
- `master-a`, `master-b`: the ten models against Data, Model, Output, Loss, At test time (from `RECIPES`).
- `pipeline`: the overall flowchart.
- `summary`: the lecture in three statements.
- Appendix: every lecture with its links (from `data/course.json`).
