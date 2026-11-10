# Digest 06: Lecture 25 (Autoencoder and VAE) and Lecture 26 (Object Detection)

Sources read (read only): `/Users/nipun/git/generative/lectures/lecture5` (branch `rebuild/lecture1-story`; its `content.js` is byte-identical to the live one) and `/Users/nipun/git/object-detection` (main; `content.js` byte-identical to live). Slide text below was taken from the rendered slides (the decks' `content.js` evaluated with `data/*.json`), so every number is the one shown in class.

Numbering trap: the VAE deck calls itself "Lecture 5" (`meta.number:5`) and calls the KL/VI lectures "Lecture 1"; in the course these are Lecture 25 and Lectures 23 and 24. The detection deck has `meta.number:22`, its title slide says "Lecture 22", its ladder slide says segmentation is "(Lecture 23)", and its `index.html` "Course" link points at `schedule.html#lecture-22`; on the schedule it is Lecture 26.

---

## 1. Lecture 25: Autoencoder and VAE

Main route: 61 slides, 85 minutes, six parts and 25 numbered sections; 24 optional slides (`O()`), not taught. Live: https://nipunbatra.github.io/generative/lectures/lecture5/

**Aim.** Train an autoencoder on one running digit, show that it cannot generate, turn it into a VAE with three fixes, and show that the fixes' loss (reconstruction plus KL) is minus Lecture 23 and 24's ELBO for one image.

### Running example

- MNIST test digit #90, a 3 ("our 3"); 784 pixels in [0, 1]; image key `example` in Lecture 2's `digits.json`.
- Autoencoder: encoder `g_φ` 784 → 400 (ReLU) → 2, decoder `f_θ` 2 → 400 → 784 then sigmoid. Code size 2. Batch 128, Adam 1e-3, 30 epochs, 469 steps per epoch, 14,070 updates.
- Our 3 through the trained 2-number autoencoder: squared error 26.3 over 784 pixels (`ae-loss`). Untrained network: about 185 (`ae-untrained`). Test error by code size: 2 numbers 31.1, 8: 12.9, 32: 3.6, 128: 1.2 (`code-size`).
- VAE: encoder ends in 4 numbers (μ and log σ² for 2 code numbers); pixel noise σ = 0.2, so the reconstruction weight is 1/(2 · 0.2²) = 12.5 and log p(x | z) carries the constant +541.4 = −784 log(0.2 √(2π)). A 16-number VAE made the fakes on `realfake` and the sharper samples.
- Encoder's q for our 3: μ = (−0.211, 0.070), σ = (0.013, 0.010). Exact posterior: a patch around (−0.23, 0.06), about 0.018 wide.
- ELBO numbers for our 3 with the encoder's q: E_q[log p(x | z)] = 251.31, KL(q ‖ N(0, I)) = 7.94, ELBO = 243.37, log p(x) = 244.07 (grid), gap = KL to the posterior = 0.70. A q fitted by VI to this image alone reaches 244.05 (`opt-q3`, optional).
- Codes A, B, C (in the posterior patch, a prior draw, among the 1s): log p(x | z) = 252, −360, −505.
- β sweep: KL per image 68.3 at β = 0, 9.0 at β = 1, 0.00 at β = 256 (posterior collapse).

### Ideas in teaching order (deck's wording)

1. Part 1: generative models share one recipe, "random numbers in, a learned network, a sample out"; x is the data, y a label; a classifier learns p(y | x), a generative model p(x), which can "score and sample"; the pixel average is "the blur in the middle of many real 3s", not a sample.
2. Part 2: "An autoencoder: encoder, short code, decoder"; the loss is "how far each rebuilt pixel is from the real one"; codes cluster by digit with no labels, then "drift out"; on 200 digits "the codes fly apart" (overfitting).
3. Part 3: "Keep the decoder, feed it random numbers": N(0, I) draws miss the real codes, and the best rebuilder (128 numbers) gives static; sampling the real codes works only with a second fitted model.
4. Part 4, three fixes: "Fix 1: choose where the codes live, before training" (z ~ N(0, I)); "Fix 2: send each image to a small cloud of codes, not a point" (z = μ + σ ⊙ ε, tool ①); "Fix 3: a penalty that pulls every cloud towards N(0, I)" (a KL, tool ②).
5. "The three fixes give a loss: reconstruction plus KL"; "Two terms pull in opposite directions".
6. Part 5.1 to 5.3: "Same problem as Lecture 1: the same picture, new names" (θ → z, 𝒟 → x); the model's story z → x; the likelihood scores a code; Bayes' rule backwards; the evidence p(x) is "Lecture 1's wall".
7. 5.4: "Variational inference: a guide q close to the posterior": KL(q ‖ p(z | x)) = reconstruction term + KL to the prior + log p(x); log p(x) does not depend on q, so the VAE loss is −ELBO; "The guide is a neural network: the encoder" (amortised inference).
8. 5.5: "Put in the model's choices": Gaussian pixels with σ = 0.2 turn the log-likelihood into squared error / (2 · 0.2²), the KL becomes a KL between Gaussians; three code changes to the autoencoder.
9. 5.6 to 5.8: the ELBO explorer; "Now sample: z ~ N(0, I), then decode"; "The codes now fill N(0, I)"; "Put a weight on the KL: β".
10. Part 6: "The whole lecture in five lines of maths", "Where each term of the VAE's loss comes from", "Autoencoder and VAE, side by side".

### Key equations (deck notation)

Colour macros (defined in `shared/deck.js`): `\pcol` teal #0F766E, `\qcol` purple #8B2CDE, `\lcol` red #BE123C, `\ncol` orange #AA4E08, `\dcol` blue #245EDB; plus `\E`, `\KL`, `\N`, `\ELBO`. There is no green macro: the prior is written `\textcolor{#147737}{...}` (helper `PRI()` in content.js). The colours carry two meanings in this deck. On loss slides: teal = target x, purple = prediction x̂, red = loss. On Bayes/ELBO slides (Lecture 1's code): purple = q and the ELBO, orange = likelihood, green = prior, blue = evidence, teal = posterior, red = KL to the posterior.

1. Autoencoder loss (`ae-loss`):
   `\lcol{L(\phi,\theta)}=\frac1n\sum_{i=1}^{n}\ \sum_{j=1}^{784}\big(\pcol{x_{ij}}-\qcol{\hat x_{ij}}\big)^2,\qquad \qcol{\hat x_i}=f_\theta\big(g_\phi(\pcol{x_i})\big)`
2. Part 4's loss (`fix-loss`, underbraces dropped):
   `L(x)=\big\lVert \pcol{x}-\qcol{f_\theta(\mu+\sigma\odot\epsilon)}\big\rVert^2+\KL\big(\qcol{\N(\mu,\sigma^2)}\,\|\,\textcolor{#147737}{\N(0,I)}\big)`
3. Bayes' rule and the evidence (`vae-wall`):
   `\pcol{p(z\mid x)}=\frac{\ncol{p_\theta(x\mid z)}\ \textcolor{#147737}{p(z)}}{\dcol{p_\theta(x)}}`, `\dcol{p_\theta(x)}=\int \ncol{p_\theta(x\mid z)}\ \textcolor{#147737}{p(z)}\,dz`
4. The regrouped KL (`vi-kl`, build 4):
   `\lcol{\KL\big(\qcol{q}\,\|\,\pcol{p(z\mid x)}\big)}=\underbrace{-\E_{z\sim\qcol q}\big[\log\ncol{p_\theta(x\mid z)}\big]}_{\text{reconstruction term}}+\underbrace{\KL\big(\qcol{q}\,\|\,\textcolor{#147737}{p(z)}\big)}_{\text{KL to the prior}}+\underbrace{\dcol{\log p_\theta(x)}}_{\text{the same for every }q}`
5. The budget (`maths-chain` ②, `same-problem`):
   `\dcol{\log p_\theta(x)}=\qcol{\ELBO(x)}+\lcol{\KL\big(q_\phi(z\mid x)\,\|\,p(z\mid x)\big)}`
6. The loss with the model's choices (`vae-loss`, build 4):
   `-\qcol{\ELBO(x)}=\underbrace{\E_{q}\Big[\tfrac{1}{2\cdot 0.2^2}\lVert \pcol{x}-\qcol{f_\theta(z)}\rVert^2\Big]}_{\text{reconstruction term: Fix 2}}+\underbrace{\KL\big(q_\phi\,\|\,\textcolor{#147737}{\N(0,I)}\big)}_{\text{KL term: Fixes 1, 3}}+\textcolor{#68717B}{\text{const}}`

The closed-form KL appears only on the optional `vae-loss-terms`: `\KL=\tfrac12\sum_{k=1}^{2}\big(\mu_k^2+\sigma_k^2-\log\sigma_k^2-1\big)`.

### Worked numbers worth reusing

1. Reconstruction, pixel by pixel (`ae-loss`): missed ink x = 0.99, x̂ = 0.20, (x − x̂)² = 0.63; extra ink 0.00 vs 0.65 gives 0.42; background 0.00 vs 0.00 gives 0; all 784 pixels sum to 26.3. Untrained: 784 × 0.235 ≈ 185 (`ae-untrained`).
2. KL to the prior for one code number (`fix-kl`): N(5, 1) costs 12.5; N(0, 0.01²) costs 4.1; N(0, 1) costs 0; our 3's first number, μ = −0.21, σ = 0.013, costs 3.9 (from ½(μ² + σ² − log σ² − 1)).
3. The likelihood as a score (`sigma`, notes): squared errors 23.1, 72.1, 83.7 times 12.5, plus 541.4, give log p(x | z) = 252, −360, −505; A beats B by e^612.
4. Bayes for three codes (`vae-wall` table): log prior −1.9, −2.5, −7.0; log (prior × likelihood) 250, −362, −512; relative to A: × 1, × e^−613, × e^−762.
5. ELBO and gap (`vi-kl` build 5 expected answer; `elbo-explorer`): −251.31 + 7.94 + 244.07 = 0.70, so ELBO = 251.31 − 7.94 = 243.37 sits 0.70 below log p(x) = 244.07.
6. One VAE step (optional `vae-worked`): ε = (0.5, −1.0), z = μ + σ ⊙ ε = (−0.204, 0.060); squared error 23.18 × 12.5 = 289.8; plus KL 7.94 gives L = 297.7; L − 541.4 = −243.7, close to −ELBO = −243.4.

### Code (verbatim from `vae-code`, split in two)

```python
class VAE(AutoEncoder):          # same decoder
    def __init__(self, z=2):
        super().__init__(z)
        self.encoder[-1] = nn.Linear(400, 2 * z)  # (1)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps       # (2)
        return self.decoder(z), mu, logvar
```

```python
x_hat, mu, logvar = model(x)
recon = ((x_hat - x)**2).sum(1) / (2 * 0.2**2)
kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(1)  # (3)
loss = (recon + kl).mean()
```

The autoencoder's training loop (`ae-step`) is unchanged; its loss line is `loss = ((x_hat - x)** 2).sum(1).mean()`.

### The deck's own recap and summary slides

- `maths-chain` "The whole lecture in five lines of maths", rows ① KL "Lecture 1, 2.4"; ② the budget "Lecture 1, 4.1 · today 5.4"; ③ its two terms; ④ Gaussian pixels "today 5.2 and 2.3"; ⑤ the two tools "Lecture 1, 3.5, 2.6 · today 4.2–4.3". Line 2: "The budget for one image: log p(x) splits into the ELBO and a KL to the posterior. So the ELBO is a lower bound on log p(x)."
- `maths-map` "Where each term of the VAE's loss comes from" (figure `lossMap()`): line 4 "Together they are minus Lecture 1's ELBO for one image. Making the loss smaller pushes up a lower bound on log p(x)."
- `summary` "Autoencoder and VAE, side by side", a table:

| | autoencoder | VAE |
|---|---|---|
| the encoder outputs | a code z: 2 numbers | a cloud: μ and log σ², that is q_φ(z \| x) |
| the decoder gets | z | a draw z = μ + σ ⊙ ε (tool ①) |
| the loss | squared error | squared error / 2σ² + KL to N(0, I) (tool ②) = −ELBO + const |
| random z ~ N(0, I) gives | mostly junk | digits |
| can it score an image? | no | yes, through the ELBO, a lower bound on log p(x) |

  Takeaway (build 2): "A VAE is an autoencoder trained with Lecture 1's objective. The KL to the prior makes the codes fill N(0, I), so random codes decode to digits."
- Also useful: `wishes` (what we saw / what we want, three rows) and `vae-tug` takeaway "A VAE is an autoencoder whose codes are noisy clouds held near N(0, I). To generate, draw z ~ N(0, I) and decode."

### Deep links

- https://nipunbatra.github.io/generative/lectures/lecture5/#ae-loss/3
- https://nipunbatra.github.io/generative/lectures/lecture5/#fix-loss/4
- https://nipunbatra.github.io/generative/lectures/lecture5/#vi-kl/5
- https://nipunbatra.github.io/generative/lectures/lecture5/#maths-chain/5
- https://nipunbatra.github.io/generative/lectures/lecture5/#summary/2

### Connections

- Likelihood to loss: Gaussian pixels make −log p(x | z) the squared error over 2σ² (`vae-loss`, `maths-chain` ④); the deck's expected answer ties it to "Least squares in linear regression: maximum likelihood with Gaussian noise".
- KL and ELBO from Lectures 23 and 24 (the deck's "Lecture 1", linked by `l1()` chips): KL ≥ 0 (L1 2.4), the Gaussian KL "tool ②" (2.6), the reparameterisation trick "tool ①" (3.5), the ELBO and budget (4.1), posterior predictive (1.5), and the evidence as the wall. `same-problem` draws Lecture 1's tiny network (7 weights, 17 points) and the VAE as one picture.
- Bayes' rule with y as the unknown (`gen-disc` notes); compression costs log₂(1/p) bits, "Lecture 1's codes" (`uses`).
- Optional `next`: GANs and diffusion replace the pixel likelihood.

### Exam-style questions

1. Why does the 128-number autoencoder, the best rebuilder, produce static from z ~ N(0, I)? Answer: nothing in its loss places codes near N(0, I); its codes for real digits sit far from where N(0, I) draws land, and the decoder was never trained there (`ae-random`, `overlay`).
2. log p(x) is unknown, so how can minimising the VAE loss find the q closest to the posterior? Answer: KL(q ‖ p(z | x)) = (reconstruction term + KL to the prior) + log p(x), and log p(x) is the same for every q, so minimising the first two terms (minus the ELBO) minimises the KL to the posterior; since that KL ≥ 0, the ELBO is a lower bound on log p(x) (`vi-kl`).
3. For our 3 the encoder gives E_q[log p(x | z)] = 251.31 and KL(q ‖ N(0, I)) = 7.94, and the grid gives log p(x) = 244.07. Find the ELBO, the KL to the posterior, and the value of the training loss without its constant. Answer: ELBO = 251.31 − 7.94 = 243.37; KL to the posterior = 244.07 − 243.37 = 0.70. The trained loss is −ELBO with the constant 541.35 dropped: −243.37 + 541.35 ≈ 298.0 (equivalently 12.5 × E_q[squared error] + KL = 12.5 × 23.20 + 7.94 ≈ 298.0; the single-draw step on `vae-worked` gives 297.7).

---

## 2. Lecture 26: Object Detection, from one label to a set of objects

Main route: 111 core slides, 130 minutes, eight parts plus wrap-up; Appendix A (DETR, `A_()`, 20 slides) and Appendix B (`O()`, 23 slides) are optional. Live: https://nipunbatra.github.io/object-detection/

**Aim.** Answer "How can a network with a fixed-size output return a list whose length changes from photo to photo?" by building YOLO step by step (boxes and IoU, a grid of cells with targets and a loss, NMS, three grids) and then measuring it (TP/FP/FN, precision, recall, AP, mAP).

### Running examples

- Courtyard photo `figures/cat-dog.jpg`, 960 × 640 (sides multiples of 32). Annotation (teal): dog (407, 83, 768, 588), w 361, h 505, centre (587.5, 335.5), normalised (0.612, 0.524, 0.376, 0.789); cat (112, 296, 321, 607), centre (216.5, 451.5). YOLOv8n's four detections: cat 0.921, potted plant 0.890, dog 0.889, potted plant 0.290.
- Teaching grid: 320-pixel cells, 3 × 2, a 2 × 3 × 7 table (object?, b_x, b_y, b_w, b_h, cat, dog). Dog in cell (1, 1), cat in cell (1, 0).
- Real grid: YOLOv8n's 32-pixel cells, 30 × 20 = 600; the dog's cell is (10, 18). Three grids (8, 16, 32 px) give 9,600 + 2,400 + 600 = 12,600 records of 84 numbers (4 box + 80 COCO class scores). 31 boxes pass score ≥ 0.25 (10 dog, 10 cat, 10 plant, 1 small plant from the 16-px grid); NMS leaves 4.
- A guess box (467, 123, 828, 628): the dog box moved 60 px right and 40 px down.
- Evaluation: `figures/dog-park.jpg`, 1344 × 768, five annotated dogs A to E (E is the far dachshund) and eight made-up boxes with scores 0.95, 0.90, 0.80, 0.70, 0.60, 0.45, 0.35, 0.20 (`data/dog-park.json`).

### Ideas in teaching order (deck's wording)

1. Part 1 "Why detect objects?": six jobs (road, crowd, camera trap, cells, parcel, crack), "Six jobs, one kind of answer"; the list is used to count, locate, track and crop; hard because sizes vary 40×, small objects get unsure scores, objects overlap.
2. Part 2 "From one label to a list": a classifier gives "golden retriever, 0.19"; detection wants a set Y = {(c_i, b_i)}; "A network's output has a fixed shape".
3. Part 3 "Boxes and IoU": a box is four numbers (corners, or centre and size, or normalised); IoU = overlap ÷ union, computed in three steps, then `box_iou` for every pair.
4. Part 4 "One box, then many": a box head beside the class head; the target with "?"; "The loss: compare only where it counts"; two animals make one head learn the average box.
5. Part 5.1: sliding windows (336 crops, 29 s for VGG16, best window IoU 0.28): slow, repeated work, the box is only the window.
6. Part 5.2 "YOLO, part 1: the ground-truth table": a person draws boxes; each centre picks a cell; the box in the cell's units; `build_target`.
7. Part 5.3 "predict, compare, train": a CNN writes a table of the same shape; the loss cell by cell; training on one photo; crowds need smaller cells or two template boxes per cell (anchors).
8. Part 5.4 "The real grid": 32-px cells; "a box and 80 class scores"; decode back to pixels; "keep the confident ones" (score ≥ 0.25).
9. Part 6 "NMS": keep the best box, delete same-class boxes with IoU above 0.7, repeat; the threshold is a trade-off; class-aware NMS; "The whole detector at test time".
10. Part 7 "Three grids" (8, 16, 32 px; small from fine, large from coarse) and oriented boxes; Part 8 "Measuring a detector": TP if IoU ≥ 0.5, FP, FN, no TN; precision and recall; PR curve; AP; AP at ten IoU thresholds; mAP.

### Key equations (deck notation)

Colours (CLAUDE.md house rule, every loss slide): teal `\pcol{}` = annotation or target y (made by people; text `kt()`, record class `t`); purple `\qcol{}` = the network's ŷ (text `kp()`, class `p`); red `\lcol{}` = the loss (text `kl()`, class `l`).

1. IoU (`iou-def`): `\mathrm{IoU}(A,B)=\frac{\text{area}(A\cap B)}{\text{area}(A\cup B)}`
2. One record's loss (`loss-y`): `\lcol{\mathcal L}=\textstyle\sum_{i=1}^{7}(\qcol{\hat y_i}-\pcol{y_i})^2=\lcol{0.17}`, and for an empty patch `\lcol{\mathcal L}=(\qcol{\hat y_1}-\pcol{y_1})^2=(\qcol{0.3}-\pcol{0})^2=\lcol{0.09}`
3. The grid loss (`loss-grid`): `\lcol{\mathcal{L}}=\sum_{\text{cells}}\Big[(\qcol{\hat o}-\pcol{o})^2+\pcol{o}\sum_{i=2}^{7}(\qcol{\hat y_i}-\pcol{y_i})^2\Big]`
4. Output squashing (`grid-structure`): `\text{object?},\,b_x,\,b_y,\,\text{cat},\,\text{dog}=\sigma(t)\in(0,1),\qquad b_w,\,b_h=e^{\,t}>0`
5. Precision and recall (`pr-def`): `\text{precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}=\frac{4}{4+4}=0.5`, `\text{recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}=\frac{4}{4+1}=0.8`
6. AP and its averages (`ap-area`, `map-iou`, `map`): `\mathrm{AP}=0.2\times1+0.2\times1+0.2\times0.75+0.2\times0.57`, `\mathrm{AP}_{[.5:.95]}=\tfrac1{10}\sum_t\mathrm{AP}_t\approx0.44`, `\mathrm{mAP}=\frac1C\sum_{c=1}^{C}\mathrm{AP}_c=0.936`

### Worked numbers worth reusing

1. IoU of the dog box A = (407, 83, 768, 588) and the guess B = (467, 123, 828, 628) (`iou-overlap`, `iou-overlap-area`, `iou-union`): left max(407, 467) = 467, top max(83, 123) = 123, right min(768, 828) = 768, bottom min(588, 628) = 588; width 768 − 467 = 301, height 588 − 123 = 465, overlap 301 × 465 = 139,965; each area 361 × 505 = 182,305; union 182,305 + 182,305 − 139,965 = 224,645; IoU = 139,965 / 224,645 = 0.623. Centre distance √(60² + 40²) = 72 px (`compare`).
2. Target record for the dog's cell on the 3 × 2 grid (`dog-box`, `target-photo`): centre (588, 336) in cell (1, 1); b_x = (588 − 320)/320 ≈ 0.84, b_y = (336 − 320)/320 ≈ 0.05, b_w = 361/320 ≈ 1.13, b_h = 505/320 ≈ 1.58; record (1, 0.84, 0.05, 1.13, 1.58, cat 0, dog 1). Cat, cell (1, 0): (1, 0.68, 0.41, 0.65, 0.97, 1, 0). On the 32-px grid (`fine-grid`): 587.5 ÷ 32 ≈ 18.36, 335.5 ÷ 32 ≈ 10.48, cell (10, 18), b = (0.36, 0.48, 11.3, 15.8).
3. Loss at a cell (`align-loss`, untrained toy CNN): ŷ = (0.49, 0.48, 0.48, 1.19, 1.07, 0.52, 0.51) against y = (1, 0.84, 0.05, 1.13, 1.58, 0, 1); squares 0.26, 0.12, 0.19, 0.003, 0.26, 0.27, 0.24; sum 1.35. Empty cell (0, 0): (0.49 − 0)² ≈ 0.24. All six cells: 3.38. After 300 Adam steps on this photo (`train-toy`): 0.002, record (0.98, 0.84, 0.01, 1.13, 1.58, 0.01, 0.99).
4. Decoding (`decode`, `decode-box`): cell (10, 18) writes (0.39, 0.37, 11.3, 16.0), dog 0.87 (σ(1.91)); corner (18 × 32, 10 × 32) = (576, 320); centre (576 + 12, 320 + 12) = (588, 332); size 11.3 × 32 ≈ 362 by 16.0 × 32 = 512; edges (407, 76, 769, 588); IoU with the annotation ≈ 0.98.
5. One NMS round (`nms-dog`): ten dog boxes, scores 0.889 (cell (9, 19)) down to 0.766; keep (9, 19); the other nine have IoU 0.956 to 0.996 with it, all above 0.7, so all nine go. Over all classes 31 boxes become 4 (`nms-all`). Crowd: 112 candidates, 11 kept at 0.7, 13 at 0.95, 9 at 0.0 (`nms-threshold`).
6. Precision, recall, AP on the dog park (`eval-walk` to `map-iou`): verdicts TP, TP, FP (bag), TP, FP (bench), FP (IoU 0.40 with D), TP (0.68 with D), FP (bush); dog E is FN. 4 TP, 4 FP, 1 FN: precision 0.5, recall 0.8. Cuts: ≥ 0.85 gives 1 and 0.4; ≥ 0.50 gives 0.6 and 0.6. Envelope heights 1, 1, 0.75, 0.57 on strips of 0.2: AP ≈ 0.66 (0.664; COCO's 101-point 0.668). AP75 = 0.55, AP@[.5:.95] ≈ 0.44.

### Code (verbatim)

`iou-code`:
```python
def box_iou(a, b):  # a: (N, 4), b: (M, 4)
    lt = torch.max(a[:, None, :2], b[:, :2])
    rb = torch.min(a[:, None, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    union = area_a[:, None] + area_b - inter
    return inter / union

a = torch.tensor([[407., 83., 768., 588.]])
b = torch.tensor([[467., 123., 828., 628.]])
box_iou(a, b)        # tensor([[0.6230]])
```

`nms-code`:
```python
def nms(boxes, scores, thr=0.7):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i, rest = order[:1], order[1:]
        keep.append(i)
        ious = box_iou(boxes[i], boxes[rest])[0]
        order = rest[ious <= thr]
    return torch.cat(keep)
```

`target-code` (12 lines):
```python
def build_target(boxes, classes, s=320):
    y = torch.zeros(2, 3, 7)  # object? 0 everywhere
    for (x1, y1, x2, y2), k in zip(boxes, classes):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r, c = int(cy // s), int(cx // s)
        y[r, c, 0] = 1               # object?
        y[r, c, 1] = cx / s - c      # b_x
        y[r, c, 2] = cy / s - r      # b_y
        y[r, c, 3] = (x2 - x1) / s   # b_w
        y[r, c, 4] = (y2 - y1) / s   # b_h
        y[r, c, 5 + k] = 1           # cat or dog
    return y
```

Also short and verbatim: `loss-grid` (`m = y[..., 0] == 1` / `loss = ((p[..., 0] - y[..., 0])**2).sum()` / `loss += ((p[m, 1:] - y[m, 1:])**2).sum()`), `detect-code` (8 lines ending `k = batched_nms(box, score, label, 0.7)  # 4`), `ap-code` (8 lines, ends `ap = (w * p).sum()  # 0.664`).

### The deck's own recap and summary slides

Each part ends with `recap-k` (no `recap-2`; Part 2 ends with `route`), whose body is the route strip, a `chart()` flowchart (Parts 3 to 8) and short numbered lines. Examples:
- `recap-3`: "A box is four numbers: corners (407, 83, 768, 588), or centre and size; a label file holds one line per object." / "IoU = overlap ÷ union: 1 for the same box, 0 for boxes apart. The 60-px guess: 0.623. IoU ≥ 0.5 is the usual “found it”."
- `recap-6`: "NMS keeps the best box and deletes boxes of its class with IoU above 0.7, then repeats: 31 boxes became 4."
- `recap-8`: "A box is a TP if its IoU with an object is at least 0.5; every other box is an FP; every object no box found is an FN. There are no TNs."

`pipeline` "The whole lecture, as one flowchart" (3 builds; a `chart()` of three rows, the step notes give the section):
- training: "photos with people's boxes" → "ground-truth table" (5.2) → "network's table" (5.3) → "loss, cell by cell" (4 · 5.3) → "adjust the weights"
- test time: "new photo" → "network: three grids of cells" (5.3 · 7) → "box + scores in every cell" (5.4) → "keep scores ≥ 0.25" (5.4) → "NMS: one box per object" (6 · IoU, 3) → "detections"
- measuring: "detections vs annotations" → "IoU ≥ 0.5 match: TP, FP, FN" (8.1) → "precision, recall" (8.2) → "PR curve, AP" (8.3) → "mean over classes: mAP" (8.4)

`summary` "The lecture in one slide": "The question: a network's output has a fixed shape; detection needs a list of any length." Three columns (`.sum3`): **Train** "People draw boxes." / "Each centre picks a cell, which gets the box and class." / "Loss: the network's table against it." **Detect** "Every cell of three grids writes a box and scores." / "Keep scores ≥ 0.25." / "NMS removes duplicates." **Measure** "IoU ≥ 0.5 with an object: TP; else FP. Misses: FN." / "Precision and recall." / "PR curve, its area AP; mean over classes: mAP." Story line: "Every cell writes a box; the score filter and NMS decide how many survive."

### Deep links

- https://nipunbatra.github.io/object-detection/#iou-union/3
- https://nipunbatra.github.io/object-detection/#target-photo/3
- https://nipunbatra.github.io/object-detection/#nms-dog/4
- https://nipunbatra.github.io/object-detection/#ap-area/3
- https://nipunbatra.github.io/object-detection/#pipeline/3

### Connections

- Classification (CNN classifier, ImageNet, ResNet-50 on `one-label`) is the starting point; Part 4 adds a box head beside the class head; convolutions "keep the photo's layout", which is why output position (r, c) belongs to cell (r, c) (`grid-structure`).
- Losses: Part 4 and 5 use squared error with "?" entries skipped (Ng's version); the deck notes that real detectors use cross-entropy for object? and the classes (`loss-y` build 5), and that each of YOLOv8's 80 class scores is a sigmoid trained with binary cross-entropy (`scores-80` notes, optional `bce`). Cross-entropy was taught in Lectures 23 and 24 (schedule); the deck does not cite the lecture.
- IoU is reused three ways (`iou-uses`): training assignment, NMS, evaluation.
- DETR (Appendix A, not taught) uses a transformer encoder (self-attention among 950 image tokens) and a decoder whose 100 learned queries use cross-attention (`detr-pipeline`), matched to objects by the Hungarian algorithm with cost −p(class) + 5·L1 + 2·(−GIoU); this reuses Lectures 14 and 15's attention, though the deck does not cite them.
- Segmentation is the next topic (`ladder`, `summary` notes: "Beyond Boxes").

### Exam-style questions

1. Why does detection report precision and recall rather than accuracy? Answer: accuracy needs true negatives, and "no box, no object" has no fixed count (every empty patch of grass), so detection counts only TP, FP and FN (`eval-fn`, `recap-8`).
2. Why does NMS loop instead of keeping the single best box, and why must it be class-aware? Answer: one round removes only boxes overlapping the kept one, so a second object's boxes (IoU 0.0) survive to the next round (`nms-two-dogs`); without classes, a bicycle overlapping a higher-scoring person (IoU 0.339) would be deleted at threshold 0.3 (`class-aware`).
3. On the 3 × 2 grid of 320-pixel cells, write the cat's target record from its box (112, 296, 321, 607). Answer: centre ((112 + 321)/2, (296 + 607)/2) = (216.5, 451.5); cell (⌊451.5/320⌋, ⌊216.5/320⌋) = (1, 0); b_x = 216.5/320 − 0 ≈ 0.68, b_y = 451.5/320 − 1 ≈ 0.41; b_w = 209/320 ≈ 0.65, b_h = 311/320 ≈ 0.97; record (1, 0.68, 0.41, 0.65, 0.97, cat 1, dog 0) (`cell-box`).

---

## 3. Reusable engineering

### 3.1 How a deck is wired

`index.html` (object-detection) loads, in order: `shared/katex/katex.min.css`, `shared/deck.css`, `lecture.css`, `shared/katex/katex.min.js`, then `<script type="module" src="shared/deck.js">`. The body holds `<main id="lecture">`, a `<nav class="controls">` with `data-action` buttons (prev, next, reset, finish, notes, scenes, present), `#read-link` (`?reading`), a Course link, and two asides `#notes` and `#overview`. `deck.js` imports `content.js` from the page's URL (`new URL('content.js',location.href)`), so the deck must be served over HTTP. `content.js` must export `meta`, `slides`, and optionally `mount(stage,{reading,printing,builds})`, which deck.js calls after rendering and KaTeX (object-detection: `export const mount=(stage,ctx)=>mountWidgets(stage,ctx,S)`). `content.js` loads its data with top-level await (`fetch('./data/science.json')`). Lecture 5's `index.html` adds an import map with `?v=` hashes and points at `../shared/` and `../lecture2/widgets.js`.

URL modes: `?all` shows optional slides, `?reading` a scrolling page of every slide at its final build plus notes, `?print` and `?print=builds` for PDF; deep link `#<id>/<build>`. Keys: arrows, space, Home/End, N notes, O scenes, R reset, F present.

Two copies of the engine exist. `object-detection/shared/` differs from `generative/lectures/shared/` in three lines: the page title comes from `meta.series`, "optional extension" is not added to Appendix kickers, and `author.js`'s `sectionDeck` has the object-detection Colab URL. Copy object-detection's.

### 3.2 `meta`, chapters and parts (object-detection)

```js
export const meta={id:'object-detection',number:22,title:'Object detection',series:'ES 667 Deep Learning',
 subtitle:'From one label to a set of objects',minutes:130,notebook:'object-detection.ipynb',
 sections:[{id:'map',title:'Lecture map',minutes:1},{id:'why',title:'Part 1 · Why detect objects?',minutes:12}, … ,
  {id:'wrap',title:'Wrap-up',minutes:4},{id:'detr',title:'Appendix A · DETR: slots matched to objects',minutes:0},
  {id:'optional',title:'Appendix B · More depth',minutes:0}]};
export const slides=[];
const CHAPTERS=[['1.1','Where detectors work'],['1.2','One output for all of them'], … ,['9.1','The whole lecture'],['A.1','Several objects, several slots'], …];
const PARTS=['Why detect?','A list','Boxes, IoU','One box, many','Dense grid','NMS','Three grids','Measuring'];   // short names for the route strip
const chap=k=>{chapter=CHAPTERS.find(c=>c[0]===k);…};   // call chap('3.2') before the slides of section 3.2
```
`meta.sections[].id` is the `section` argument of every slide; `minutes` feeds the Scenes panel and the speaker-notes pacing table. The kicker reads `${meta.number} · ${section title} · ${chapter}`.

### 3.3 Author API as used in `object-detection/content.js`

From `shared/author.js` (imported): `m, eq, at, only, cols, code, table, flow, palette, esc, highlight` (also exported but unused here: `image`, `plot`, `sectionDeck`). Defined in `content.js` itself: everything else below. `takeaway` and `maxH` do not exist in object-detection; they are Lecture 5 helpers (given at the end of this list).

| helper | signature | real example |
|---|---|---|
| `S_` | `S_(id, section, title, {builds, body, lines, ask, expected, point, misconception, transition, demo, source, nb, part, optional})`; throws unless `lines`, `ask`, `expected` each have exactly `builds` entries; `nb:null` hides the notebook link | `` S_('iou-def','boxes','Intersection over union (IoU)',{builds:3, body:cols(iouDiagram(),stack(at(1,p('Intersection: …')),at(2,p('Union: …')),at(3,eq(R`\mathrm{IoU}(A,B)=…`))),'1fr 1fr'), lines:[…], ask:[…], expected:[…], transition:'Compute it for the dog and the guess.'}) `` (three entries in each array) |
| `O` | `O(id, title, o)` = `S_(id,'optional',title,{...o,optional:true,lines:o.lines\|\|o.ask})` (Appendix B) | `` O('opt-yolov1','YOLOv1 (2016): one responsible cell per object',{builds:1, body:stack(eq(R`7\times7\ \text{cells}\times(2\times5+20)=7\times7\times30`),p('…'),small('…')), ask:[…], expected:[…]}) `` |
| `A_` | `A_(id, section, title, o)` = `S_` with `optional:true` (Appendix A, keeps its own section) | `A_('k-slots','detr','Give the head K slots',{builds:4, …})` |
| `partBreak` | `partBreak(k, section, title, lead, how[], {line, ask, expected, transition})`: makes slide `part-k` with the route strip, the lead and an ordered list | `partBreak(3,'boxes','Describe a box, compare two boxes','Every later step needs two small tools.',['Write a box as four numbers, …','Measure how well two boxes agree: …','Try IoU live, then write it in six lines of PyTorch.'],{line:'Two tools: four numbers for a box, and IoU to compare two boxes.',ask:'…',expected:'…',transition:'Positions in a photo.'})` |
| `recap` | `recap(k, section, did[], next, {flow, ask, expected, point, transition})`: slide `recap-k`, story line "<b>Next:</b> " + next, `nb:null` | `recap(8,'eval',['A box is a TP if …','Precision TP/(TP + FP) and …'],'the whole lecture, in one picture.',{flow:chart([{steps:[{t:'boxes, sorted\nby score'},…,{t:'PR curve,\nAP, mAP'}]}],{fs:19,alt:'…'}),ask:'…',expected:'…',transition:'The whole lecture.'})` |
| `chart` | `chart(rows, {W=1120, bh=96, gx=42, gy=26, fs=20, loop=null, out=null, alt})`; `rows=[{name, at, steps:[{t, n, hi}]}]`; `t` may hold `\n`, `n` a small note, `hi` tints purple, `at` reveals the row with `gat`; `loop:[row,from,to,label]` draws a return arrow | `pipeline` and every recap (see 2 above) |
| `codeSteps` | `codeSteps([[build, line], …])`: build 0 = always shown; the current build's line is tinted by `lecture.css` | `codeSteps([[1,'def nms(boxes, scores, thr=0.7):'],[2,'    order = scores.argsort(descending=True)'],…,[8,'    return torch.cat(keep)']])` |
| `at` | `at(build, html)` → `<div data-at>`: shown from that build on | `` at(2,eq(R`(c_x,c_y,w,h)=(…)`)) `` in `centre` |
| `only` | `only('3-4', html)` → `<div data-only>`: shown only on those builds ("2", "1-3", "1,4"); siblings swap in one `.state-slot` | `only('3-4',courtyard(gtBox(DOG,'dog',…)+rect(SW.box,…)))` in `sliding-cons` |
| `gat` | `gat(b, svgInner)` → `<g data-at>` | `photo(name,W2,H2,gat(2,boxes),alt,{view})` in `app()` |
| `gonly` | `gonly(b, svgInner)` → `<g data-only>` | `grid3(allNames+gonly('1-2',ownCell(N3.dog_cell,COL.amber))+gonly(3,ownCell([0,0],COL.amber))+gat(4,cellLossLabels),…)` in `align-loss` |
| `cols` | `cols(left, right, ratio='1fr 1fr')` | `cols(courtyard(…),sm(eq(…),eq(…),at(2,eq(…))),'0.85fr 1.15fr')` in `centre` |
| `stack` | `stack(...html)` → `.equation-stack` (vertical, gap 18px); `sm(...)` is the `eq-sm` variant | `stack(pipeFig,at(3,p('The output never changes shape. …')))` in `dense-recap` |
| `small` | `small(s)` → `<p class="caption">`; also `p(s)` → `<p class="lead">`, `ps(s)` → `<p class="small">` | `at(3,small('Its 118 dog classes together hold 0.39; …'))` |
| `m` | `m(tex)` inline KaTeX | `m('b_x')` in the record helpers |
| `eq` | `eq(tex)` display KaTeX in `.formula-scroll` | `` eq(R`\mathrm{IoU}(A,B)=\frac{\text{area}(A\cap B)}{\text{area}(A\cup B)}`) `` |
| `kt`, `kp`, `kl` | coloured text spans: target teal, prediction purple, loss red | `` kp(`network ${m('\\hat y')}`) `` |
| `table` | `table(headers, rows)` → `.table-scroll` | `dupes`, `nms-dog`, `eval-walk`, `pr-curve` |
| `flow` | `flow(nodes, edges, {width, height, label})`; node `{id,x,y,w,h,label,sub,role,shape}`, edge `{from,to,fromSide,toSide,role,label,dashed}`; `role` is a palette key | `one-object`: `flow([{id:'img',x:10,y:110,w:150,h:90,label:'photo',role:'data'},{id:'cnn',x:220,…,label:'CNN',sub:'backbone',role:'ink'},…],…)` |
| `code` | `code(text, lang='python')` | `eval-code` |
| `app` | `app(id, title, {name, boxes, alt, view, question, find, use, world, lines, ask, expected, point, transition, source})`: the six Part 1 application slides | `app('app-road','On the road: who is where?',{name:'street',view:[40,300,1200,404],…})` |

Lecture 5 equivalents (its `content.js`): `S(id, section, title, o)` (same contract; `nb` names a notebook anchor), `O(id, title, o)`, `partOpener(k, section, lead, ask, expected)` (body filled at the end with a figure and a `.pb-grid` of sections), `acts(k)` route strip, `story5(k)` the Part 5 sub-route strip, `takeaway(s)` → `<div class="takeaway"><b>Takeaway.</b> …</div>`, `maxH(h, html)` → `<div class="mh" style="--mh:${h}px">` (caps a diagram's height), `card(title, body, color, src)`, `grid(n, ...cards)`, `rows(...[label, html])`, `l1(id, sec, text)` (a chip linking `../lecture1/#id/1`), `mini(s)`, `PRI(tex)` (green prior). The derive grid pattern (`vi-kl`, `vae-loss`): rows `[tag, description, tex]` mapped to `<div class="derive">` with row k > 0 wrapped in `data-at=k+1`.

### 3.4 Figure helpers worth reusing

object-detection `widgets.js` (exported; all draw SVG strings; only `photo` needs an image file):

| helper | signature | draws | data | used on |
|---|---|---|---|---|
| `photo` | `photo(name, W, H, inner, alt, {view, cls})` | `figures/${name}.jpg` in its own pixel coordinates with SVG on top; `view=[x,y,w,h]` crops | the jpg | nearly every photo slide |
| `rect` | `rect(b, {stroke, w, dash, fill, op, extra})` | a box from corners | none | everywhere |
| `pill` | `pill(x, y, text, {fill, color, size, anchor, below, maxX})` | a detector-style class/score tab, kept inside the photo | none | `dupes`, `decode-box`, … |
| `label`, `line`, `arrow` | `label(x,y,s,{fill,size,anchor,weight,halo})`, `arrow(x1,y1,x2,y2,{stroke,w,head,both})` | SVG text (optional halo), lines, arrows | none | everywhere |
| `gridLines`, `cell`, `heat` | `gridLines(W,H,s,{stroke,w,op})`, `cell(r,c,s,{fill,op,stroke,w})`, `heat(map,s,{color,max,min})` | a cell grid, one filled cell, a score heat map | none | `centre-cell`, `candidates`, `dense-recap` |
| `fig`, `svg`, `panel` | `fig(W,H,inner,alt)`; `panel({ox,oy,W,H,L,R,T,B,xMin,xMax,yMin,yMax,ticksX,ticksY,xLabel,yLabel,fs})` → `{x,y,axes}` | a scaled SVG; axes with fixed ticks | none | `align-loss`, `pr-play` |
| `iouParts` | `iouParts(a,b)` → `{i, I, U, iou}` | the arithmetic only | none | `nms-play` |
| `tabs` | `tabs(items, {size, bounds})`, `items={box,text,fill}` | non-overlapping tabs | none | via `teachSVG` |
| `teachSVG` | `teachSVG(T, {upto, judge, t, cut, current, fn, dogs, size})` | the dog-park boxes judged TP (green) / FP (red) / FN (amber) | `S.teach` (gt, dets, iou, view) | `eval-setup`, `eval-walk`, `eval-fn`, `pr-threshold` |
| `scoreStrip` | `scoreStrip(top, names, {W, bh, box, size, alt})` | a cell's 80 class scores as bars | `S.grid.top`, COCO names | `scores-80` |
| `num`, `big`, `COL` | `num(v,n)` true minus, no −0.00; `big(v)` thousands commas | | | |

object-detection `content.js` (not exported; copy the function): `chart`, `codeSteps`, `gat`/`gonly`, `iouDiagram()` (self-contained A/B/A∩B/A∪B diagram with reveals 1 and 2; `iou-def`), `gtBox(b,lbl,o)` and `prBox(b,lbl,o)` (teal and purple boxes with tabs), `courtyard(inner,alt,o)` (`photo('cat-dog',…)`), `yvec(vals,cls)` / `rec7` / `ycol` (record boxes; `target-y`, `cell-box`, `grid-3x2`), `lossRow(label, vals, kinds, b, head)` (`loss-y`), `cellZoom` (`cell-box`, `dog-box`), `pipeFig` (thumbnails photo → CNN → 12,600 records → 31 → 4; `dense-recap`), `tensorBlocks` (`fixed-shape`), `vol` (3-D tensor block; `target-code`).

Lecture 2 `widgets.js` (exported; need `init(science, digits, nets)` first, and a browser canvas): `panel(o)` (richer than object-detection's: returns `{x,y,axes,clip,line,area,dot,bars}`), `label`, `fig`, `svg`, `widget`, `legend`, `covEllipse`, `animated(el,ctx,{count,render,step,…})`, `imgs(key)`, `digitURL(values)`, `digitSrc(key,i)`, `digit(key,i,size)`, `digits(key,{idx,cols,size,gap,labels,marks,title})` (a digit grid; `mnist`, `vae-samples`), `rawDigit(values,size)`, `scatterSVG(src,{lim,prior,…})` (codes of 1,000 test digits for `'ae2'`, `'vae2'`, `'pca'`, `'beta:<b>'`; `ae-grid`, `vae-scatter`), `posteriorSVG({S,view:'wide'|'zoom',q,enc,prior,title})` (the exact posterior for our 3 with q ellipses; `opt-q3`, inside `inferFig`), `squareW`, `classLegend`, `CLASS`, `MUTED`, `LINE`, `TINT`, `num`, `comma`.

Lecture 5 `content.js` figures (module-local, copy the function and its data):

| helper | draws | data | used on |
|---|---|---|---|
| `aeDiagram({kind:'ae'\|'vae', stage, showOut})` | image → encoder → code (or μ, log σ², ε, z = μ + σ ⊙ ε) → decoder → image, built by `stage` | digit images `example`, `ae2_recon3`, `vae2_mean_recon3` | `ae-arch`, `fix-net`, `part-2` |
| `genDiagram(kind)` | sample z ~ N(0, I) → decoder → new digit | prior-sample digit | `gen-question`, `part-3` |
| `fixesFig()` | ① where codes live → ② a cloud per image → ③ pull clouds to N(0, I) → a loss | self-contained | `part-4` |
| `klFig()` | four clouds over N(0, 1) with their KL penalties | `V.q3` (μ, σ of our 3) | `fix-kl` |
| `tugFig()`, `cloudsFig()` | sketches: rebuild-only vs KL-only vs both; points vs overlapping clouds | self-contained (`CLASS` colours) | `vae-tug`, `fix-cloud` |
| `budgetFig(one)` | log p(x) bar split into ELBO and KL to the posterior | self-contained | `part-6` |
| `castFig()`, `sameFig()` | Lecture 1's prior/likelihood/guide/ELBO above today's; the two problems as one picture | self-contained (`sameFig` uses our 3's image) | `part-5`, `same-problem` |
| `lossMap()` | the loss's two terms, each traced to its source, then −ELBO | self-contained | `maths-map` |
| `codePlane({ox,oy,S,alpha,draws,lim,ticks})` → `{s,P}` | the code plane with prior shading and draws | `d.gen.prior_points` | inside `storyFig`, `inferFig`, `likMapFig`, `vaePgm` |
| `storyFig()` | prior draw → decoder → mean image → + noise σ = 0.2 → x | `codePlane`, `V.prior_z`, digits | `vae-draw` |
| `inferFig(stage)` | our 3 → which codes? → zoomed posterior (and the encoder's q) | `posteriorSVG`, `V.grid` | `vae-wall`, `encoder-q` |
| `likMapFig(stage)` | codes A, B, C, what each draws, its score; stage 3 shades every code | `data/lik.json`, decoder weights (`logLikFor`) | `sigma` |
| `vaePgm(stage)` | the graphical model: z (green) → x (shaded), θ outside the plate | `codePlane`, digits | `vae-pgm` (optional) |
| `lossFig()` + `lossTable()` | our 3, its rebuild, the per-pixel error map, three marked pixels | digits `example`, `ae2_recon3` | `ae-loss` |
| `stdNormalFig()` | 500 draws of N(0, I) with radius-1 and radius-2 circles | `d.gen.prior_points` | `fix-prior` |
| `bayesTable()` | log prior, log likelihood, sum, × e^k for A, B, C | `LIKZ` (decoder) | `vae-wall` |
| `mcChart()` | naive Monte Carlo estimate of log p(x) vs number of draws | `V.naive_mc` | `evidence` (optional) |
| `inkFig()` | p(x), p(x, y) humps and p(y = 1 \| x) on one number per image | `data/xy.json` | `gen-disc` |

Self-contained (copy as is): `fixesFig`, `tugFig`, `cloudsFig`, `budgetFig`, `castFig`, `lossMap`, plus helpers `sub`, `arrowTo`, `imgBox`. Everything else needs Lecture 2's `science.json` / `digits.json` / `nets.json` or Lecture 5's `data/*.json`. Live widgets in Lecture 5 (`widget5`): `elbo-explorer`, `code-epochs`, `ae-train-long`; from Lecture 2 (`widget`): `scatter`, `explorer`, `ae-train`, `vi3`, `beta`. All run Lecture 2's decoder in the browser from `nets.json`.

### 3.5 CSS the figures rely on

| class | where | what it does |
|---|---|---|
| `.act-strip` (`.done`, `.now`) | `shared/deck.css` | the route strip of pills; `.story5` (Lecture 5) is a smaller variant with `→` separators |
| `.part-break`, `.pb-list`, `.pb-grid`, `.steps` | deck.css; `.steps` in lecture.css | part-opening and recap layouts |
| `.equation-stack`, `.eq-sm`, `.columns`, `.state-slot` | deck.css | vertical stack; smaller display maths; two-column grid; `only()` alternates overlaid in one grid cell |
| `.reveal-hidden` | deck.css | `visibility:hidden; opacity:0`: hidden builds still take their space |
| `.derive`, `.d-tag` | Lecture 5 CSS-in-JS | 230px label column + equation column for a step-by-step derivation |
| `.mchain`, `.mread`, `.mc-h/.mc-n/.mc-w/.mc-s` | Lecture 5 | the "five lines of maths" table (number, name, maths, source) |
| `.takeaway`, `.mh`, `.l1-link`, `.cards/.card`, `.dg/.dgrid/.dg-cell`, `.digit-rows` | Lecture 5 | takeaway box, height cap, back-link chip, cards, digit grids |
| `.yvec` (`.sm`, `.bare`, `.kinds` with cells `.t/.p/.l/.q`) | lecture.css | a 7-entry record as equal boxes (gaps after object? and the box); `.kinds` colours target/prediction/loss/"?" cells; `.bare` drops the entry names |
| `.ycol`, `.loss-row` (`.sl`), `.rec-table`, `.record` | lecture.css | a record as a column (Ng style); one labelled record row per build; record tables |
| `.kt`, `.kp`, `.kl` | lecture.css | teal / purple / red text for y, ŷ, loss |
| `.code-steps .cl` | lecture.css | tints the line whose `data-at` equals the current build (rules written for builds 1 to 9 only) |
| `.photo-fig` (`.tall`, `.short`, `.mid`) | lecture.css | photo SVG max-height 430 / 470 / 360 / 300 px, `overflow:hidden` |
| `.sum3` (`.st` teal, `.sd` purple, `.sm` green) | lecture.css | the three-column summary slide |
| `.table-sm`, `.tight`, `.xtight`, `.hit`, `.miss` | deck.css / lecture.css | smaller tables; TP green and FP red text |

Lecture 5 injects its CSS from `content.js` into `<style id="lecture5-style">`; object-detection keeps it in `lecture.css`.

### 3.6 Scripts (object-detection/scripts)

| script | run | needs | checks / writes |
|---|---|---|---|
| `serve.py` | `python3 scripts/serve.py [--port 9000]` | any Python 3 (stdlib) | serves the repo root at http://127.0.0.1:8780/ with `Cache-Control: no-store`; all the Node scripts expect it running |
| `verify.mjs` | `node scripts/verify.mjs [slide ids…]` | Node, `puppeteer` resolvable (here `node_modules` is a symlink to `/Users/nipun/git/dl-teaching/node_modules`, puppeteer 24.40.0), Chrome (`PUPPETEER_EXECUTABLE_PATH`, else `/Applications/Google Chrome.app/…`), `DECK_URL` (default `http://127.0.0.1:8780`) | renders every state of `?all` at 1280 × 800; flags page errors, 4xx responses, KaTeX errors, text outside the 60 px margins, content overflow, wide formula/diagram/table/code boxes, missing images, "stray content after an SVG", photo SVGs painting outside, SVG text outside its viewBox; UI checks (arrows, finish, reset, notes, F, deep link, `?present#title` opens slide `title`, mobile `?reading` width 390); writes `qa/screenshots/NNN-id-build.jpg` (deletes old ones on a full run), `qa/browser-checks.json`, and regenerates `manifest.json` and `speaker-notes.md`; exit code 1 on failure |
| `density.mjs` | `node scripts/density.mjs` | same as verify | words per state (body + story line + half the SVG text; code, KaTeX and the route strip excluded); WARN 70, MAX 100 words, story line ≤ 30 words, code ≤ 12 lines; writes `qa/density.json`; exit 1 if a limit is broken |
| `export.mjs` | `node scripts/export.mjs` | same as verify | loads `?print` and `?print=builds`, fails on page errors, maths errors or `.content` overflow; writes `exports/object-detection.pdf` and `exports/object-detection-builds.pdf` (names hard-coded) |
| `contact_sheets.py` | `python3 scripts/contact_sheets.py [--all]` | Pillow (the `python3` on PATH, `/Users/nipun/.uv/base/bin/python3`, has no PIL; `~/base/bin/python` has Pillow 11.3.0), `manifest.json`, the verify screenshots | 3 × 3 sheets of final states (or every state with `--all`) into `qa/contacts/final-NNN.jpg` / `states-NNN.jpg` |

`.gitignore` excludes `node_modules`, `qa/screenshots/`, `qa/contacts/`. Order in CLAUDE.md: `run_models.py` → `science.py` → `build_notebook.py --run` → `verify.mjs` → `density.mjs` → `contact_sheets.py --all` → `export.mjs`.

### 3.7 Visual system

- Palette (`shared/author.js`): `data:'#245EDB'` blue, `q:'#8B2CDE'` purple, `p:'#0F766E'` teal, `noise:'#AA4E08'` amber/orange, `loss:'#BE123C'` red, `d:'#147737'` green, `ink:'#14171F'`, `paper:'#FAFAFB'`. KaTeX macros map the same: `\pcol` p, `\qcol` q, `\dcol` data, `\ncol` noise, `\lcol` loss.
- object-detection `COL`: `target` = teal (annotation, made by people), `guess` = purple (model output), `loss` = red, `data` = blue, `amber` = orange (other classes, FN, the cell highlight), `green` = green (TP), `ink`. `MUTED '#68717B'`, `AX '#AAB3BC'`, `PAPER '#FAFAFB'`.
- Lecture 2 `widgets.js`: `MUTED '#68717B'`, `AX '#AAB3BC'`, `LINE '#E1E6EB'`; `CLASS` (digits 0 to 9) `['#245EDB','#AA4E08','#147737','#BE123C','#8B2CDE','#8C564B','#D6489B','#4B5563','#9A8A00','#0F9BAE']`; `TINT` 24/2E alpha versions. Lecture 5: `GREEN='#147737'` for the prior.
- `deck.css` variables: `--paper #FAFAFB`, `--ink #14171F`, `--muted #68717B`, `--line #D7DCE1`, the six palette colours, `--font: "Avenir Next", Avenir, "Helvetica Neue", Arial, sans-serif`; code in `"SFMono-Regular", Menlo, …`; maths in KaTeX (bundled in `shared/katex`). Slide canvas 1280 × 720, h2 40px, `.lead` 28px, captions 22px, code 25px (`.code-xs` 18px).

### 3.8 Things that will bite a new deck

1. Reveals inside an SVG must be `gat()`/`gonly()`; an HTML `at()` inside an SVG string closes the SVG early (verify reports "stray content after an SVG").
2. `builds` must equal the length of `lines`, `ask` and `expected`, or `S_` throws at load and the whole deck fails to render; `at(k)` with k > builds never shows.
3. Hidden builds keep their space (`visibility:hidden`), so a slide must fit at its final build; `only()` swaps need siblings directly inside `.content` or a `div` in it to be stacked in a `.state-slot`.
4. KaTeX macros are fixed in `deck.js`; there is no green macro (use `\textcolor{#147737}{…}`), and adding one means editing `deck.js` (note the `##` escaping there).
5. `verify.mjs` assumes a slide with id `title` (the `?present#title` check) and overwrites `manifest.json` and `speaker-notes.md`; `export.mjs` hard-codes the PDF names; the `.code-steps` tint covers builds 1 to 9 only.
6. `content.js` uses top-level `await fetch` and relative `figures/…` paths: serve over HTTP from the deck root; `deck.js` imports `content.js` relative to the page URL.
7. The Lecture 5 figures are not exported and need Lecture 2's data, `init()`, and a DOM canvas at call time; reuse them by copying the function plus a trimmed data file, or redraw.
8. Course numbering: set `meta.number`, title text and the Course link to the schedule's numbers (25, 26), not the decks' internal 5 and 22.
