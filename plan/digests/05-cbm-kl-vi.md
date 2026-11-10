# Digest 05: Concept Bottleneck Models (Lectures 20, 21) and KL Divergence and Variational Inference (Lectures 23, 24)

Sources read: `/Users/nipun/git/cbm` (`CLAUDE.md`, `docs/lecture.md`, `docs/worked_examples.md`, `docs/mathematical_notes.md`, `narration/order.txt`, `src/data.json`, `src/kit/theme.ts`, scene TeX) and `/Users/nipun/git/generative/lectures/lecture1` (`manifest.json`, `README.md`, `speaker-notes.md`, `content.js` evaluated in Node to dump every slide's body and story lines, `data/science.json`). Numbers below are quoted as the slides show them; where I recomputed one, I say so.

## How the sessions split

**CBM (one deck, two sessions).** The deck (live: https://nipunbatra.github.io/cbm/, notes: https://nipunbatra.github.io/cbm/notes.html, 262-page PDF) is one continuous lecture in Parts 1 to 13 (Part 9, leakage, was removed on 16 October; `#s09` and `#s20` links now redirect to `#s10.1`). Nothing in the repo records where the class stopped. I split at the schedule's own wording: Lecture 20 lists "interpretable-by-design prediction ... CUB examples, concept supervision, training schemes and interventions", which is Parts 1 to 7 (scenes `s00` to `s07`, ending with "A correct edit can still hurt"). Lecture 21 opens with "concept sufficiency", which is Part 8 (`s08`, the blue circle and blue square), then Parts 10 to 13, the recap and two appendices. Part 7 stays in Lecture 20 because it is still about interventions.

**KL/VI (one deck, two sessions).** Five parts, 126 core slides, 146 planned minutes (Part 1: 30, Part 2: 62, Part 3: 25, Part 4: 20, Part 5: 8). The deck's `README.md` says "A natural split is after Part 2". I use that: **Lecture 23 = map + Part 1 (Bayes for a network) + Part 2 (surprise to KL, applications, closed forms, KL by sampling)**; **Lecture 24 = Part 3 (find the closest q, reparameterisation, KL direction) + Part 4 (ELBO) + Part 5 (VI for the network)**. Caveat: that puts 93 planned minutes in the first session and 53 in the second, so Sections 2.6 and 2.7 may in practice have spilled into Lecture 24. The git history fits the Part 2/Part 3 boundary (the 8 and 9 November commits only touch Part 4, the ELBO). Deep links: `https://nipunbatra.github.io/generative/lectures/lecture1/#<slide-id>/<build>` (builds are 1-based).

---

## Lecture 20: Concept Bottleneck Models I

**Aim.** Replace "explain a black box afterwards" with a model whose prediction has to pass through named, human-checkable concepts that a person can correct.

**Running examples.**
- *Gull and CLIP (black box, Part 1).* CLIP ViT-B/32 on a gull photo, six species: American herring gull 0.54, glaucous-winged gull 0.21. An occlusion map (14 × 14 patches) shows the largest drops on the bill and head, but "the map shows where the score is sensitive, not which property".
- *Gull CBM (Part 2, hand-written head).* Five concepts: wingtip grey, wingtip black, bill pointed, head white, tail rounded. ĉ = (0.8, 0.2, 0.9, 0.85, 0.6), head w = (−2, 2, 0.4, −0.2, 0.2), b = 0. Glaucous-winged 0.71 vs herring 0.29. The observer corrects grey → 0, black → 1, and herring gull rises to 0.91.
- *CUB.* 11,788 photos, 200 species (8 gulls), 312 binary attributes in 28 groups. Koh et al. take a majority vote within each species and keep 112 attributes present in at least 10 species. One record = photo + 112 zeros and ones + species.
- *Zebra / horse / giraffe worksheet (Part 4).* Concepts (stripes, horse-like shape, long neck), ĉ = (0.1, 0.9, 0.1), W rows zebra (3, 1, 0), horse (−1, 2, −1), giraffe (−1, 0, 3), no bias.
- *Sleep EEG (schematic).* Spindle 0.12, K-complex 0.8, slow-wave 0.15: other stage 0.62 vs N2 candidate 0.38. Spindle set to 1 gives N2 0.86.
- *Billing message (constructed faulty head).* "The app works, but I was charged twice. Please refund the extra payment." Duplicate charge 0.2, refund requested 1.

**Ideas in teaching order.**
1. A heatmap (saliency |∂s_k/∂x_i|, occlusion) says where, not what. Adebayo's randomised-weights sanity check; Zech's hospital shortcut in pneumonia X-rays (Mount Sinai about a third positive, NIH about 1 in 100).
2. Post-hoc vs ante-hoc ("interpretable by design", Rudin). A CBM is an ante-hoc model for deep networks: x → ĉ → ŷ, with g (concept predictor) and f (task head).
3. The bottleneck rule: "the task head reads only these five numbers". Editing a concept changes ŷ, unlike in multi-task learning, where the class head never reads the concept head.
4. CUB is the standard benchmark (112 concepts, majority vote).
5. What is a concept: meaning, grounding, representation. Six ways to store one: binary, soft (σ(1.4) = 0.80, "not automatically a probability"), categorical (softmax), continuous, vector/distribution, unknown ("not visible is different from absent").
6. The arithmetic: scores = W ĉ, softmax, one edit, rerun the same head.
7. Learning: L_c (BCE per concept), L_y (cross-entropy), weighted sum; Koh's independent, sequential and joint training. In joint training the task gradient flows into g, so "a concept score can be pulled two ways".
8. Measured on CUB (frozen CLIP features → 112 → 200): joint is most accurate; under intervention, independent gains most.
9. Interventions correct categorical groups together. A correct edit can still hurt if the head has a wrong-signed weight.

**Key equations** (the deck colours them with KaTeX macros from `src/kit/theme.ts`: `\cin` input blue #5B9BFF, `\ccon` concepts teal #2DD4BF, `\chead` head grey, `\cedit` human edits amber #F5B342, `\closs` losses coral #FF7F6B, `\cleak` leakage purple; shorthands `\gth` = g_θ, `\fph` = f_φ, `\cvec` = **ĉ**):

$$x \xrightarrow{\;g_\theta\;} \hat{\mathbf c} \xrightarrow{\;f_\phi\;} \hat y,\qquad \hat{\mathbf c}=g_\theta(x),\qquad \mathbf s=f_\phi(\hat{\mathbf c})=W\hat{\mathbf c},\qquad p_k=\frac{e^{s_k}}{\sum_j e^{s_j}}$$

$$\hat c_j=\sigma(z_j)=\frac{1}{1+e^{-z_j}}$$

$$\mathcal L_c=\frac1{nk}\sum_{i,j}\mathrm{BCE}(\hat c_{ij},c_{ij}),\quad \mathcal L_y=\frac1n\sum_i-\log\hat p_{i,y_i},\quad \mathcal L=\lambda_c\mathcal L_c+\lambda_y\mathcal L_y$$

Independent: fit g to **c** and f on true **c** (at test time f gets ĉ). Sequential: fit g, freeze, fit f on ĉ = g(x). Joint: minimise 𝓛 over θ, φ together, with

$$\frac{\partial\mathcal L_y}{\partial\theta}=\frac{\partial\mathcal L_y}{\partial\hat{\mathbf c}}\,\frac{\partial\hat{\mathbf c}}{\partial\theta}$$

Intervention on a linear head: $\Delta\mathbf s=W\,\Delta\hat{\mathbf c}$ (here $0.9\,W_{:,1}=(2.7,-0.9,-0.9)^\top$). Saliency (s21): $S_i^{(k)}=\left|\partial s_k(x)/\partial x_i\right|\approx\left|\Delta s_k/\Delta x_i\right|$.

**Worked numbers worth reusing.**
1. *Zebra worksheet* (`#s04.6` to `#s04.12`): scores (1.2, 1.6, 0.2), softmax (0.350, 0.522, 0.129), so horse. Stripes 0.1 → 1 gives scores (3.9, 0.7, −0.7) and zebra 0.95; "3 times 0.9, which is 2.7".
2. *Gull group edit* (`#s02.7` to `#s02.9`, again in `s06`): z = −1.6 + 0.4 + 0.36 − 0.17 + 0.12 = −0.89, herring 0.29; after the edit z = 2.31, herring 0.91.
3. *One concept-loss term* (`#s05.4`): label 1, prediction 0.8, −log 0.8 ≈ 0.22.
4. *Faulty billing head* (`#s07.6`, `#s07.8`): s = 1.2 − 2·dup + 0.6·refund. Before: 1.2 − 0.4 + 0.6 = 1.4, billing 0.80. After the correct edit dup = 1: −0.2, billing 0.45.
5. *CUB run* (`#s23.4`, `#s23.6`; 5,994 train / 5,794 test, one run): task accuracy standard 73.3 %, joint 73.5 %, sequential 68.9 %, independent 49.9 %. With all 28 groups corrected: independent 100.0 %, sequential 98.5 %, joint 90.2 %. Mean concept accuracy 0.909 (sequential predictor).

**Code.** No code is shown on the slides. The measured shape experiment's loss lives in a script, not on a slide: `loss = CE + 2 * BCEWithLogits` (joint, λ_c = 2, λ_y = 1, Adam, lr 0.003; `docs/mathematical_notes.md` §4).

**Figures and widgets worth reusing** (scene components in `src/scenes/`, shared pipeline in `src/kit/pipe.tsx`; photos in `public/img/`):
- `S01.tsx` CLIP black box + occlusion map on `public/img/gull.jpg`.
- `src/kit/pipe.tsx` (used by `S02`, `S06`, `S14`): the recurring gull pipeline x → g → five teal concept bars → f → shares, with the amber edit.
- `S04.tsx`: the W ĉ layer diagram with the column contributions and the softmax.
- `S05.tsx`: the three training schemes as diagrams with L_c and L_y.
- `S23.tsx`: CUB accuracy bars and intervention curves.
- `S07.tsx`: the billing head's wrong-signed weight.

Deep links: https://nipunbatra.github.io/cbm/#s21.15 (CBM vs post-hoc), https://nipunbatra.github.io/cbm/#s02.9 (edit flips the gull), https://nipunbatra.github.io/cbm/#s04.11 (zebra 0.95), https://nipunbatra.github.io/cbm/#s23.6 (CUB intervention curves), https://nipunbatra.github.io/cbm/#s07.9 (fault is in the head).

**The lecture's own summary lines** (slide headings): "An edited belief, a new prediction" (s02.10). "In a CBM, the concepts lie on the path to the prediction" (s02.13). "The concept vector is the point where a person can intervene" (s04.13). "Correcting a concept does not repair the decision rule" (s07.10). "A constructed case: the failure can happen" (s07.11).

**Connections.** Builds on CLIP (Lecture 17): the Part 1 black box is CLIP zero-shot, and the CUB run uses frozen CLIP ViT-B/32 features (512-d). Saliency uses a ResNet-50. L_y is the familiar classification cross-entropy. Lecture 21 reuses the zebra/horse/giraffe classes and the gull for every variant. Lecture 23 later reinterprets this same cross-entropy as KL(label ‖ prediction).

**Exam-style questions.**
1. *Why does editing the concept output of a multi-task model leave the prediction unchanged, while the same edit in a CBM changes it?* Answer: in multi-task learning the concept head is a side branch off a shared encoder, and the class head never reads it. In a strict CBM, ŷ = f(ĉ), so the concepts lie on the only path to the prediction.
2. *On CUB, why does the independent model gain most from interventions even though it is the least accurate without them?* Answer: its head was trained on true binary concepts, so it trusts corrected (true) values. The joint head learned to read predicted, possibly drifted scores, so true values match what it expects less well.
3. *Numeric.* W rows zebra (3, 1, 0), horse (−1, 2, −1), giraffe (−1, 0, 3); ĉ = (0.1, 0.9, 0.1). Find the prediction, then correct stripes to 1. Answer: s = (0.3 + 0.9 + 0, −0.1 + 1.8 − 0.1, −0.1 + 0 + 0.3) = (1.2, 1.6, 0.2). Softmax: e^1.2 = 3.32, e^1.6 = 4.95, e^0.2 = 1.22, total 9.49, so (0.35, 0.52, 0.13) and the model says horse. After the edit Δs = 0.9·(3, −1, −1) gives (3.9, 0.7, −0.7). Softmax: 49.4, 2.01, 0.50 out of 51.9, so zebra 0.95.

---

## Lecture 21: Concept Bottleneck Models II

**Aim.** Show what a bottleneck can and cannot carry (sufficiency), how newer variants cut the cost of concept labels (CEM, post-hoc CBMs, classification by description), and how to evaluate a CBM and where it fails.

**Running examples.**
- *Blue circle vs blue square (Part 8).* The vocabulary has only colour and size, so both give the same **c**, while y₁ ≠ y₂. The best possible accuracy is one half.
- *Zebra / horse / giraffe photo set (Part 10).* 36 Wikimedia photos, 24 train and 12 test, used for every rung.
- *CEM stripes.* c⁺ = [0.82, 0.3], c⁻ = [0.22, 0.74], p = 0.1, mixed vector [0.28, 0.696]. Shares (zebra, horse, giraffe) [0.1436, 0.7227, 0.1336]; setting p_stripes = 1 gives [0.8599, 0.1182, 0.0219].
- *PCBM on `zebra_12.jpg`.* An 8-phrase CLIP bank: black and white stripes, a long neck, a flowing mane, brown patches on a pale coat, hooves, a saddle, green grass, a wooden fence. Raw stripes cosine 0.274, standardised +1.41.
- *Classification by description on the gull.* The five herring gull descriptors score 0.32 to 0.34 (average about 0.33); slaty-backed gull edges ahead (0.337 vs 0.334), a near miss. The LLM's California gull "black head" is wrong.
- *Synthetic shapes (Part 11, measured).* Colour × shape × size, 8 classes, 5 concepts, 512 test images, one seed.

**Ideas in teaching order.**
1. Sufficiency: "the concepts say all the input says about the label", $Y \perp\!\!\!\perp X \mid C$. A strict CBM guarantees only that ŷ depends on x through ĉ. The fix is a richer vocabulary (add shape), not better colour and size estimates.
2. "Labelling every concept on every image is the main cost." A ladder of shrinking supervision: rung one (x, c, y: CBM, CEM), rung two (x, y: post-hoc), rung three (no fitting on the new task: CBD).
3. CEM: two embeddings per concept, a learned p, and the mix p c⁺ + (1 − p) c⁻. An intervention sets p, not the coordinates.
4. PCBM: frozen CLIP image and text encoders, cosine scores, per-prompt standardisation on the 24 training photos, and a sparse L1 head (4 of 24 weights non-zero). PCBM-h adds a residual head that bypasses the concepts.
5. CBD (Menon and Vondrick): an LLM checklist per class; the class score is the average CLIP cosine. "A match score is not a fact check."
6. Comparison table: supervision, concept source, what the head reads, what is trained. "Each rung up the ladder removes a cost and a check."
7. Evaluation: task accuracy, concept quality, faithfulness, intervention utility.
8. Beyond images: knee X-ray (OAI, 10 findings → grade 0 to 3), CEBaB reviews, ICU vasopressor onset, X-CHAR wearables, sleep EEG.
9. Five failure points along x → ĉ → ŷ: distribution shift (AUC about 0.93 to about 0.82 at a new hospital), wrong grounding, missing information, wrong head relationship, a wrong or costly intervention. Appendices: TCAV (grey wings, about 0.99) and VLG-CBM (detector-grounded concepts, NEC = 5).

**Key equations** (same colour macros as Lecture 20):

$$Y \perp\!\!\!\perp X \mid C\quad(\text{sufficiency}),\qquad \hat y=f_\phi(\hat{\mathbf c}),\ \hat{\mathbf c}=g_\theta(x)$$

$$\text{CEM: }\hat{\mathbf c}_j=\hat p_j\,\mathbf c_j^{+}+(1-\hat p_j)\,\mathbf c_j^{-},\qquad \mathbf c_j^{\pm}\in\mathbb R^m$$

$$\text{PCBM: }\hat c_j=\frac{\cos(e,\,v_j)-\mu_j}{\sigma_j}\ (\text{saved training mean and spread}),\quad f\ \text{sparse linear};\ \text{PCBM-h adds a residual } r$$

$$\text{CBD: }\hat c_d=\cos\big(\phi(x),\psi(d)\big),\qquad \hat y_k=\frac1{|D_k|}\sum_{d\in D_k}\hat c_d$$

$$\text{NEC: }\frac1K\sum_{k=1}^{K}\lVert W_k\rVert_0$$

**Worked numbers worth reusing.**
1. *Insufficiency* (`#s08.4` to `#s08.7`): c(x₁) = c(x₂), labels differ, so any head scores at most 50 % on the pair.
2. *CEM mix* (`#s15.7`): 0.1 × 0.82 + 0.9 × 0.22 = 0.28 (second entry 0.696). The head favours horse (0.72). Setting p = 1 swaps in c⁺ and zebra leads at 0.86 (`#s15.9`).
3. *PCBM zebra* (`#s16.7`, `#s16.12`): stripes z = 1.41 (1.4062). Zebra logit 1.4062 × 1.4856 − 0.0545 ≈ 2.03, shares (0.774, 0.113, 0.113). Editing stripes to −0.6594 (the non-zebra training mean) drops zebra to 0.14. PCBM-h: 2.03 + 0.31 = 2.34, and the same edit leaves zebra at 0.20 because the residual is untouched (`#s16.16` to `#s16.17`).
4. *Evaluation* (`#s11.2` to `#s11.3`): test image small red square, large scored 0.78, predicted large red square. Setting large to 0 gives small red square. Accuracy with one oracle correction per image: strict CBM 81.6 % → 97.3 %; multi-task stays at 82.0 %.
5. *VLG-CBM loss term* (`#s24.10`, appendix): box present, prediction 0.8, −log 0.8 ≈ 0.22. No box, prediction 0.2, −log(1 − 0.2), the same cost. NEC with rows of 3, 2 and 4 non-zero weights = 3 (`#s24.11`).

**Code.** None on slides.

**Figures and widgets worth reusing:**
- `S08.tsx`: the circle/square collision; shape "disappears" at the bottleneck.
- `S10.tsx` + `ApproachBreak.tsx`: the three-rung supervision ladder with learned vs fixed heads.
- `S15.tsx`: CEM c⁺/c⁻ vectors mixed by p.
- `S16.tsx`, `S16Standardisation.tsx`, `S16Residual.tsx`: CLIP phrase scores, the standardisation table across three test photos (`public/img/worked/`), the sparse head, PCBM-h.
- `S25.tsx`: the CBD checklist for gull species with cosine scores.
- `S19.tsx`: the four-column comparison table. `EvaluationBrief.tsx` / `S11.tsx`: the intervention-utility chart.

Deep links: https://nipunbatra.github.io/cbm/#s08.9 (sufficiency in symbols), https://nipunbatra.github.io/cbm/#s10.7 (rung two vs rung three), https://nipunbatra.github.io/cbm/#s16.17 (residual bypasses the edit), https://nipunbatra.github.io/cbm/#s19.8 (a cost and a check), https://nipunbatra.github.io/cbm/#s13.8 (five questions).

**The lecture's own summary lines.** "Each rung up the ladder removes a cost and a check" (s19.8). "Five questions for any concept bottleneck": are the concepts meaningful, grounded, sufficient, used, and do corrections improve decisions (s13.8). Recap (s14.6): "whatever the answer uses should be visible in the concepts." "What to remember" (s14.7), four things: post-hoc methods explain after the fact while a CBM makes the answer pass through named concepts; fixing a concept helps only if the head reads it the right way; a soft score can carry more than its name says, and that leaks to the head; newer variants need fewer labels, so check each concept looks at the right evidence.

**Connections.** Rung two and rung three reuse CLIP's two encoders and cosine similarity (Lecture 17) and LLM-written text (the VLM lectures 18 and 19). The circle/square argument is the formal version of Lecture 20's "the head reads only these numbers". The shape experiment's multi-task comparison closes the loop on Lecture 20's multi-task contrast.

**Exam-style questions.**
1. *A CBM with perfect colour and size detectors cannot separate a blue circle from a blue square. Why, and what is the fix?* Answer: both images map to the same concept vector, so any head, linear or deep, gives them the same output (at most 50 % on the pair). Sufficiency fails because the label depends on shape, which C lacks. Add shape to the vocabulary.
2. *Why does PCBM-h keep a higher zebra probability (0.20) than PCBM (0.14) after the same stripes edit, and what is the trade-off?* Answer: the residual head reads the image features directly and adds its logits (0.31 for zebra), and the edit does not touch it. Better predictive capacity, less control through named concepts.
3. *Numeric (CEM).* c⁺ = (0.82, 0.30), c⁻ = (0.22, 0.74). Compute ĉ for p = 0.1 and for p = 1. Answer: p = 0.1: (0.1·0.82 + 0.9·0.22, 0.1·0.30 + 0.9·0.74) = (0.082 + 0.198, 0.030 + 0.666) = (0.28, 0.696), close to c⁻. p = 1: ĉ = c⁺ = (0.82, 0.30). The intervention changes p only; the weights and both vectors stay fixed.

---

## Lecture 23: KL Divergence and Variational Inference, session 1 (map, Part 1, Part 2)

**Aim.** Show that Bayes for even a seven-parameter network is blocked by the evidence p(𝒟), and build the KL divergence from surprise, entropy and codes as the "distance" that variational inference will minimise.

**Running examples.**
- *Tiny BNN.* $f_\theta(x)=v_1\tanh(w_1x+b_1)+v_2\tanh(w_2x+b_2)+c$, θ = (w₁, w₂, b₁, b₂, v₁, v₂, c) ∈ ℝ⁷, prior θ_j ~ N(0, 1). 17 points with a gap in the middle, y = 0.8 tanh(3x) + 0.3x + N(0, 0.1²), seed 7. The likelihood uses noise 0.1.
- *θ_A* (MAP: Adam, lr 0.05, 2,000 steps, best of ten starts) = (w₁ −0.48, w₂ −2.00, b₁ −0.59, b₂ 0.45, v₁ −0.46, v₂ −0.96, c −0.16) (recomputed from `science.json`). log p(𝒟 | θ_A) = 12.17, log p(θ_A) = −9.41. The prior draw θ_B has log-likelihood −2,157.5.
- *Weather (Part 2).* Forecast 0.9 sun / 0.1 rain (H = 0.47 bits). Our town p = (½, ¼, ¼) for sunny, cloudy, rainy (H = 1.5 bits, code 0, 10, 11). Another town q = (¼, ¼, ½) (code 10, 11, 0). Sunny town q = (0.98, 0.01, 0.01) for asymmetry.
- *Pretend posterior* p = N(1, 4²), q = N(μ, σ²); candidates q₁ = N(0, 2²), q₂ = N(1, 3²).
- *Classifier.* Ape, bear, cat, dog; prediction q = (0.05, 0.05, 0.60, 0.30), true label cat. Teacher (0.02, 0.03, 0.75, 0.20). Distillation demo: 4,547-parameter teacher, 27-parameter student.
- *Toy posterior for the three methods.* 0.7·N(−2, 1) + 0.3·N(2, 1).

**Ideas in teaching order.**
1. Bayes' rule with the four names (posterior, likelihood, prior, evidence). In ML, A = θ and B = 𝒟. "Usual training keeps one θ", while Bayes keeps them all.
2. One draw of θ is one whole function. The prior predictive is about ±2.7 wide everywhere.
3. MAP training = "squared error plus weight decay". θ_A scored by likelihood and prior.
4. The evidence is an area. With 100 grid points per unknown, our 7 unknowns need 100⁷ = 10¹⁴ points, "about 28 hours at a billion a second"; a real network needs 10^2,000,000. "That is what intractable means."
5. Three ways to approximate the posterior: Laplace, MCMC, VI. VI chooses q_ψ (Pyro's "guide"); for the network, one Gaussian per parameter, ψ = 14 numbers.
6. Surprise I(x) = log 1/p(x): only the log makes independent surprises add. Entropy is average surprise and counts yes/no questions.
7. Codes: prefix codes, Huffman, and "each codeword is exactly as long as its surprise". Cross-entropy is the cost of a code that expects q; KL is "the extra bits we waste". It is never negative and not symmetric. Read KL(p ‖ q) as "data from p, coded with q".
8. KL at work: the classification loss is KL(label ‖ prediction); distillation's cross-entropy = H(teacher) + KL(teacher ‖ student). VI flips the order: KL(q ‖ posterior).
9. Closed forms (Bernoulli, categorical, Gaussian, diagonal Gaussian), PyTorch's `kl.py`, and KL as an expectation estimated by Monte Carlo.

**Key equations.** Colour macros (`lectures/shared/deck.js`): `\pcol` teal #0F766E = p, the target or posterior; `\qcol` purple #8B2CDE = q, ours; `\dcol` blue #245EDB = data / evidence p(𝒟); `\ncol` orange #AA4E08 = the likelihood log p(𝒟 | θ) (also ε in 3.5); prior p(θ) in green #147737 (raw `\textcolor`); `\lcol` rose #BE123C = extra bits and the KL to the posterior. `\N` = 𝒩, `\E` = 𝔼, `\KL`, `\ELBO`.

$$\pcol{p(\theta\mid\mathcal D)}=\frac{p(\mathcal D\mid\theta)\,p(\theta)}{\dcol{p(\mathcal D)}},\qquad \dcol{p(\mathcal D)}=\int p(\mathcal D\mid\theta)\,p(\theta)\,d\theta$$

$$I(x)=\log_2\frac1{p(x)},\qquad H(p)=\sum_x p(x)\log_2\frac1{p(x)}=\E_{x\sim p}\Big[\log_2\frac1{p(x)}\Big]$$

$$H(\pcol p,\qcol q)=\sum_x\pcol{p(x)}\log_2\frac1{\qcol{q(x)}},\qquad \KL(\pcol p\,\|\,\qcol q)=H(\pcol p,\qcol q)-H(\pcol p)=\sum_x\pcol{p(x)}\log_2\frac{\pcol{p(x)}}{\qcol{q(x)}}$$

$$\KL(q\|p)=\log\frac{s}{\sigma}+\frac{\sigma^2+(\mu-m)^2}{2s^2}-\frac12\quad(q=\N(\mu,\sigma^2),\ p=\N(m,s^2))$$

$$p=\N(0,I):\quad \KL(q\|p)=\sum_{j=1}^{d}\Big[-\log\sigma_j+\frac{\sigma_j^2+\mu_j^2}{2}-\frac12\Big]$$

$$\KL(q\|p)=\E_{\theta\sim q}\big[\log q(\theta)-\log p(\theta)\big]\approx\frac1S\sum_{i=1}^{S}\big[\log q(\theta_i)-\log p(\theta_i)\big]$$

**Worked numbers worth reusing.**
1. *Cross-entropy and KL in bits* (`#cross-entropy/4`, `#kl-extra-bits/3`): H(p, q) = ½·2 + ¼·2 + ¼·1 = 1.75; H(p) = 1.5; KL = ½(+1) + ¼(0) + ¼(−1) = 0.25 bits per day.
2. *Asymmetry* (`#kl-asymmetry/2`): q = (0.98, 0.01, 0.01): KL(p‖q) = 1.84 bits, KL(q‖p) = 0.86 bits.
3. *Classification = KL* (`#ce-surprise/3`, `#ce-is-kl/4`): −ln 0.6 = 0.51 nats (0.74 bits); H(one-hot) = 0, so `F.cross_entropy` = KL = 0.511. Distillation (`#soft-labels`): 0.77 = 0.72 + 0.05 nats.
4. *Bernoulli KL* (`#bern-kl/4`): a = 0.8, b = 0.5: 0.8 log 1.6 + 0.2 log 0.4 = 0.376 − 0.183 = 0.19 nats.
5. *Gaussian KL, which q is closer* (`#compare-formula/2`): q₁ = N(0, 2²): log 2 + (4 + 1)/32 − ½ = 0.35; q₂ = N(1, 3²): log(4/3) + 9/32 − ½ = 0.07. MC check (`#kl-sampling/2`): 0.067 from 10,000 draws vs exact 0.069.

**Code** (verbatim from slides).

`#kl-sampling`:
```python
theta = q.sample((S,))
log_q = q.log_prob(theta)
log_p = p.log_prob(theta)
est = (log_q - log_p).mean()
exact = kl_divergence(q, p)
```
`#torch-normal` (PyTorch's own source, as shown):
```python
@register_kl(Normal, Normal)
def _kl_normal_normal(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
```
`#ce-is-kl`: `F.cross_entropy(logits, target)   # 0.511 nats: here equal to KL(p ‖ q)`

**Figures and widgets worth reusing** (`widgets.js` widgets; static helpers in `content.js`; PNG/SVG in `lectures/lecture1/figures/`):
- widget `priorfuncs` (`#prior-functions`): twenty prior draws of the network, θ on hover, prior predictive band.
- `nnDiagram()` (`#tiny-net`, `#theta-a`): the 1-2-1 tanh network with θ_A written on it.
- `threeWay()` (`#three-ways`): Laplace / MCMC / VI on the two-bump target.
- `huffmanSteps()` (`#huffman`) and widget `days-codes` (`#bits-days`): 1.5 vs 1.75 bits per day.
- widget `distill` (`#distill-demo`): KL(teacher ‖ student) falls from 1.07 to 0.009 nats.
- widget `klarea` (`#kl-area`): KL as a net area, with a direction flip. Figure `figures/bnn-evidence-1d.svg`: the 1-D evidence slice.

Deep links: https://nipunbatra.github.io/generative/lectures/lecture1/#intractable/4, #core-problem/2, #kl-extra-bits/3, #ce-is-kl/4, #apps-same/2 (all on the same base URL).

**The lecture's own summary lines.** "The core problem of probabilistic machine learning: how do we compute, or approximate, the posterior p(θ | 𝒟)?" (`core-problem`). "KL divergence: the extra bits we waste" (`kl-extra-bits`). "Never negative, and zero only for the right code" (`kl-properties`). "One principle in many disguises": classification minimises KL(label ‖ prediction), distillation KL(teacher ‖ student); "variational inference minimises KL(q ‖ posterior), the same KL with the order flipped" (`apps-same`).

**Connections.** MAP training is shown to be squared error plus weight decay, the ordinary training of the earlier course. Cross-entropy, used for "every classifier", is re-derived as KL(label ‖ prediction). The diagonal-Gaussian KL against N(0, I) is flagged as "exactly the KL term in our network's ELBO, and in every VAE" (Lecture 25). The deck's "PML L23 / L24" timestamps in speaker notes refer to Nipun's 2023 PML course videos, not to these ES 667 lectures.

**Exam-style questions.**
1. *Why is p(𝒟) intractable for the tiny network, and why does that block the posterior?* Answer: p(𝒟) = ∫ p(𝒟 | θ) p(θ) dθ over ℝ⁷; a grid of 100 points per parameter needs 100⁷ = 10¹⁴ evaluations (about 28 hours at 10⁹ a second), and 10^2,000,000 for a real network. The likelihood and prior are easy for any θ, but without p(𝒟) we cannot normalise them into the posterior.
2. *Why is the classification loss KL(label ‖ prediction) and not the reverse?* Answer: with a one-hot label H(p) = 0, so cross-entropy = KL(p ‖ q). The label is where the data come from (left slot, the weights); the prediction is the code (right slot). KL(prediction ‖ label) would need log q/0 for every wrong class: infinite.
3. *Numeric.* Our weather p = (½, ¼, ¼), their code built for q = (¼, ¼, ½). Compute H(p), H(p, q) and KL(p ‖ q) in bits. Answer: H(p) = ½·1 + ¼·2 + ¼·2 = 1.5. Lengths under q are (2, 2, 1), so H(p, q) = ½·2 + ¼·2 + ¼·1 = 1.75. KL = 0.25 bits per day = ½ log₂(½ ÷ ¼) + ¼ log₂ 1 + ¼ log₂(¼ ÷ ½) = ½ − ¼.

---

## Lecture 24: KL Divergence and Variational Inference, session 2 (Parts 3, 4, 5)

**Aim.** Find the closest q by gradient descent on KL (with Monte Carlo and the reparameterisation trick), see what the KL direction does, then derive the ELBO so that VI works without p(𝒟), and run it on the tiny network.

**Running examples.**
- *Gaussian fit.* p = N(1, 4²), q starts at N(4, 0.5²), Adam lr 0.05, 200 steps; six candidates (best N(1, 3²) at 0.07); a grid of 2,040 q's.
- *2-D covariance.* Target widths 1.5 and 0.75, correlation 0.8. Isotropic q: radius 0.57, KL 0.73. Diagonal: 0.90 and 0.45, KL 0.51. Full: exact, KL 0.
- *Mixture.* p = ½ N(−0.2, 0.4²) + ½ N(1, 0.5²). `kl_divergence` raises NotImplementedError. With `sample()` q stays at N(6, 0.1²), KL about 53 after 400 steps; with `rsample()`, N(0.43, 0.72²), KL 0.056.
- *Forward vs reverse* on two narrow bumps (0.7 / 0.3): reverse 0.36 (left bump) or 1.20 (right), forward 0.72 (spread).
- *Coin.* Beta(5, 3) prior, 17 heads in 30 tosses, posterior Beta(22, 16), log p(𝒟) = −21.40 (ordered sequence, no binomial coefficient).
- *The tiny BNN again.* Mean-field q (14 numbers), 16 draws per step, Adam lr 0.02, 4,000 steps, best of five starts. MCMC (HMC) reference: log p(𝒟) ≈ −1.43.

**Ideas in teaching order.**
1. A grid of μ, σ works in 1-D; for 14 numbers it needs 40¹⁴ ≈ 3 × 10²². So walk downhill: ψ ← ψ − η ∇_ψ KL(q_ψ ‖ p), with ρ = log σ to keep σ positive.
2. In 2-D, q needs a covariance: isotropic, diagonal (mean-field) or full (Σ = LLᵀ, d(d + 1)/2 numbers). "A q that cannot tilt shrinks instead": mean-field VI underestimates uncertainty.
3. Mixtures have no closed-form KL, but KL(q ‖ p) is an average under q, so sample it. With `sample()`, "the code runs, and q does not move".
4. Why: the gradient stops at a stochastic node (autograd gets 0.0013 and −0.0003 vs the true −0.063 and −0.375). The fix: ε ~ N(0, 1), θ = μ + σε, `rsample()`. "Change one word."
5. Direction: reverse KL(q ‖ p) "charges q for mass where p is empty" and hugs one bump; forward KL covers both. VI must use reverse KL, so it can miss modes and be overconfident.
6. Back to Bayes: KL(q ‖ posterior) needs log p(θ | 𝒟), hence p(𝒟). The goal is stated first (`elbo-goal`), then four steps: Bayes in logs; substitute and open the bracket; a constant leaves an average; rearrange and name the ELBO.
7. log p(𝒟) = ELBO + KL, a fixed budget. KL ≥ 0, so the ELBO is a lower bound, and raising the ELBO lowers the KL by exactly as much.
8. Two terms: "fit the data" (make it large) minus "stay near the prior" (make it small). On the coin, the balance point is the posterior.
9. VI for the network in seven lines of PyTorch. q prunes hidden unit 1. Full covariance raises the ELBO by 0.86 nats. Bands stay too narrow against MCMC.

**Key equations** (same macros as Lecture 23: teal p / posterior, purple q, blue evidence, orange likelihood, green prior, rose KL to the posterior):

$$\psi^*=\arg\min_\psi\KL(q_\psi\,\|\,p),\qquad \psi\leftarrow\psi-\eta\,\nabla_\psi\KL(q_\psi\,\|\,p)$$

$$\ncol{\epsilon}\sim\N(0,1),\qquad\theta=\mu+\sigma\ncol{\epsilon},\qquad \frac{\partial\theta}{\partial\mu}=1,\ \ \frac{\partial\theta}{\partial\sigma}=\epsilon$$

$$\pcol{\log p(\theta\mid\mathcal D)}=\ncol{\log p(\mathcal D\mid\theta)}+\textcolor{#147737}{\log p(\theta)}-\dcol{\log p(\mathcal D)}$$

$$\dcol{\log p(\mathcal D)}=\underbrace{\E_q\big[\ncol{\log p(\mathcal D\mid\theta)}+\textcolor{#147737}{\log p(\theta)}-\qcol{\log q(\theta)}\big]}_{\qcol{\ELBO(q)}}+\lcol{\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)}$$

$$\lcol{\KL}\ge0\ \Rightarrow\ \qcol{\ELBO(q)}\le\dcol{\log p(\mathcal D)},\qquad \max_q\ELBO(q)\iff\min_q\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)$$

$$\qcol{\ELBO(q)}=\underbrace{\E_q\big[\ncol{\log p(\mathcal D\mid\theta)}\big]}_{\text{fit the data}}-\underbrace{\KL\big(\qcol{q(\theta)}\,\|\,\textcolor{#147737}{p(\theta)}\big)}_{\text{stay near the prior}},\qquad \text{in code: loss}=-\ELBO$$

(Speaker point: ELBO(q) = E_q[log p(𝒟, θ) − log q(θ)], since p(𝒟 | θ) p(θ) is the joint.)

**Worked numbers worth reusing.**
1. *Reparameterisation by hand* (`#reparam/3`): ε = −0.75, μ = 0, σ = 2, so θ = −1.5, ∂θ/∂μ = 1, ∂θ/∂σ = −0.75.
2. *Direction table* (`#kl-lesson/1`): on the left bump, reverse 0.36 (best) and forward 9.0; across both, 2.09 and 0.72 (best); on the right bump, 1.20 and 21.7.
3. *Lower bound on the coin* (`#elbo-bound/2`): q = Beta(10, 10): log p(𝒟) = −21.40, ELBO = −22.02, gap 0.62 = KL.
4. *How far should q drift* (`#elbo-drift/3`): q = prior Beta(5, 3): fit −22.87, KL 0, ELBO −22.87. q = Beta(22, 16), the posterior: fit up 1.93, KL 0.46, ELBO −21.40 = log p(𝒟). q = spike at 17/30 (Beta(567, 433) in `science.json`): fit up 0.40 more, KL 1.57 more, ELBO −22.57.
5. *VI on the network* (`#bnn-vi/2`, `#vi-families/3`, `#bnn-vs-ref`): the ELBO climbs from −1,235 to about −7.2 (mean-field −7.18 = E[log lik] 8.10 − KL 15.28, from `science.json`). Full covariance −6.32 (w₂, v₂ correlation −0.76). log p(𝒟) ≈ −1.43. Predictive sd at x = −2.5: mean-field 0.110, full 0.117, MCMC 0.215. v₁ ≈ 0.01 with width 0.035: "the first hidden unit is switched off".

**Code** (verbatim from slides).

`#fit-code`:
```python
mu  = Parameter(tensor(4.0))
rho = Parameter(tensor(log(0.5)))
opt = Adam([mu, rho], lr=0.05)
for step in range(200):
    q = Normal(mu, rho.exp())
    loss = kl_divergence(q, p)
    opt.zero_grad()
    loss.backward()
    opt.step()
```
`#rsample-fit` (the `#now-optimise` version is identical but with `q.sample`):
```python
# one word changed
theta = q.rsample((5000,))
log_q = q.log_prob(theta)
log_p = p.log_prob(theta)
loss = (log_q - log_p).mean()
opt.zero_grad()
loss.backward(); opt.step()
```
`#bnn-elbo-code`:
```python
mu  = Parameter(0.5 * randn(7))         # 7 means
rho = Parameter(full((7,), log(0.1)))   # 7 log-widths
q = Normal(mu, rho.exp())               # one bell per parameter
theta = q.rsample((16,))                # 16 sampled networks
log_lik = log_likelihood(theta).mean()  # forward pass
kl = kl_divergence(q, prior).sum()      # closed form
loss = -(log_lik - kl)                  # minus the ELBO
```

**Figures and widgets worth reusing:**
- widget `olive` (`#olive`) and `klGrid()` / widget `klgrid` (`#grid-search`, `#walk-downhill`): q sliding onto p with its loss.
- widget `cov3` (`#cov-fit`): three covariance families fitted to one tilted target.
- `compGraph()` (`#missing-path`, `#reparam`, `#reparam-compare`): stochastic vs deterministic nodes, with the backward pass blocked or flowing (after Lilian Weng). `reparamPicture()` (`#reparam-picture`).
- widget `fwdrev` (`#fwdrev`) + `klWhyFig()` (`#kl-why`): forward vs reverse fits.
- `ceilingFig()` (`#elbo-bound`), `elboLadder()` (`#elbo-max`), `driftFig()` (`#elbo-drift`), widget `elboslider` (`#elbo-slider`): log p(𝒟) as a ceiling, ELBO + KL = constant.
- widget `bnnvi` (`#bnn-vi`) and `viVsHmc()` (`#bnn-vs-ref`; also `figures/bnn-vi-vs-hmc.svg`): VI band vs the MCMC reference.

Deep links (base https://nipunbatra.github.io/generative/lectures/lecture1/): `#now-optimise/2`, `#reparam/3`, `#kl-lesson/2`, `#elbo-goal/3`, `#elbo-drift/3`, `#bnn-vs-ref/2`.

**The lecture's own summary lines.** "The lesson: VI uses reverse KL, so a VI posterior can miss modes and be overconfident" (`kl-lesson`). "log p(𝒟) is one fixed number, so whatever the ELBO gains, the KL to the posterior loses" (`elbo-goal`). "Raise the ELBO, and the KL falls by exactly as much" (`elbo-max`). "Six obstacles, six tools" (`five-tools`): p(𝒟) is impossible → approximate with q; "close" needs a number → KL(q ‖ p); a grid cannot search 14 numbers → gradient descent; no formula for the average → Monte Carlo; no gradient through a draw → θ = μ + σε, `rsample()`; KL to the posterior needs p(𝒟) → maximise the ELBO. Close: "Generative models create this inference problem again" (`next-lecture`).

**Connections.** Directly reuses Lecture 23's Gaussian and diagonal KL formulas, Monte Carlo estimation and the BNN posterior problem. The mixture's latent bump z is set up as the template for the VAE's continuous z. The reparameterisation trick (Kingma and Welling, 2014) is "the step that makes VAEs trainable". Lecture 25 derives the VAE's reconstruction + KL loss from this ELBO (the deck's "Lecture 2" is schedule Lecture 25). MAP (θ_A) returns as "the spike": what the fit term alone would choose.

**Exam-style questions.**
1. *Why does the `sample()` loop on the mixture run without error yet never move q, and what fixes it?* Answer: θ is a stochastic node, so autograd cannot pass the gradient from θ back to μ and σ. Only the path through log q's formula survives, and it averages to about zero. Writing θ = μ + σε with parameter-free ε ~ N(0, 1) (`rsample()`) makes θ deterministic in μ, σ.
2. *Why does maximising the ELBO minimise KL(q ‖ posterior) even though neither p(𝒟) nor that KL can be computed?* Answer: log p(𝒟) = ELBO(q) + KL(q ‖ posterior), and the left side does not depend on q. Every gain in the ELBO is an equal loss in the KL. Since KL ≥ 0, the ELBO is also a lower bound on log p(𝒟).
3. *Numeric.* Coin with prior Beta(5, 3), 17 heads in 30. For q = prior, the fit term is −22.87. For q = Beta(22, 16), the fit is 1.93 higher and KL(q ‖ prior) = 0.46. Find both ELBOs and the KL to the posterior for q = prior, given log p(𝒟) = −21.40. Answer: q = prior: ELBO = −22.87 − 0 = −22.87. q = Beta(22, 16): fit −20.94, ELBO = −20.94 − 0.46 = −21.40 = log p(𝒟), so this q is the posterior. KL(prior ‖ posterior) = −21.40 − (−22.87) = 1.47 nats, the amount the ELBO rose.

---

## The two blocks in one paragraph each

**Concept Bottleneck Models (Lectures 20 and 21).** A heatmap tells you where a black box was sensitive, not what it saw, and gives an expert nothing to fix. A concept bottleneck model forces the prediction through named concepts, x → ĉ = g_θ(x) → ŷ = f_φ(ĉ). A person can correct ĉ and rerun the same head: stripes 0.1 → 1 flips horse to zebra (0.95), and a black-wingtip edit flips the gull to herring (0.91). The model is trained with λ_c L_c + λ_y L_y in independent, sequential or joint schemes; on CUB, joint is most accurate and independent responds best to interventions. The interface is only as good as its parts. A wrong-signed head turns a correct edit into a wrong answer (billing 0.80 → 0.45), and a vocabulary that is not sufficient cannot be rescued by any head (blue circle vs blue square, at most 50 %). Newer variants climb a ladder of shrinking supervision: CEM (vectors per concept), post-hoc CBMs on CLIP phrase scores with a sparse head, and classification by description with no fitting at all. Each rung removes a labelling cost and, with it, a human check.

**KL divergence and variational inference (Lectures 23 and 24).** Bayes for a seven-parameter tanh network is blocked by one number, the evidence p(𝒟), which needs 10¹⁴ grid points. Variational inference instead picks a q_ψ and makes it close to the posterior. "Close" is the KL divergence, built from surprise log 1/p, entropy and codes as the extra bits a code built for q wastes on data from p (0.25 bits per day for the weather). The same KL hides inside the classification loss and distillation. Fitting q is gradient descent on KL. Monte Carlo handles distributions with no formula, and the reparameterisation trick θ = μ + σε lets gradients pass through the draws. Reverse KL makes q hug one mode and be overconfident. Because log p(𝒟) = ELBO + KL(q ‖ posterior) is fixed, maximising the computable ELBO, "fit the data" minus "stay near the prior", minimises the KL we cannot compute. Seven lines of PyTorch then fit the network, with honest but too-narrow error bars.

## The three things a student must remember

**CBM block**
1. In a strict CBM the head reads only the concepts (ŷ = f_φ(ĉ)), so a person can correct a concept and the same head recomputes without retraining.
2. An intervention is only as good as the head and the vocabulary: a wrong-signed weight can make a correct edit hurt, and a missing concept (sufficiency, Y ⊥ X | C) caps accuracy for any head.
3. Cheaper variants (CEM, post-hoc CLIP scores, classification by description) trade concept labels for checks: a match score is not proof that the concept is in the image.

**KL/VI block**
1. KL(p ‖ q) = H(p, q) − H(p) = Σ p log p/q: the extra cost of using q when data come from p. It is ≥ 0, not symmetric, and cross-entropy training is KL(label ‖ prediction).
2. VI minimises the reverse KL(q ‖ posterior), estimated by Monte Carlo and differentiated with the reparameterisation trick θ = μ + σε (`rsample()`). Reverse KL hugs one mode and underestimates uncertainty.
3. log p(𝒟) = ELBO + KL(q ‖ posterior), so maximise ELBO = E_q[log p(𝒟 | θ)] − KL(q ‖ p(θ)): fit the data, stay near the prior. This is the VAE loss of Lecture 25.

## Things that look odd in the sources (for the summary author)
- CBM: Part 9 (leakage) was removed, so the parts jump from 8 to 10, yet the final "What to remember" (s14.7) still says a soft score "leaks to the head". Only PCBM-h's residual illustrates leakage in the taught route.
- KL deck: some story lines still refer to the grades route, now an optional appendix ("the number we built by hand in Way 1" on `cat-kl`; "the sum over grades" on `densities`). `app-distill` says "The third application" although 2.5 now has two (RLHF was dropped). Some references point to "Section 1.5.3", which is no longer a numbered section.
- The KL deck calls itself "Lecture 1" and the VAE "Lecture 2"; in the course these are Lectures 23 and 24 and Lecture 25. The "PML L23 / L24" timestamps in the speaker notes are from the 2023 PML course.
