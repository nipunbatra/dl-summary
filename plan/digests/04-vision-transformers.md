# Digest 04: Vision and language with Transformers (Lectures 16 to 19, note on 22)

Sources read: `origin/main` of `nipunbatra/attention` at `cde4dea` (3 Nov 2026), via `git show`. Decks: `vision1.html` (L16), `clip/index.html` (L17), `clip-addendum/index.html`, `embeddinggemma/README.md`, `from-clip-to-vlm/index.html` (L18 and L19), plus READMEs, the VLM notebook `from-clip-to-vlm/notebooks/vlm-three-connections.ipynb`, and the YouTube descriptions of the four recordings. Live base URL: `https://nipunbatra.github.io/attention/`. Deep links use the slide or frame id as the hash (all ids below were checked on the live pages).

Conventions to carry over:
- Every lecture uses the same photograph: **Newfoundland dog `newfoundland_31` from Oxford-IIIT Pet** (CC BY-SA 4.0). L16 adds a Persian cat (`Persian_98`) for the CLS comparison.
- Shapes are written as rows x features with the batch axis omitted ("196 × 192", "197 × 192"). Row-vector convention: `x_i W`.
- Each deck labels numbers as either measured (saved checkpoint runs) or authored/illustrative. Keep that distinction in the summary deck.

---

## Lecture 16: Vision Transformer for image classification (`vision1.html`)

Recording: https://www.youtube.com/watch?v=AuhrnX_x-i8 (53 min, chapters in the description). PDF: `pdf/vision1.pdf` (55 pages). A 151-page reference deck `vision1-reference.html` holds the implementation lab and extensions.

**Aim**: Give the encoder we already know an image, by turning a photograph into a sequence of patch tokens, updating them with full self-attention, and reading the final CLS row to classify.

**Running example(s)**
- Photo `newfoundland_31`, resized and centre-cropped to a 224 × 224 RGB input. Checkpoint `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (ViT-Tiny), used throughout. Output: 1,000 ImageNet classes.
- Patch size 16: 224 ÷ 16 = 14 per side, 14 × 14 = 196 patches, ordered row by row. One patch is 16 × 16 × 3 = 768 numbers. Followed patch: **P63** (zero-based grid row 4, column 6; "row 5, column 7" counting from one). Pixel tokens would be 224 × 224 = 50,176.
- Width 192; 3 heads of 64; 12 encoder blocks; MLP 192 → 768 → 192 with GELU; pre-LN; 197 learned position vectors (one for CLS).
- Measured P63 content vector: c₆₃ = [−0.852, 1.339, 0.504, …].
- Measured attention: block 4, head 1, query **P74**; source P60 gets 9.088265%, CLS 2.089876%.
- Prediction: Newfoundland 95.73% (0.9572675228), Tibetan mastiff 1.63%, briard 0.67%. ImageNet index 256 = Newfoundland.
- Covering experiments (fresh copy each time, top-left etc. 112 × 112 set to normalized 0 = gray): see worked numbers.

**Ideas in teaching order**
1. Recap the three Transformer families; ViT is the encoder (full attention) branch. "Five questions build the architecture": make tokens, keep location, choose a readout, share information, predict a class.
2. Why not pixels: 50,176 tokens, and attention scores grow as N². Use 16 × 16 patches: 196 tokens.
3. Flatten channel-major (all R, then G, then B) to xᵢ ∈ ℝ⁷⁶⁸; one shared learned projection cᵢ = xᵢ W_E + b_E gives 192 features; stack to C: 196 × 192.
4. Location: add (not concatenate) a learned position vector, eᵢ = cᵢ + pᵢ ("Add WHAT and WHERE").
5. Readout: prepend a learned CLS vector (same start for every image), giving 197 × 192; after the encoder CLS depends on the image. Mean pooling is mentioned as a valid alternative not used by this checkpoint.
6. Reuse the encoder: Q, K, V are three projections of the same normalized image sequence (contrast: translation cross-attention takes Q from the target, K and V from the source). Full attention, no causal mask: all 197 × 197 pairs allowed.
7. One receiver's message is a weighted sum of values; 3 heads of 64 are concatenated to 192, projected, and added to the row (residual); then the MLP adds its own update. 12 blocks, shape stays 197 × 192.
8. After final LayerNorm read CLS (192) and apply Linear(192, 1000). Covering pixels changes the probability; attention maps are "mixing, not a segmentation".
9. CNN versus ViT (inductive bias, local versus content-dependent mixing), patch size versus cost, then the hook: the 1,000-class head is a fixed vocabulary; "What if our class vocabulary could come from words?"

**Key equations** (lecture notation)
- Patch embedding: $c_i = x_i W_E + b_E$, with $x_i$ 1 × 768, $W_E$ 768 × 192, $b_E \in \mathbb R^{192}$; stacked $C = X W_E + b_E$, $X$: 196 × 768, $C$: 196 × 192.
- Position: $e_i = c_i + p_i$ (192 features); sequence $[\,\mathrm{CLS};P_1;\dots;P_{196}] + \text{position} \to 197 \times 192$.
- Q/K/V in one head: $q_i = \mathrm{LN}(e_i)W_Q,\; k_i = \mathrm{LN}(e_i)W_K,\; v_i = \mathrm{LN}(e_i)W_V$ (64 wide); score $q_{74}\cdot k_{60}/\sqrt{64}$ (that is, divide by 8), row softmax gives $a_{74,60}$.
- Block (pre-LN): $U = E + \mathrm{MSA}(\mathrm{LN}(E))$, $E_{\text{next}} = U + \mathrm{MLP}(\mathrm{LN}(U))$.
- Readout and head: $h_{\text{CLS}}$ = CLS row of the final LN (192); class score $s_c = h_{\text{CLS}}^\top w_c + b_c$ for 1,000 classes.

**Worked numbers worth reusing**
- Token arithmetic (`story-patch-grid`, `story-patch-768`, `story-pixel-budget`): 224 ÷ 16 = 14; 14 × 14 = 196; 16 × 16 × 3 = 768; with CLS 197; pixels 224 × 224 = 50,176.
- Toy value mixture (`story-value-mixture`): 0.6[2,0] + 0.3[0,1] + 0.1[1,1] = [1.3, 0.4] (chosen weights, not from the checkpoint).
- Patch size versus cost (`story-patch-sizes`, `story-patch-cost`): patches 32/16/8 give 49/196/784 tokens; with CLS 197² = 38,809 versus 785² = 616,225 scores per head, about 15.88 times (exactly 16 times for patch-only counts).
- Quadrant covers (`story-cover-results`), P(Newfoundland) from 95.73%: top left 83.00% (−12.73 points), top right 79.62% (−16.11), bottom left 84.54% (−11.19), bottom right 82.16% (−13.57). All still predict Newfoundland; drops are not additive.
- Single-patch covers (`story-cover-small`): 196 separate tests; largest drop at P78, 95.73% → 91.89% (−3.84 points; 3.836018).

**Code** (verbatim)

Covering one quarter (`story-cover-code`):
```python
covered = x.clone() # x: (1, 3, 224, 224), normalized
covered[:, :, :112, :112] = 0 # gray RGB; 112 × 112 pixels
model.eval()
with torch.inference_mode():
    before = model(x).softmax(-1)[0, 256]
    after = model(covered).softmax(-1)[0, 256]
```
The whole ViT in six lines (`story-six-lines`, comments trimmed):
```python
x = patch_embed(image) # (B, 196, 192)
x = prepend_cls(x) + pos # (B, 197, 192)
for block in blocks:
    x = block(x) # (B, 197, 192)
h = norm(x)[:, 0] # (B, 192)
logits = head(h) # (B, 1000)
```
Patchify: the main deck has no unfold code; the reference deck `vision1-reference.html` (implementation lab) has:
```python
linear = nn.Linear(768, 192, bias=True)

def project_with_linear(images, linear):
    patches = F.unfold(images, kernel_size=16, stride=16)
    # B × 768 × 196; each patch: all R, then G, then B
    # Within each channel: left to right, top to bottom
    patches = patches.transpose(1, 2)
    # B × 196 × 768: one flattened patch per row
    return linear(patches)  # B × 196 × 192
```
(It also shows the equivalent `nn.Conv2d(3, 192, kernel_size=16, stride=16)` followed by `grid.flatten(2).transpose(1, 2)`.)

**Figures and widgets worth reusing**
The main deck embeds its images as base64; source files exist in `figures/vision1/` (generated by `src/build_vision1_lesson.py`; I did not verify a one-to-one match with each frame).
1. `figures/vision1/newfoundland_31.jpg` (JPG) and `figures/vision1/model-input.png` (PNG, the exact 224 × 224 crop): the running photo.
2. Frame `story-patch-grid`: photo split into the 14 × 14 grid. Live: https://nipunbatra.github.io/attention/vision1.html#story-patch-grid
3. Frame `story-prepared`: pipeline strip IMAGE 3 × 224 × 224 → PATCHES → 196 × 192 → +POSITION +CLS 197 × 192 → ENCODER × 12 → FINAL CLS 192 → 1,000 logits. Live: https://nipunbatra.github.io/attention/vision1.html#story-prepared
4. Frame `story-measured-attention`: measured attention map for query P74 (block 4, head 1); links to `vision1-explorer.html` ("Explore the nine examples"). Live: https://nipunbatra.github.io/attention/vision1.html#story-measured-attention
5. Frame `story-cover-results` (files `figures/vision1/cover-1.svg` to `cover-4.svg`, `figures/vision1/occlusion-small-result.svg`): quadrant and single-patch covering. Live: https://nipunbatra.github.io/attention/vision1.html#story-cover-results
6. Frame `story-summary-shapes`: "Follow the whole model through its shapes". Live: https://nipunbatra.github.io/attention/vision1.html#story-summary-shapes (also `#story-six-lines`, `#story-fixed-vocabulary`).

**The lecture's own summary lines**
- "Turn patches into tokens, update them with the encoder, then use final CLS to predict the class."
- "Pixels 3 × 224 × 224 → Patch projection 196 × 192 → + CLS + position 197 × 192 → 12 encoder blocks 197 × 192 → Final LN → read CLS 192 → Class head 1,000 logits. The blocks keep the same shape. Each row now includes context."
- "CNNs gather wider context through local layers. Global ViT attention can connect distant patches in a single block."
- "Its 1,000 output classes form a fixed vocabulary." Then: "We will pick up this question in the CLIP lecture."

**Connections**
- Builds on the previous lecture "From Attention to Applications" (called "Transformers beyond next-token prediction" / "Beyond Attention" in the decks): the three model families, the CLS sentence readout, multi-head self-attention, and the translation cross-attention contrast.
- Reused by L17 (CLIP's image branch is a ViT read at CLS; the CLIP deck opens with this ViT-Tiny head `nn.Linear(192,1000)`), by L18/L19 (the patch rows, not CLS, become the visual prefix or memory), and re-implemented in L22.

**Three exam-style questions with answers**
1. *Why does the ViT add the position vector instead of relying on the patch projection to encode location?* The projection $W_E, b_E$ is shared by all 196 patches and has no position index, so identical crops at different places would get identical content vectors; the learned table adds an explicit "where" vector of the same width (addition keeps 192 features).
2. *An attention map shows P60 receiving 9.09% of P74's weight. Why does the deck refuse to call this an explanation of "Newfoundland"?* It is one head in one block, measures mixing weights over (contextual) value vectors, and the patch rows carry information from the whole image; the covering experiment measures a different quantity (change in output probability), and the deck stresses "An attention map visualizes mixing, not a segmentation".
3. *Numeric.* A 224 × 224 image with 32 × 32 patches. Tokens and attention scores per head (with CLS)? 224 ÷ 32 = 7, so 7 × 7 = 49 patches, 50 with CLS, 50² = 2,500 scores. With 8 × 8 patches: 28 × 28 = 784, 785 with CLS, 785² = 616,225 scores, about 246 times more than the 32-pixel case.

---

## Lecture 17: CLIP, from fixed labels to language-defined vision (`clip/index.html`, plus addendum note)

Recording: https://www.youtube.com/watch?v=W3WhuZczsNQ (68 min). PDF `clip/clip.pdf` (168 pages). Subtitle "Attention follow-up 1". Applications Colab: `clip/notebooks/clip-applications.ipynb`. The schedule also lists the addendum (`clip-addendum/`) and the EmbeddingGemma lab under L17, but the addendum was published on 13 Oct 2026 and its demos match the L19 recording (see L19).

**Aim**: Replace the stored class rows of the ViT head with vectors computed from text, so that an image can be compared with any description, and derive the symmetric contrastive loss that makes those comparisons meaningful.

**Running example(s)**
- Opening tour: 15 applications with frozen pretrained CLIP ViT-B/32 (saved Node CPU q8 measurements, raw cosines). Newfoundland vs "a photo of a dog" +0.2723, "a photo of a cat" +0.2152, "a photo of a bird" +0.2184. Others: coffee/tea/juice/soup, text search "something to drink" over a 22-image gallery, a hat added to a portrait, mug colour change, brick kiln aerial tile, chimney, stripes, chest radiograph, sketch, the wrong-menu trap (no dog option: bird wins), negation ("a photo of a cat" 0.2737 vs "a photo without a cat" 0.2699).
- Real dimensions (ViT-B/32): $h_I \in \mathbb R^{768}$ (final CLS after LN), $W_I$ 768 × 512; text branch is a causal Transformer read at EOT, $h_T = \mathrm{LN}(h_{EOT}) \in \mathbb R^{512}$, $W_T$ 512 × 512; both normalized to 512-D unit vectors $u$, $v$.
- Toy training run: 12 authored pairs (cat, dog, car, bicycle, airplane, tree, shoe, mug, banana, chair, building, bird); batch 1 = cat, dog, car; 3-D vectors; **τ = 0.5**. Raw outputs: images cat [2.0, 0.5, 0.2], dog [0.4, 2.0, 0.1], car [0.2, 0.3, 2.0]; texts cat [1.6, 0.8, 0.4], dog [0.7, 1.4, 0.3], car [0.4, 0.5, 1.5].
- Notation (stated on `main-scale`): **C** = cosines, **S** = logits = C/τ, **P** = row softmax, **Q** = column softmax, calligraphic **L** = loss. (The source interactive used different letters; the lecture says it keeps U, V, C.)
- Closing: the hat pair again, 512-D measured vectors, words "hat", "boat", "cup", "cat".

**Ideas in teaching order**
1. Applications first: "An image goes in. We choose the descriptions to compare it with." CLIP ranks supplied descriptions; it writes no text.
2. The ImageNet head stores 1,000 class vectors ($s = Wh + b$); "Let the words supply the class vector": $u^\top v_{\text{dog}}$. (This is the zero-shot classification idea; the slides never use the phrase "zero-shot".)
3. Two encoders: image ViT read at CLS (full attention), causal text Transformer read at EOT; each projects and normalizes; they meet only at the dot product.
4. Unit length leaves only the angle: $u^\top v = \cos\theta$ (cosine 1, 0, −1).
5. Equal width is not meaning; collapse (every caption scores 1) shows why a paired caption "must beat its competitors".
6. Follow one batch: project, normalize, stack U and V, $C = UV^\top$ (diagonal = observed pairs).
7. Scale by temperature, row softmax P (image → text), go back to S, column softmax Q (text → image); cross-entropy reads the paired cells; average both directions.
8. Gradients update both encoders, both projections and the learned score scale; batches, epochs; web scale (400M pairs, batch 32,768); false negatives and multi-positive targets.
9. Return to the hat: subtract unit image vectors, renormalize, compare with word vectors; summary and Colab.

**Key equations**
- Branches: $z_I = h_I W_I,\ u = z_I/\lVert z_I\rVert$; $h_T = \mathrm{LN}(h_{EOT}),\ z_T = h_T W_T,\ v = z_T/\lVert z_T\rVert$; match score $u^\top v = \lVert u\rVert\lVert v\rVert\cos\theta = \cos\theta$.
- Fixed head versus language-defined classes: $s_{\text{dog}} = w_{\text{dog}}^\top h + b_{\text{dog}}$ (stored row) versus $u^\top v_{\text{dog}}$ (vector computed from "a photo of a dog"). Zero-shot choice = the description with the largest $u^\top v_c$ among those supplied.
- $C = UV^\top,\quad S = C/\tau$ (summary slide writes $S = \alpha UV^\top$, $\alpha = 1/\tau$ the learned score scale).
- $P_{ij} = \dfrac{e^{S_{ij}}}{\sum_k e^{S_{ik}}},\qquad Q_{ij} = \dfrac{e^{S_{ij}}}{\sum_k e^{S_{kj}}}$ (rows are images, columns are texts in both).
- $\mathcal L_{I\to T} = -\frac1N\sum_i \ln P_{ii},\ \mathcal L_{T\to I} = -\frac1N\sum_j \ln Q_{jj},\ \mathcal L = \frac{\mathcal L_{I\to T}+\mathcal L_{T\to I}}2 = -\frac1{2N}\big(\sum_i\ln P_{ii}+\sum_j\ln Q_{jj}\big)$; tie baseline $\mathcal L_{\text{tie}} = \ln N$.
- Change direction: $d = \frac{a-b}{\lVert a-b\rVert_2}$, $d^\top v = \frac{a^\top v - b^\top v}{\lVert a-b\rVert_2}$.

**Worked numbers worth reusing** (all from the authored 3-D batch, τ = 0.5)
- Projection and normalization (`worksheet-projection`, `main-projection-coordinates`, `worksheet-normalize`): $h^I_1 = [1,2,1,1]$, $W_I$ columns give $z^I_1 = [2, 0.5, 0.2]$; $\lVert z\rVert = 2.071232$; $u_1 = [0.9656, 0.2414, 0.0966]$. Text $[1.6,0.8,0.4]$, norm 1.833030, $v_1 = [0.8729, 0.4364, 0.2182]$; $u_1\cdot v_1 = 0.969281$.
- Cosine matrix C (`cosine-matrix`), rows cat/dog/car images, columns cat/dog/car captions: [0.969, 0.654, 0.400; 0.609, 0.956, 0.393; 0.365, 0.358, 0.975]. Logits S = C/0.5: [1.939, 1.309, 0.799; 1.218, 1.911, 0.787; 0.730, 0.716, 1.949].
- Cat row softmax (`worksheet-row-weights` to `main-row-shares`): e^S = 6.9487, 3.7013, 2.2239; row sum 12.8740; P₁ = [0.5398, 0.2875, 0.1727]. Full P = [0.540, 0.288, 0.173; 0.274, 0.548, 0.178; 0.186, 0.184, 0.630]. Cat column (`main-col-denominator`): 6.9487 + 3.3812 + 2.0754 = 12.405358, Q₁₁ = 0.560141. Full Q = [0.560, 0.296, 0.194; 0.273, 0.540, 0.192; 0.167, 0.164, 0.614]. "Same pair and numerator. Different competition."
- Loss (`row-loss`, `worksheet-column-loss`, `symmetric-loss`, `closing-summary-losses`): −ln P diagonal 0.616649, 0.601521, 0.461888 → $\mathcal L_{I\to T}$ = 0.560020; −ln Q diagonal 0.579567, 0.615280, 0.488252 → $\mathcal L_{T\to I}$ = 0.561033; **$\mathcal L$ = 0.560526**, versus tie baseline ln 3 = 1.098612. One SGD step (η = 0.25) → 0.539743; 100 steps on the same batch → 0.124937; batch 2 (bicycle, airplane, tree) 0.452386 → 0.442814.
- Hat (`closing-results`): before/after cosines with "hat" 0.211620 / 0.250661; $\lVert a-b\rVert = 0.483430$; (0.250661 − 0.211620) ÷ 0.483430 ≈ **0.080758**; boat +0.025401, cup +0.013169, cat +0.002556. Web scale (`main-dataset-batch`): 4² = 16, 1024² = 1,048,576 scores; 400M records is not a 400M × 400M matrix.

**Code** (verbatim, `closing-code`, imports and model load lines kept; 12 lines)
```python
model, preprocess = clip.load("ViT-B/32", device="cpu")
model.eval()
images = torch.stack([preprocess(before), preprocess(after)])
words = ["hat", "boat", "cup", "cat"]
with torch.no_grad():
    u = F.normalize(model.encode_image(images).float(), dim=-1)
    v = F.normalize(model.encode_text(clip.tokenize(words)).float(), dim=-1)
    delta = u[1] - u[0]                 # after minus before
    if delta.norm() < 1e-6:
        raise ValueError("No stable change direction")
    d = F.normalize(delta, dim=-1)      # normalize again
    scores = v @ d                     # one cosine per word
```
The slide asks "Which lines change if we compare one image with descriptions instead of describing an edit?" (answer: score `v @ u[0]` directly; that is zero-shot classification). The main deck shows the loss only as maths; it notes that PyTorch `F.cross_entropy` takes logits, so one would apply it to S row-wise and to Sᵀ. I found no verbatim CLIP-loss code block in the main deck.

**Figures and widgets worth reusing**
Images are inline; `clip/figures/` holds only `recap-cls-source.svg` and `recap-families-source.svg` (SVG recap sources).
1. `mot-new-labels-result` / `mot-wrong-menu-result`: measured cosine menus for the Newfoundland photo. Live: https://nipunbatra.github.io/attention/clip/#mot-wrong-menu-result
2. `main-text-attention-mask`: full image mask versus causal text mask (SOT, a, dog, EOT).
3. `closing-summary-encoders`: two-branch architecture summary (CLS and EOT, projection, normalization, 512-D, cosine). Live: https://nipunbatra.github.io/attention/clip/#closing-summary-encoders
4. `cosine-matrix`: the 3 × 3 C matrix with the diagonal marked. Live: https://nipunbatra.github.io/attention/clip/#cosine-matrix
5. `closing-summary-probabilities`: S, P and Q side by side with paired cells outlined. Live: https://nipunbatra.github.io/attention/clip/#closing-summary-probabilities
6. Widgets: `main-scale-slider` (temperature slider on the cat row) and `main-matrix-selector` (pick any cell, switch row/column). Compact-loss slide: https://nipunbatra.github.io/attention/clip/#main-compact-loss

**The lecture's own summary lines**
- "Two encoders build a shared comparison space": "Read one vector from each branch, project, normalize, then compare."
- "The loss teaches the two branches what should match": "Pairing supplies the targets. Gradients change the shared model parameters."
- "One score matrix gives two probability matrices": "Keep P. Compute Q from S again."
- "Average both directions, then backpropagate": "One batch loss: average all six penalties, then backpropagate."
- "Once trained, the same vectors answer different questions": "Reuse the frozen encoders. Choose which vectors to compare."
- Section 8 key idea: "Two encoders, one training objective, many ways to compare."

**Connections**
- Builds on L16 (the ViT image branch and its fixed 1,000-row head), the earlier CLS/EOT readout and causal masks from "Beyond Attention", softmax and cross-entropy (the loss is ordinary cross-entropy applied per row and per column), and embeddings/cosine similarity.
- Reused by L18/L19 (revision slides `clip-recall`, `clip-training-recap`; the ViT patch rows feed the VLM) and by the addendum and EmbeddingGemma lab (same unit-vector dot product for search). The schedule's L21 topics "post-hoc bottlenecks and classification by description" plainly rest on CLIP-style text-defined class vectors; I did not read those decks.

**Three exam-style questions with answers**
1. *Why must both u and v be unit length before the dot product?* Then $u^\top v = \cos\theta$ depends only on direction, so a long vector cannot win by magnitude; the temperature then sets how sharply softmax separates the cosines without changing their ranking.
2. *Why compute Q from S rather than from P, and why both directions?* Row softmax answers "which caption belongs to this image" (competitors: other captions); column softmax answers "which image belongs to this caption" (competitors: other images). Q is a separate normalization of S, not of P; training both makes retrieval work in both directions, and CLIP averages them.
3. *Numeric.* Two pairs, C = [[0.8, 0.2], [0.1, 0.6]], τ = 0.5. S = [[1.6, 0.4], [0.2, 1.2]]. P₁₁ = 1/(1+e^−1.2) = 0.7685, P₂₂ = 1/(1+e^−1.0) = 0.7311 → $\mathcal L_{I\to T}$ = (0.2633 + 0.3133)/2 = 0.2883. Q₁₁ = 1/(1+e^−1.4) = 0.8022, Q₂₂ = 1/(1+e^−0.8) = 0.6900 → $\mathcal L_{T\to I}$ = (0.2204 + 0.3711)/2 = 0.2958. $\mathcal L$ = 0.2920, below the tie baseline ln 2 = 0.6931.

---

## Lecture 18: From CLIP to Vision-Language Models, Part 1 (`from-clip-to-vlm/index.html`, slides 1 to about 104)

Recording: https://www.youtube.com/watch?v=S6X9_KWq_44 (63 min). Deck: 178 slides (145 main + 33 optional), "Attention follow-up 2", PDF `from-clip-to-vlm/from-clip-to-vlm.pdf`. Colab: `notebooks/vlm-three-connections.ipynb`; VLM lab `demo.html` (SmolVLM-256M versus SmolVLM-500M, live WebGPU or recorded).

**Where L18 ends and L19 begins.** One deck serves both. The L18 YouTube description says it covers "a CLIP recap and multimodal demonstrations" then "two ways to connect a vision encoder to a language decoder: visual tokens as a prefix, and a separate visual memory accessed through cross-attention ... The lecture ends with Q&A and a brief preview of compressing visual tokens into learned summaries." So L18 = Sections 01 to 04 (slides 1 to 104, `cover` to `cross-code`, plus `prefix-cross`), with at most a glance at Section 05. L19 (recorded 13 Oct 2026) "revisit[s] visual prefixes and cross-attention", then does Section 05 compression (`section-04`, slide 105 onward), Section 06 training, Section 07 summary, Backup A multiple images, and the "practical embedding and speech-to-text demonstrations", which match `clip-addendum/` (DigUp and speech-from-embeddings, committed 13 Oct 2026 07:21 IST). Supporting evidence: commit "Explain learned-query summarization step by step" on 9 Oct, after L18 was published on 8 Oct. The exact minute of the split inside the deck is my inference from these descriptions, not stated on any slide.

**Aim**: Turn CLIP's "compare an image with supplied descriptions" into "generate an answer from the image", by deciding how image feature rows reach an ordinary causal language decoder.

**Running example(s)**
- Same Newfoundland photo; prompt "What animal is shown?"; answer prefixes "It is a" and (second question) "What is behind the dog?" / "Behind it are"; authored continuations "dog" and "trees".
- Ten application questions with authored reference answers (laptop plugged in, "left of the mug" → "The red ball.", apples outside the bowl → "Three.", invoice ES667-042 and GST 180 / 1000 × 100 = 18%, chart March +45 µg/m³, geometry x = 5, before/after room, settings screenshot, thermal figure 62 °C, false premise "What color is the bicycle?" → "No bicycle is visible.").
- Toy prefix arithmetic: 3 × 4 = 12 patches, $H_v$ 12 × 2 (d_v = 2), text width d_L = 3, $W_p$ = [[1,0,1],[0,1,1]].
- Real notebook shapes: DeiT-Tiny (`facebook/deit-tiny-patch16-224`) 224 × 224, 196 patch rows × 192 (CLS dropped); DistilGPT2 width 768, vocabulary 50,257. Exact prompt "Question: What animal is shown?\nAnswer:" = 10 token IDs (24361, 25, 1867, 5044, 318, 3402, 30, 198, 33706, 25); decoder input 196 + 10 = 206 × 768. " A" = ID 317.
- Toy cross-attention: same 12 patches, 2-D keys and values, $W_Q$ = [[0,1],[0.5,−0.5]], $W_K$ swaps coordinates, $W_V$ = [[1,0],[1,2]], $W_O$ = [[1,0],[0.5,1]].

**Ideas in teaching order**
1. Revision of CLIP (two encoders, P/Q loss, applications); "Next task: generate an answer from the image".
2. What can a VLM do? Ten question families; "What must survive the visual representation?" (counting, OCR, UI, false premise); live SmolVLM lab.
3. Recall the decoder-only portrait ("Raghav goes to → school"); "Keep this decoder; give it access to the image."
4. Three ideas: Idea 1 prepend the image as visual tokens; Idea 2 let the decoder look at a visual memory; Idea 3 summarize the image, then use either connection. "Ideas 1 and 2 choose the access route. Idea 3 can be used with either."
5. Idea 1 in detail: keep all N patch rows, leave out CLS; a shared projector fixes width ($Z = H_v W_p$, N rows unchanged); prepend before USER / question / ASST role tokens; one causal mask over the whole sequence (Gemma 3 alternative: bidirectional within the image block). "This is the basic LLaVA-style connection" (LLaVA: CLIP ViT-L/14, linear W, Vicuna).
6. Generation walkthrough with real IDs: read the final ":" state, 50,257 logits, softmax, choose " A" (317), append $E[317]$, 207 positions.
7. Idea 2 in detail: the translation decoder, with English replaced by an image. Text state supplies q; every image row supplies a paired k and v; scaled scores, softmax over all 12 positions, value mixture c, project and add to the text state.
8. Two questions on the same image give two queries and two messages from the same K and V; "Check: has cross-attention already answered 'dog'?" No: the vocabulary head still chooses words.
9. "Two ways for language to access vision": in-sequence prefix versus separate memory; Idea 2 code.

**Key equations** (notation as on the slides; most are written in text, not KaTeX)
- Projector: $Z = H_v W_p$, $H_v$: N × d_V, $W_p$: d_V × d_L, $Z$: N × d_L (bias omitted; "Real connectors can use an MLP").
- Prefix input: `concat([visual, text])` gives (N + T) × d_LM; next token from `lm_head(hidden[-1])`.
- Cross-attention, one head, current text position t: $q = h_t W_Q$, $K = H_V W_K$, $V = H_V W_V$, $s_i = q\cdot k_i/\sqrt{d_k}$, $\alpha_i = \exp(s_i)/\sum_j \exp(s_j)$, $c = \sum_i \alpha_i v_i$, $h'_t = h_t + cW_O$. Memory batch shape B × N × d_V.
- Vocabulary head: $h_{205}$ (1 × 768) times $W_{\text{vocab}}$ (768 × 50,257) gives 50,257 logits.

**Worked numbers worth reusing**
- Projector (`projector-step-1` to `shared-projector`): $h_1 = [2,1]$: 2 × 1 + 1 × 0 = 2; 2 × 0 + 1 × 1 = 1; 2 × 1 + 1 × 1 = 3, so [2,1,3]; also [1,3] → [1,3,4], [0,2] → [0,2,2]. Check slide: 12 × 2 times 2 × 3 gives 12 × 3 ("Still 12 rows").
- Exact input (`token-context`): 196 × 192 → projector 192 → 768 → 196 × 768; plus 10 prompt embeddings = 206 × 768.
- Toy vocabulary head (`token-logits`, `token-distribution`): h = [1,2], logits [3, 1, 0, −1] for " A", " The", " dog", EOS; softmax ≈ [.8310, .1125, .0414, .0152] (exp sum 24.172); greedy " A" → ID 317 → 207 positions (`generate-0`).
- Cross-attention (`cross-q` to `cross-return`): $h_t = [1,2]$ → q = [1,0]; patch 6: $h_6 = [0,3]$, $k_6 = [3,0]$, $v_6 = [3,6]$; score 3/√2 = 2.121; exp 8.342; sum of all twelve 37.023; α₆ = 0.225; c = [2.867, 4.031]; $cW_O$ = [4.883, 4.031]; $h'_t$ = [5.883, 6.031].
- Second question (`cross-numeric-inputs` to `cross-numeric-mix`): $h_t = [3,0]$ → q = [0,3]; patch 1 score 9/√2 = 6.364; α₁ = α₄ = 0.396; c = [2.94, 0.34]. "Changing q can change the visual message without recomputing K or V."

**Code** (verbatim from the notebook `vlm-three-connections.ipynb`, which runs the real DeiT-Tiny + DistilGPT2 wiring; connectors untrained)
```python
def prefix(ids):
    visual = projector(H_v)
    text = lm.get_input_embeddings()(ids)
    x = torch.cat([visual, text], dim=1)
    h = lm.transformer(inputs_embeds=x).last_hidden_state
    return lm.lm_head(h[:, -1]), {"visual prefix": visual, "decoder input": x, "decoder states": h}
```
```python
def separate_memory(ids):
    memory = projector(H_v)
    text = lm.get_input_embeddings()(ids)
    h = lm.transformer(inputs_embeds=text, encoder_hidden_states=memory).last_hidden_state
    return lm.lm_head(h[:, -1]), {"separate image memory": memory, "decoder input": text, "decoder states": h}
```
Slide pseudocode `cross-code` ("Idea 2, end to end in one function"; the slide prints shapes in a separate column, shown here as comments):
```python
def next_token(image, question, answer_so_far):
    memory = vision_encoder(image).features        # N × dV
    ids = tokenize(question, answer_so_far)        # T token IDs
    text = embed(ids)                              # T × dLM
    for block in decoder_blocks:
        text = block.causal_self_attn(text)        # T × dLM
        text = block.cross_attn(text, memory)      # T × dLM
        text = block.mlp(text)                     # T × dLM
    return argmax(lm_head(text[-1]))               # Next text token ID
```
(`prefix-code` is the same shape for Idea 1: `patches = vit(image).patch_states`, `visual = projector(patches)`, `inputs = concat([visual, text], axis="tokens")`, `hidden = causal_decoder(inputs)`, `scores = lm_head(hidden[-1])`.)

**Figures and widgets worth reusing**
1. `visual-states`, `cross-intuition`, `compression-intuition`: the three idea diagrams (inline SVG, source-left/decoder-right grammar). Live: https://nipunbatra.github.io/attention/from-clip-to-vlm/#compression-intuition
2. `token-context`: real token IDs and the 206 × 768 input. Live: https://nipunbatra.github.io/attention/from-clip-to-vlm/#token-context
3. `cross-numeric-scale`: the calculated weights for both queries painted on the dog's 12 patches. Live: https://nipunbatra.github.io/attention/from-clip-to-vlm/#cross-numeric-scale
4. `mask-full`: LLaVA/LLaMA lower-triangular mask versus Gemma 3 bidirectional image block.
5. `from-clip-to-vlm/figures/papers/llava-figure-1.png` (PNG, original LLaVA Figure 1, attribution in `figures/papers/ATTRIBUTION.md`), on slide `llava-paper`.
6. Application images: `from-clip-to-vlm/figures/desk.svg`, `invoice.svg`, `chart.svg`, `no-bicycle.jpg` (SVG/JPG); VLM lab `from-clip-to-vlm/demo.html`.

**The lecture's own summary lines** (L18 has no closing summary of its own; these are its checkpoint slides)
- "Ideas 1 and 2 choose the access route. Idea 3 can be used with either."
- "Projected patch vectors occupy earlier positions; they do not become word IDs."
- "Different questions need different visual evidence from the same image."
- "Visual lookup changes the language state; the language head chooses words."
- "In-sequence visual context or a separate visual memory."

**Connections**
- Builds on L17 (CLIP revision; CLIP ViT-L/14 is LLaVA's encoder), L16 (patch rows and why CLS is dropped here), and the earlier decoder-only and English-to-Hindi translation (encoder-decoder cross-attention) lectures.
- Cross-attention with queries from one sequence and keys/values from image features is the same operation used by DETR-style detection decoders; the attention repo has an `object-detection/` deck, but I did not read it, so treat that link as a plain statement about the mechanism.

**Three exam-style questions with answers**
1. *In Idea 1, what does the projector change and what does it not change?* It changes width (d_V → d_L, here 192 → 768) row by row with one shared matrix; it does not change the number of visual tokens (196 stay 196) or turn them into word IDs.
2. *In Idea 2, the same image answers "What animal is shown?" and "What is behind the dog?" differently. What changes between the two lookups?* Only the query: it comes from the text state at the last supplied token ("a" versus "are"), so the softmax weights over the same K differ and mix the same V into a different message c.
3. *Numeric.* q = [1,0], twelve keys with first coordinates [0,1,1,0,1,3,3,1,0,2,2,0], d_k = 2. Scores are x/√2. Exponentials: 1 (four keys), e^0.707 = 2.028 (four), e^2.121 = 8.342 (two), e^1.414 = 4.113 (two). Sum = 4 + 8.113 + 16.684 + 8.227 = 37.023. Weight of patch 6 = 8.342 / 37.023 = 0.225.

---

## Lecture 19: Vision-Language Models, Part 2 (same deck, slides 105 to 154, plus `clip-addendum/`)

Recording: https://www.youtube.com/watch?v=IIFrtT3_nIM (69 min, recorded 13 Oct 2026). Description: revisits prefixes and cross-attention "through queries, keys, values and tensor shapes", introduces learned summary tokens, training and answer generation, multiple-image questions, and ends with "practical embedding and speech-to-text demonstrations".

**Aim**: Shorten the visual context with learned query summaries, then show how a VLM connection is trained with a next-token answer loss, and extend the input to several ordered images.

**Running example(s)**
- Real notebook sizes again: 224 × 224, 14 × 14 = 196 patch rows × 192; prompt "Question: What animal is shown?\nAnswer: It is a" = 13 DistilGPT2 tokens; prefix 196 + 13 = **209** positions; with 8 summaries 8 + 13 = **21** (188 fewer).
- Summarizer: S = 8 learned slots × 192 (`0.02 * torch.randn(1, 8, 192)`), `nn.MultiheadAttention(192, 3)` (3 heads × 64), projector 192 → 768: 196 × 192 → 8 × 192 → 8 × 768. Second image: a synthetic invoice (ES667-042, GST ₹180, total ₹1,180).
- Toy readers: q₁ = [1,0], q₂ = [0,3] over the same 12 toy keys/values as L18 → z₁ = [2.867, 4.031], z₂ = [2.94, 0.34].
- Training: reference answer "A black dog." → IDs 317, 2042, 3290, 13, 50256 (EOS); 196 + 10 + 5 = 211 positions × 768.
- Multi-image: two authored room drawings A (before) and B (after); reference "The chair moved to the right; the cup disappeared."
- Addendum: DigUp local file search (EmbeddingGemma 2, 768-D float16 vectors in SQLite) and a speech demo (author-reported 5.18% WER on LibriSpeech test-clean, about 13 s per hour of audio on a base M3 MacBook Air).

**Ideas in teaching order**
1. Cost of Idea 1: 209 positions in every decoder layer; KV cache stores 196 image keys and values; "Caching reuses K and V; it does not remove the image positions."
2. "Keep the text; shorten the visual prefix": 8 learned readers give 8 image-dependent mixtures, not 8 selected patches.
3. Stored query slots S versus computed summaries Z; initialize once (1,536 numbers), reuse for every image.
4. Reader Q from S, K and V from the image H (196 × 64 per head); soft lookup over all 196 keys; mix values, join 3 heads, W_O → z₁; "First reduce the count; then match the language width."
5. Training the slots: only the answer loss; "we never supply a 'correct summary vector'". Same readers on a different image give a different summary; architectures named: LLaVA (prefix of all patches), Flamingo (Perceiver Resampler + cross-attention memory), BLIP-2 (Q-Former summaries + prefix into OPT).
6. Training the connection: the dataset supplies the reference; read p[317] at the last ":"; teacher forcing versus greedy inference; five shifted targets; labels −100 on context; causal mask still hides the future.
7. Freeze ViT and LM, train the projector; gradients still flow through the frozen LM. LLaVA stage 1 (captions, update projector) then stage 2 (instructions, update projector and LM, vision fixed).
8. Summary: "ViT represents. CLIP compares. A VLM generates."; the three connections; the full image-to-next-word path.
9. Optional A: two images encoded and summarized separately, packed in order with "Before:"/"After:" text, one causal decoder reads both. Then the addendum demos (search by embeddings; a trained reader on frozen embeddings for speech).

**Key equations**
- Summarizer shapes: $H \in \mathbb R^{196\times192} \xrightarrow{\text{reader}(S,H)} Z \in \mathbb R^{8\times192} \xrightarrow{\text{projector}} \mathbb R^{8\times768}$; per head $Q$: 8 × 64, $K, V$: 196 × 64.
- Soft lookup per slot and head: $\alpha_{1i} = \mathrm{softmax}_i\!\left(q_1\cdot k_i/\sqrt{64}\right)$, $c_1 = \sum_i \alpha_{1i} v_i$ (1 × 64); concat 3 heads → 192 → $W_O$ → $z_1$.
- Generation model: $p(y_t \mid y_{<t}, I, P)$ (slide `image-counterfactual`; I = image, P = prompt).
- Answer loss: mean of $-\ln p(\text{target}_t)$ over the answer tokens only (context labels set to −100); decoder input [1, 211, 768].
- Update: projector weights ← weights − learning rate × gradient.
- Addendum classroom reader: logits $= \mathrm{normalize}(\mathrm{reader}(X))\,E^\top/\tau$, $E$: 8 × 768, τ = 0.1, cross-entropy.

**Worked numbers worth reusing**
- Token counts (`cost-full`, `cost-short`): 196 + 13 = 209 versus 8 + 13 = 21; 188 fewer decoder positions.
- Summarizer parameters and shapes (`resampler-init`, `resampler-code`): S has 8 × 192 = 1,536 numbers; Z: B × 8 × 192; attention weights: B × 3 × 8 × 196; projector output B × 8 × 768.
- Training probability (`training-align`): p[317] = 0.20 → −ln 0.20 = 1.609 (1.609437912); if 0.80, 0.223.
- Answer loss (`teacher-loss`): p = .20, .40, .50, .80, .90 for " A", " black", " dog", ".", EOS → penalties 1.609, 0.916, 0.693, 0.223, 0.105; total 3.547; **mean 0.709** (0.7094759783680473).
- Labels (`loss-mask`): 206 context positions = −100; positions 206 to 210 = 317, 2042, 3290, 13, 50256; logits 205 to 209 graded. Multi-image (`multi-image-summarize`, `multi-image-read`): 2 × 196 = 392 patch vectors → 8 + 8 = 16 visual vectors; authored attention mass A 0.30, B 0.40, text 0.30 under one softmax.

**Code** (verbatim)

Summarizer (`resampler-code`; the two comment lines and the shape column omitted):
```python
S = nn.Parameter(0.02 * torch.randn(1, 8, 192))
reader = nn.MultiheadAttention(192, 3, batch_first=True)
projector = nn.Linear(192, 768)
slots = S.expand(H.shape[0], -1, -1)
Z, weights = reader(slots, H, H,
                    average_attn_weights=False)
visual = projector(Z)
```
Notebook Idea 3 (return dictionary abbreviated to `{...}`):
```python
def compressed_prefix(ids):
    q = slots.expand(H_v.shape[0], -1, -1)
    summary, _ = reader(q, H_v, H_v)
    visual = projector(summary)
    text = lm.get_input_embeddings()(ids)
    x = torch.cat([visual, text], dim=1)
    h = lm.transformer(inputs_embeds=x).last_hidden_state
    return lm.lm_head(h[:, -1]), {...}
```
Training step (`instruction-example`, comments as on the slide):
```python
# 1. Ignore context targets (-100); score the reference answer.
labels = torch.full(X.shape[:2], -100, device=X.device)
labels[:, -answer_ids.shape[1]:] = answer_ids
# 2. Predict next tokens and average the five answer losses.
loss = lm(inputs_embeds=X, labels=labels).loss
# 3. Differentiate through the frozen LM; change the projector.
optimizer.zero_grad() # clear previous gradients
loss.backward() # compute projector gradients
optimizer.step() # update projector weights
```
Multi-image packing (`multi-image-code`, pseudocode): `parts = [before, A, after, B, text]`, `inputs = concat(parts, axis="tokens")` giving (16 + T) × 768.

**Figures and widgets worth reusing**
1. `summary-connections`: "Choose how image features reach the decoder" (prepend / read memory / summarize). Live: https://nipunbatra.github.io/attention/from-clip-to-vlm/#summary-connections
2. `architecture-names`: LLaVA versus Flamingo versus BLIP-2 in three columns. Live: https://nipunbatra.github.io/attention/from-clip-to-vlm/#architecture-names
3. `summary`: "From this image to the next word", 196 + 13 = 209 → 50,257 scores. Live: https://nipunbatra.github.io/attention/from-clip-to-vlm/#summary
4. `cost-short`: 209 versus 21 length bars on one scale; `resampler-toy-read`: two readers' weights painted on the dog.
5. `summary-training` and `training-instruct` / `training-instruction-tuning`: LLaVA two-stage training with FIXED/UPDATE badges. Paper figures `from-clip-to-vlm/figures/papers/flamingo-figure-3.png`, `blip2-figure-3.png` (PNG, backup D).
6. `synthesis` ("ViT represents. CLIP compares. A VLM generates."). Live: https://nipunbatra.github.io/attention/from-clip-to-vlm/#synthesis. Addendum figures `clip-addendum/figures/digup-demo.jpg`, `speech-demo.jpg` (JPG stills of the author videos).

**The lecture's own summary lines**
- "ViT represents. CLIP compares. A VLM generates." / "A generative VLM adds a language decoder and a path from the image to that decoder."
- "Projection changes vector width. Summarization changes row count. Access chooses the path." / "Project width. Choose the access path. Optionally shorten the visual sequence."
- "The image supplies context; the language vocabulary supplies the next token."
- "The reference supplies the targets; only the projector weights change." / "Attention controls what a state reads. Loss labels select what we grade."
- Multi-image recap: "Prepare each image, preserve its order, then answer using both."
- Addendum close: "Shared representations enable new applications. The task still determines the readout and the evaluation."

**Connections**
- Builds on L18 (Ideas 1 and 2, the toy K/V reused for the readers), L17 (CLIP vectors; the addendum reuses the unit-vector dot product for search), next-token cross-entropy and teacher forcing from the language-model lectures, and KV caching.
- The learned-query reader (fixed set of learned queries cross-attending to image features) is the same pattern as DETR's object queries; I did not open the detection deck, so cite it only as a mechanism link. The backup evaluation slides (false premise, image counterfactual with SmolVLM-256M: dog "There is a black dog in the foreground.", cat "A cat is visible.", blank "A squirrel.") connect to later evaluation discussion.

**Three exam-style questions with answers**
1. *What is the difference between the slots S and the summaries Z?* S (8 × 192) are trainable parameters shared across all images, initialized once; Z (8 × 192) are activations recomputed for each image by attending from S to that image's 196 rows. Z has no vocabulary ID or predefined object label.
2. *During teacher-forced training the whole reference "A black dog." is in the input. Why can the model not cheat?* The causal mask stops the state at position 205 (the last ":") from reading positions 206 onward, so it predicts " A" from image and prompt only; label masking (−100) only decides which outputs are graded, it does not hide context.
3. *Numeric.* Image 224 × 224, patch 16, CLS dropped, prompt 13 tokens. Prefix length with all patches, and with M = 8 summaries? 14 × 14 = 196 → 196 + 13 = 209; 8 + 13 = 21, saving 188 positions per layer. If the five answer-token probabilities are .20, .40, .50, .80, .90, the mean answer loss is (1.609 + 0.916 + 0.693 + 0.223 + 0.105) / 5 = 3.547 / 5 = 0.709.

---

## Note on Lecture 22: Learning ViTs with AI Tools

Schedule: "Socratic live coding of a Vision Transformer in PyTorch: using AI tools to understand patches, attention, tensor shapes and MNIST training." Recording only: https://www.youtube.com/watch?v=hHZIIKiOOr8. No slides. It re-implements L16's pipeline (patchify, shared projection, CLS and positions, encoder blocks, CLS head) on MNIST in PyTorch. I did not watch it, so I have no numbers (patch size, width, accuracy) to report.

---

## Module in one paragraph

The vision module takes the Transformer encoder students already know and changes only its input and its readout. L16 cuts a 224 × 224 photograph (the Newfoundland, `newfoundland_31`) into 196 patches of 16 × 16 × 3 = 768 numbers, projects each to 192 features with one shared matrix, adds learned positions and a CLS row (197 × 192), runs 12 full-attention blocks and reads CLS into a fixed 1,000-class head (95.73% Newfoundland). L17 replaces that fixed head with vectors computed from words: a ViT read at CLS and a causal text Transformer read at EOT are each projected and normalized to 512-D unit vectors, cosines form C = UVᵀ, S = C/τ, and the symmetric loss averages row-softmax (image → text) and column-softmax (text → image) cross-entropy (0.560526 on the cat/dog/car batch at τ = 0.5, versus ln 3 = 1.0986 for a tie). L18 and L19 then stop comparing and start generating: the ViT's patch rows reach a causal language decoder either as a projected prefix (LLaVA, Z = H_v W_p, 196 + 13 = 209 positions), as a separate memory read by cross-attention (Q from text, K and V from the image), or after a learned-query summarizer shrinks 196 rows to 8 (Flamingo, BLIP-2); the connection is trained with the ordinary next-token loss on reference answers, with frozen encoder and LM passing gradients to the projector.

## The three things a student must remember

1. **An image becomes tokens by patching and a shared linear projection**; after that the Transformer is unchanged (196 patches + CLS = 197 × 192 for ViT-Tiny at patch 16; attention cost grows as tokens²).
2. **CLIP's loss is cross-entropy over a batch similarity matrix, done twice**: S = UVᵀ/τ, softmax across each row (pick the caption) and down each column (pick the image), take −ln of the diagonal, average both directions. After training, any text becomes a class vector (zero-shot).
3. **A VLM is a language decoder plus a path for image features**: prefix (projector changes width, keeps count), cross-attention (text queries, image keys and values), or learned summaries (changes count); the vocabulary head, not the visual lookup, picks the next word, and training uses the same next-token loss on supplied answers.
