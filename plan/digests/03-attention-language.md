# Digest 03: Attention and Language (Lectures 11 to 15)

Sources read (published branches only):
- attention repo `origin/main` at `cde4dea` (2026-11-03): `part1.html`, `attention.html`, `part3.html`. The live pages are byte-identical to `origin/main` (checked for `attention.html`).
- dl-teaching repo `origin/master` at `118c7bce` (2026-11-09): `cheatsheets/P01-next-token-prediction.typ`, `cheatsheets/P02-self-attention.typ`, `interactives/from-attention-to-applications/{index.html, TEACHING_GUIDE.md, QUESTIONS.md, SOURCES.md}`.
- Numbers marked "computed" were recomputed by me from the model data embedded in the pages (`window.__TOY__` in `part1.html` and `attention.html`). They match every value the slides and cheat sheets print, so they are the numbers the widgets show.

Live bases: `https://nipunbatra.github.io/attention/` and `https://nipunbatra.github.io/dl-teaching/interactives/from-attention-to-applications/`. Append `?present` before the hash for presentation mode.

## How L12, L13 and L14 split inside `attention.html`

The schedule points L12, L13 and L14 at the same `attention.html` (19 sections, `s01` to `s19`) and adds `part3.html` for L14. The schedule gives no section boundaries, so I used three pieces of evidence:
1. **L12 full recording** (YouTube `fRSLf13z6wo`, 1:11:18). Its chapters run from "Why limited context fails" through hard and soft retrieval, "A worked QKV example", "Attention as a contextual update", "'Bank': same token, different context", "Contextual embeddings to predictions", and end at 1:11:08 with "Where queries, keys and values come from". So L12 covered `s01` to `s10`, with a short preview of `s14`, and stopped at `s11`.
2. **L13 has no full recording.** Its schedule row links two short videos: "From attention to prediction" (causal mask, the final "the", the matrix form, generation and training) and "Position and order". The schedule subtitle is "Embedding updates, next-token prediction, and positional encoding".
3. **The deck's own route** in `s01` says the first pass is `s01` to `s09`, `s13`, `s14`; the next sitting covers `s10` to `s12`, then `s15` to `s16`, then position (`s17`).

My split:
- **L12** = `s01` to `s10`. The `s09` update (`Δe`) and the `s10` bank comparison were taught here per the recording, though L13's subtitle repeats "embedding updates".
- **L13** = `s11` to `s14` plus `s17` (Q/K/V projections, the `√d_k` scaling, the causal mask, prediction from the final "the", and positional encoding).
- **L14** = `s15`, `s16`, `s18`, `s19` (the numerical walkthrough, the full matrix form, the checks, generation and training), then all of `part3.html` (multi-head attention and TinyStories).

The L13/L14 boundary is a judgment call. The L13 short video also covers the matrix form and generation/training (`s16`, `s19`), so those may have been taught in L13 instead.

---

## Lecture 11: From characters to next-token prediction

**Aim**: Build a complete character-level language model (embedding lookup, concatenation, a one-hidden-layer MLP, softmax, cross-entropy, sampling), then show that its fixed window motivates attention.

**Running example(s)**
- Data: Indian given names from `balasahebgulave/Dataset-indian-names` (`Indian_Names.csv`), lowercased, stripped to a to z, deduplicated, and split by name. That gives 5,830 train names and 648 held-out names, and 42,903 / 4,749 windows (computed from the embedded metrics).
- Vocabulary: `$\mathcal V=\{\text{-},a,\ldots,z\}$`, 27 tokens. The boundary `-` has ID 0. It pads the opening window and is the target that ends a name. `stoi`: `"-"→0, "a"→1, "b"→2`.
- The worked name is **aabid**, with context length `w = 3`. It gives six context-target pairs: `--- → a`, `--a → a`, `-aa → b`, **`aab → i`**, `abi → d`, `bid → -`. A name of length `L` gives `L+1` pairs.
- Sizes: embedding width `d = 2` (`nn.Embedding(27, 2)`), concatenated input of 6 numbers, 32 hidden ReLU units, 27 logits. The trained model's embedding axis 1 was constrained to be positive for vowels and negative for consonants (the slide says this is a teaching constraint, not a discovery).
- Trained rows (2 d.p.): `e_a ≈ [0.63, 0.09]`, `e_b ≈ [-1.25, -0.15]`, `e_i ≈ [0.76, -0.45]`.
- Side examples: the next-word distribution after "The student opened the laptop and"; Word2vec "king − man + woman ≈ queen" with toy 2D vectors `man [1,1], woman [3,1], king [1,3], queen [3,3]`; one stored "bank" row for both "deposited money at the bank" and "river bank". The tokenizer demo uses "deep learning is amazing"; the word model uses "The cat sat on → the".
- Generation: start from `- - -`. The saved samples (seeded, temperature 0.75) include jajin, sukshana, mushan, shila, ramina, guriya, vinder.

**Ideas in teaching order**
- Language tasks differ in their output (one label, a label per word, a sequence). Produce text by predicting one next token at a time.
- A token is a unit of text. Start with characters so that every one of the 27 probabilities is visible.
- Choose a fixed context length: `w = 3` previous characters, with boundary padding.
- One-hot rows give zero dot product between any two characters. A learned row is selected instead: `$e_c=E_{\text{tok}}[\operatorname{id}(c)]$`. "This is selection, not multiplication."
- Detour on embeddings: Word2vec offsets, document vectors (mean pooling loses order, so "dog bites man" = "man bites dog"), image and time-series encoders. The slide says equal widths do not imply shared meaning; CLIP comes later.
- Concatenate, do not add. Concatenation keeps a slot per position, while addition is commutative and loses order.
- MLP: `a_1 = ReLU(a_0 W_1 + b_1)` and `z = a_1 W_2 + b_2`. Row vectors multiply matrices on the right. PyTorch stores `W` transposed.
- Softmax (with the max-subtraction trick), the loss `-log p(target)`, then SGD on all five parameter tensors. Only the looked-up rows `a` and `b` receive embedding gradients.
- Generation: sample (or greedy), append, shift the window. Temperature `softmax(z/T)`. Training and generation use the same forward pass.
- Tokens can be characters, words or subwords (`<UNK>` for unseen words). The chain rule turns a sequence probability into a product of next-token probabilities. A longer window widens `W_1` (`wd` inputs). That is the question attention answers.

**Key equations** (lecture notation)
- Next-character probability: `$p_{\theta}(t_i\mid t_{i-3},t_{i-2},t_{i-1})$`, where `θ` is the embedding table, weights and biases.
- Lookup: `$e_c=E_{\text{tok}}[\operatorname{id}(c)]$`, with `E_tok` of shape 27 × 2. Concatenation: `$a_0=[e_1,\ldots,e_w]\in\mathbb R^{1\times6}$`.
- Hidden layer: `$a_1=\operatorname{ReLU}(a_0W_1+b_1)$`, with `W_1` 6 × 32 and `b_1` 1 × 32.
- Logits and probabilities: `$z=a_1W_2+b_2$` with `W_2` 32 × 27, and `$p_j=\exp(z_j)/\sum_{r\in\mathcal V}\exp(z_r)$`.
- Loss: `$\operatorname{loss}=-\log p(\text{target})$`. Temperature: `$p=\operatorname{softmax}(z/T)$`. (The cheat sheet writes `τ` for the temperature and `ℓ = -ln p_y`.)
- Width cost: `$(wd)\times d_h=(100\times256)\times1{,}024=26{,}214{,}400$` weights in `W_1` alone.

**Worked numbers worth reusing**
1. `aab → i` with the trained model: `a_0 = [0.63, 0.09, 0.63, 0.09, -1.25, -0.15]`. Then `p(i | aab) = 0.117`, so the loss is `-ln 0.117 = 2.15` (slides `s07`, `s08`). Surprise: the top prediction after `aab` is the boundary `-` at 0.464, with i second (computed). The untrained "before" checkpoint gives `p(i) ≈ 0.052`.
2. All six aabid losses (`s08` "Inspect all six rows"): `---→a 0.078 / 2.55`, `--a→a 0.085 / 2.47`, `-aa→b 0.023 / 3.77`, `aab→i 0.117 / 2.15`, `abi→d 0.095 / 2.36`, `bid→- 0.235 / 1.45`.
3. Confident correct versus wrong (`s08`): `p = 0.80 → loss 0.223` and `p = 0.01 → loss 4.605`.
4. Temperature on the input `− s a` (`s10`): the top token is n. `p(n)` is 0.342 at T = 0.5, 0.161 at T = 1 and 0.112 at T = 1.5 (computed; the slide plots T = 0.5, 1, 1.5).
5. Parameter count (cheat sheet): `E_tok 54 + W_1 192 + b_1 32 + W_2 864 + b_2 27 = 1,169`. Training loss falls from 3.30 to 2.20 (held-out 2.20) over 6,000 steps. With `w = 5, d = 4` the count is 1,671 (`s14` code).

**Code** (verbatim)
```python
# s16 "PyTorch: the complete forward pass in five lines"
e = embedding(ctx)
a0 = e.flatten(1)
a1 = torch.relu(hidden(a0))
z = output(a1)
p = z.softmax(dim=-1)
```
```python
# s09 "one update on the six training rows"
optimizer.zero_grad()
z = model(X)
loss = F.cross_entropy(z, y)
loss.backward()
optimizer.step()
```
```python
# s10 "append, shift, and predict again"
for _ in range(18):
    next_id = sample_next(ctx, temperature)
    if next_id.item() == stoi["-"]: break
    name.append(vocab[next_id.item()])
    ctx = torch.cat([ctx[:, 1:], next_id], dim=1)
```

**Figures and widgets worth reusing**
- `figures/indian-names/github-dataset.png`: the dataset screenshot (aabid, aabida, aachal, aadesh). It is embedded in the page as a data URI.
- `#s03`, sliding the window across aabid: a stepper and a six-row table.
- `#s04`, the embedding geometry: one-hot zero dot products, the 2D points for a, b, i, and the king/queen offset parallelogram.
- `#s05`, concatenation keeps order while the sums are equal: the `abi` versus `iba` comparison.
- `#s06`, the network drawing 6 → 32 → 27, with worksheets for one hidden unit and the logit for i.
- `#s10`, the live generator (seed, greedy or sample, temperature) and the step-through of one generated name.
- Deep links (frames in this page mostly lack ids, so these are section anchors):
  - https://nipunbatra.github.io/attention/part1.html#s04
  - https://nipunbatra.github.io/attention/part1.html#s06
  - https://nipunbatra.github.io/attention/part1.html#s10
  - https://nipunbatra.github.io/attention/part1.html#s14
  - https://nipunbatra.github.io/attention/part1.html#s16

**The lecture's own summary lines**
- "Use the last few characters to guess what character tends to come next."
- "Look up embeddings, concatenate the window, run the MLP, apply softmax, then sample or score the observed target."
- "Attention will replace the fixed concatenation. We will keep using embeddings, logits, softmax, loss, and generation."
- Cheat sheet: "IDs select rows. Embeddings are learned parameters. Hidden activations depend on the current input."

**Connections**
- Builds on: cross-entropy as negative log-likelihood (L1 to L3), the MLP and ReLU (L4), SGD and batches (L5), backprop/autograd and the transposed `nn.Linear` weight (L6, L8), and the held-out split (L10).
- Reused by: the whole attention series. `E_tok`, softmax, cross-entropy and generation carry into L12 to L15. The `s04` "image-text retrieval needs training that aligns the two spaces" points to CLIP (L17).

**Exam-style questions**
1. *Why concatenate the three embeddings rather than add them?* Addition is commutative: `aab` and `aba` would give the same input row, and order would be lost. Concatenation gives each position its own slot and its own weights in `W_1`.
2. *During training on `aab → i`, does the embedding row for `i` get a gradient?* No. Only the looked-up input rows (a, used twice, and b) receive embedding gradients. The target `i` only selects which log-probability enters the loss. It does reach the output weights `W_2`, `b_2` through the softmax.
3. *Numeric.* With `w = 3`, `d = 2`, 32 hidden units and `|V| = 27`, count the parameters, and give the loss if `p(i) = 0.117`. Answer: `27·2 + 6·32 + 32 + 32·27 + 27 = 54 + 192 + 32 + 864 + 27 = 1,169`. The loss is `-ln 0.117 ≈ 2.15` nats.

---

## Lecture 12: Self-attention, from first principles

**Aim**: Derive attention as a learned, content-dependent weighted average: queries and keys choose the weights, and the weights mix values, so that a token like "bank" gathers context from earlier tokens.

**Running example(s)**
- Two ten-token sentences with the same ending:
  - A: "The fisherman sat beside the river bank and watched the ___" (plausible next word: water).
  - B: "She deposited the cheque at the bank and watched the ___" (plausible next word: teller).
- The model is hand-chosen, not trained. `d_model = 4` with named axes **water, finance, person, glue**; `d_k = 3` (q/k axes "setting: water?", "setting: finance?", "who?"); `d_v = 2` (v axes "says: water scene", "says: finance scene"). The vocabulary has 20 words, `T = 10` tokens, and a 4 → 8 → 20 prediction head.
- Starting rows add small hand-chosen position offsets: `e_i^{(0)} = E_tok[t_i] + p_i`. Examples: `E_tok[river] = [3.0, 0, 0, 0]`, `E_tok[bank] = [0.7, 0.7, 0, 0.7]`, `E_tok[the] = [0, 0, 0, 2.4]`, `e_6 = [3.1, -0.1, 0.0, 0.1]` (river at position 6), `e_7 = [0.7, 0.7, 0.1, 0.8]`.
- The receiver is **bank at position 7**, reading tokens 1 to 7, with `q_7 = [1.26, 1.26, 0.04]`.
- The search analogy uses six videos (Backpropagation, Gradient descent, CNNs, Transformers, Batch normalisation, Regularisation). Each has a 4-number key (gradient flow, optimisation, architecture, generalisation) and an 8-number value. Query 1 ("How do gradients flow?") is `q = [2.0, 0.6, 0.2, 0.0]`. Two other examples: a contact directory (Library, extension 204, floor 2, 09:00 to 20:00), and the Maya/coat pronoun story ("She …" versus "It …").

**Ideas in teaching order**
- Recap Part 1, then the 2017 paper (Vaswani et al., original Figure 1). A token is a lookup plus a position row: `e_i^{(0)} = E_tok[t_i] + p_i`.
- Baseline 1: only the last token. Both sentences end with "the" at position 10, so the head sees the same row, and every candidate ties at about 0.094.
- Baseline 2: a fixed window of `w` tokens (concatenate into `c_t`). The parameters grow with `w`, and the clue can still fall outside: river enters only at `w = 15`, cheque at `w = 16`.
- Summarize the prefix into a fixed width `m_i`: first the plain mean, then hand-chosen weights `α_{ij}` (try `α_{7,6} = 0.5` for river). Ask where the weights should come from.
- The search analogy: the query says what I want, the key says when to retrieve me, the value says what I send. The score is `s_j = q·k_j`.
- Hard retrieval (`argmax`, return one value) gives way to soft retrieval (softmax weights, mix all values). Temperature `τ` interpolates between them.
- Every token becomes a searchable record. Two jobs: Phase A, "Q and K choose weights"; Phase B, "the same weights mix V" into `m_7`.
- The message `m_7` (2 numbers) is mapped by `W_O` to `Δe_7` (4 numbers) and added by the residual: `e_7' = e_7 + Δe_7`.
- "Bank": the same start, different context. River dominates in A, while cheque and deposited dominate in B. The same row ends up in different places.

**Key equations**
- Weighted pooling: `$m_i=\sum_{j=1}^{i}\alpha_{ij}\,e_j$`, with `$\alpha_{ij}\ge0$` and `$\sum_{j}\alpha_{ij}=1$`. The mean is the case `$\alpha_{ij}=1/i$`.
- Hard retrieval: `$j^*=\arg\max_j q\cdot k_j$`, return `$v_{j^*}$`. Soft retrieval: `$\alpha_j=e^{s_j}/\sum_r e^{s_r}$`, and with temperature `$\alpha=\operatorname{softmax}(s/\tau)$`.
- The score in the sentence: `$s_{7j}=\dfrac{q_7\cdot k_j}{\sqrt{d_k}}$`, then `$\alpha_{7j}=\dfrac{e^{s_{7j}}}{\sum_{r\le7}e^{s_{7r}}}$`.
- The message: `$m_7=\sum_{j\le7}\alpha_{7j}v_j$`, with shape 1 × `d_v` = 1 × 2.
- The update: `$\Delta e_7=m_7W_O$` (`W_O` is 2 × 4), and `$e_7^{\text{new}}=e_7+\Delta e_7$`. The deck notes that `Δe` is "a teaching label", not standard notation.

**Worked numbers worth reusing**
1. Video search, query 1 (`s06-frame-scores`): `s_1 = 2.0×2.2 + 0.6×0.6 + 0.2×0.2 + 0.0×0.0 = 4.8`. The six scores are `[4.8, 1.4, 0.6, 0.2, 0.9, 0.5]`, and the softmax gives Backpropagation 0.916 (computed, no `√d` scaling in this section).
2. Bank in sentence A (`s08`, also in the P02 cheat sheet). The weights over The, fisherman, sat, beside, the, river, bank are **`[0.050, 0.226, 0.059, 0.067, 0.054, 0.414, 0.130]`**, with river highest. River's contribution is `0.414 × [3.1, -0.1] = [1.283, -0.041]`, and the message is `m_7 ≈ [1.89, 0.07]`.
3. The update (`s09`): `W_O = [[0.8, -0.2, 0.3, 0.1], [-0.1, 1.1, 0.2, -0.2]]`, so `Δe_7 ≈ [1.51, -0.30, 0.58, 0.18]` and `e_7^new ≈ [2.21, 0.40, 0.68, 0.98]`.
4. Bank in sentence B (`s10`, computed): cheque gets 0.400 and deposited 0.261. Then `m_7 ≈ [0.11, 1.86]`, `Δe_7 ≈ [-0.10, 2.02, 0.40, -0.36]` and `e_7' ≈ [0.60, 2.72, 0.50, 0.44]`. The same start moves toward water in A and toward finance in B.
5. The last-token baseline (`s02`): all eight candidates (water, boats, fish, ducks, teller, clerk, queue, money) get about 0.094 each in both sentences, because the hidden layer is all zeros for `e_10^{(0)}`. A window parameter count (`s03`): `w = 10`, `d = 4`, `V = 20` gives `40 × 20 = 800` weights plus 20 biases, so 820.

**Code** (verbatim, `s08` and `s09`)
```python
scores = (q_bank @ K.T) / math.sqrt(K.shape[-1])
alpha = scores.softmax(dim=-1)
m_bank = alpha @ V  # [7] @ [7, d_v] → [d_v]
```
```python
delta_e_i = m_i @ W_O
e_new_i = e_i + delta_e_i
```

**Figures and widgets worth reusing**
- `figures/transformer-paper/architecture.png`: the original Transformer Figure 1 (`#s01-paper-architecture`). The first page of the paper is `figures/transformer-paper/first-page.png`.
- `#s04-frame-choose`: sliders for hand-chosen weights on bank (equal, river, fisherman, only bank) feeding one fixed head.
- `#s06-frame-temperature`: a hard/soft toggle and a `τ` slider on the video search.
- `#s08-frame-phases`: an inline SVG of Phase A (q·k, scale, softmax) and Phase B (α·v, add to `m_7`). This is the best single picture of attention in the deck.
- `#s09-frame-motif`: the residual branch preserves `e_7`, and the attention branch computes `Δe_7`.
- `#s10-frame2`: two arrows leaving the same blue dot for bank and ending far apart, with a slider.
- Deep links:
  - https://nipunbatra.github.io/attention/attention.html#s02-frame-probabilities
  - https://nipunbatra.github.io/attention/attention.html#s04-frame-weight-sum
  - https://nipunbatra.github.io/attention/attention.html#s06-frame-scores
  - https://nipunbatra.github.io/attention/attention.html#s08-frame-phases
  - https://nipunbatra.github.io/attention/attention.html#s10-frame-sources

**The lecture's own summary lines**
- "Only the query and key determine the match score. Once we choose an item, we retrieve its value."
- "The scores determine the weights, which determine the mixture."
- "First route with Q and K. Then retrieve with V."
- Cheat sheet banner: "Q and K choose the weights. Those weights mix V. The output projection makes an update that can be added to E."

**Connections**
- Builds on: L11's lookup table `E_tok`, its MLP head (here 4 → 8 → 20, with `ℓ` "called z in Part I"), softmax, and the fixed-window limit.
- Reused by: L13 (where q, k, v come from), L15 (the same `m_i`, `Δe_i = m_i W_O`, `e'_i` notation, which the L15 teaching guide states explicitly), and L16 ViT ("embedding updates and CLS readout").

**Exam-style questions**
1. *Why keep separate keys and values instead of returning the matched key?* The key only decides whether a source matches. The value is what the source sends. The deck's experiment mutes the values' finance column while keeping Q and K fixed: the weights are unchanged, but the message changes.
2. *Why can the last-token baseline not separate the river and cheque sentences, however good its head is?* Both inputs are the row for "the" at position 10. The same row gives the same logits and the same probabilities, and softmax cannot add missing context.
3. *Numeric.* Take `q = [2.0, 0.6, 0.2, 0.0]` and two keys, `k_1 = [2.2, 0.6, 0.2, 0.0]` and `k_2 = [0.4, 1.0, 0.0, 0.2]`, with no scaling. Give the scores and the softmax weights. Answer: `s_1 = 4.4 + 0.36 + 0.04 = 4.8` and `s_2 = 0.8 + 0.6 = 1.4`. Then `α_1 = 1/(1 + e^{-3.4}) = 0.968` and `α_2 = 0.032`.

---

## Lecture 13: Self-attention, continued (projections, scaling, masking, prediction, position)

**Aim**: Explain where q, k and v come from, why scores are divided by `√d_k`, how the causal mask keeps training honest, how the updated final token predicts the next word, and how position enters attention.

**Running example(s)**
- The same hand-chosen bank model (`d_model = 4`, `d_k = 3`, `d_v = 2`, `T = 10`, vocabulary 20). Its projections are sparse: `W_Q = [[1,0,0],[0,1,0],[0,0,0.4],[0.7,0.7,0]]`, `W_K = [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]`, and `W_V = [[1,0],[0,1],[0,0],[0,0]]`. So the glue word "the" has an almost zero key and value, but asks for both settings.
- The receiver moves from bank (`q_7`) to **the final "the" at position 10** (`q_10 = [1.61, 1.61, 0.0]`), which predicts word 11.
- Masking example: row 5 ("the") in sentence A, whose target is position 6, river.
- Scaling example: `d_k = 64`, with raw scores −8 and +8. Simulations use ±1 entries.
- Position examples: "Maya helps Ravi today" versus "Ravi helps Maya today", with 2D rows `Maya [1,0]`, `Ravi [0,1]`, `helps [0.2,0.2]`, `today [0.8,0.2]` and `W_Q = W_K = W_V = I`. Relative position uses "Maya carries red flowers".

**Ideas in teaching order**
- Q, K and V are projections of the current row, `e_i W_Q`, `e_i W_K`, `e_i W_V`, shared by every token. They are working vectors, not lookup tables. Q and K need equal width `d_k`, while `d_v` can differ.
- Raw dot-product spread grows like `√d_k`. Dividing keeps the score SD near 1 and avoids softmax saturation. This is not the same problem as overflow, which the max-subtraction trick handles.
- Causal mask: position `i` may read only `1…i`. Add `-∞` above the diagonal before softmax. Without it, row 5 can read its own answer.
- Prediction: the final "the" supplies `q_10`, mixes ten values, and gives `Δe_10` and `e'_10`, which the 4 → 8 → 20 head reads. "Same prediction head, different context."
- The attention softmax (over source positions) is not the vocabulary softmax (over next tokens).
- Alternatives compared (fixed window, mean, fixed positional weights, attention): only attention computes its weights from content.
- Without position, attention is permutation-blind: swapping Maya and Ravi leaves today's message unchanged. Fixes: add a position row `p_i` (learned table, or sinusoidal); relative schemes (ALiBi biases the scores, RoPE rotates Q and K); appending the raw index (which needs scaling).

**Key equations**
- Projections: `$q_i=e_iW_Q,\;k_i=e_iW_K,\;v_i=e_iW_V$`, with `W_Q, W_K` of shape `d_model × d_k` and `W_V` of shape `d_model × d_v`.
- Scaling: `$\operatorname{Var}(q\cdot k)=d_k$` and `$\operatorname{SD}(q\cdot k)=\sqrt{d_k}$`, so `$s_{ij}=q_i\cdot k_j/\sqrt{d_k}$`.
- Mask: `$S=QK^\top/\sqrt{d_k}$`, `$M_{ij}=-\infty$ for $j>i$` (0 otherwise), and `$A=\operatorname{softmax}_{\text{row}}(S+M)$`. The 4 × 4 `M` is drawn on the slide.
- Prediction head: `$h_{10}=\operatorname{ReLU}(e_{10}'W_1+b_1)$`, `$\ell=h_{10}W_2+b_2$`, `$p(x_{11}\mid x_{\le10})=\operatorname{softmax}(\ell)$`.
- Sinusoidal position: `$\omega_r=b^{-2r/d}$`, `$p_{i,2r}=\sin(i\omega_r)$`, `$p_{i,2r+1}=\cos(i\omega_r)$`, with `b = 10000`.
- ALiBi: `$s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_k}}-m(i-j)$`. RoPE: `$(R_iq)^\top(R_jk)=q^\top R_{j-i}k$`.

**Worked numbers worth reusing**
1. Scaling (`s12-frame-softmax`): for raw scores [−8, 8], `p_A ≈ 0.000000113`. After dividing by `√64 = 8`, the scores are [−1, 1] and `p_A ≈ 0.1192`. In the bank model (`s12-frame-bank`, computed), river's weight is 0.619 unscaled and 0.414 with `/√3`. The ranking is unchanged.
2. The mask leak (`s13`, and the L13 short video): with the mask off, row 5 puts **0.67** of its weight on positions 6 to 10 (river alone gets 0.477). The cheat-sheet toy: `[1,2,3,4] + [0,0,−∞,−∞]` gives a softmax of `[0.269, 0.731, 0, 0]`.
3. Final "the" (`s14`, computed; the short video prints the top numbers). In sentence A, the weights are river 0.450, fisherman 0.195, bank 0.102, and the prediction is **water 0.367**, boats 0.201, fish 0.148, ducks 0.093. In sentence B, the weights are cheque 0.422, deposited 0.242, bank 0.105, and the prediction is **teller 0.521**, clerk 0.242, queue 0.087, money 0.058. With attention bypassed, all eight candidates get 0.094.
4. The position swap (`s17`): today's query `[0.8, 0.2]` scores Maya `0.8/√2 ≈ 0.566`. With token-only rows, the weights are Maya 0.310, helps 0.203, Ravi 0.203, today 0.285 (computed) in both orders, so the message is identical. With position rows, Maya's slot-1 score is 0.990 and its slot-3 score is 1.400.
5. Sinusoid at `d = 4`, `i = 3` (`s17-position-worked-sine`): `p_3 = [0.141, −0.990, 0.030, 1.000]`. Added to the word row `[0.8, 0.2, 0.4, −0.1]`, it gives `e_3 ≈ [0.941, −0.790, 0.430, 0.900]`. ALiBi with `m = 0.25`, receiver flowers and all content scores 2: the scores are `[1.25, 1.50, 1.75, 2.00]` and the weights `[16.5%, 21.2%, 27.3%, 35.0%]`.

**Code** (verbatim)
```python
# s13
scores = scores.masked_fill(future, float('-inf'))
weights = scores.softmax(dim=-1)
```
```python
# s14
q_last = E[-1] @ W_Q  # the current last known position
scores = (q_last @ K.T) / math.sqrt(K.shape[-1])
alpha = scores.softmax(dim=-1)  # ten source weights; sum = 1
delta_last = (alpha @ V) @ W_O
e_last_new = E[-1] + delta_last
h = (e_last_new @ W_hidden + b_hidden).relu()  # W1, b1: 4 → 8
logits = h @ W_vocab + b_vocab  # W2, b2: 8 → 20
p_next = logits.softmax(dim=-1)
```

**Figures and widgets worth reusing**
- `#s11-frame1` and the token picker: one row `e` becomes q, k, v under the shared matrices, with axis names.
- `#s12-frame-spread`: histograms of ±1 dot products at four widths (3,000 pairs each), raw versus scaled.
- `#s13-frame3`: the real 10 × 10 attention matrix for sentence A, with a mask toggle. With the mask off, weight flows into the hatched upper triangle.
- `#s14-probabilities`: a context toggle (river or cheque) that lights up the attention row, the update and the probability bars.
- `#s17-position-sine-rule` and `#s17-position-waves`: the sinusoid rule and the fast/slow waves. `#s17-position-method-comparison` compares additive position with RoPE.
- Deep links:
  - https://nipunbatra.github.io/attention/attention.html#s11-frame1
  - https://nipunbatra.github.io/attention/attention.html#s12-frame-softmax
  - https://nipunbatra.github.io/attention/attention.html#s13-frame2
  - https://nipunbatra.github.io/attention/attention.html#s14-probabilities
  - https://nipunbatra.github.io/attention/attention.html#s17-position-order

**The lecture's own summary lines**
- "Scaling preserves the score ranking. It controls how strongly softmax concentrates on the highest scores."
- "Token $i$ may retrieve from $j\le i$, never from $j>i$."
- "Same prediction head, different context."
- "Position describes order. The causal mask restricts visibility. Neither removes attention's pairwise compute cost."

**Connections**
- Builds on: L12's routing and mixing. Softmax saturation links to the gradient-flow ideas from backprop (L6). The variance argument uses independence, mean 0 and variance 1.
- Reused by: L15 (causal decoder mask, whole-input encoder mask, positions `e_i + p_i`), L16 ViT (learned position table for patches; the attention repo's ViT commits mention "learned position tables"), and L18/19 VLMs ("positions in the visual prefix").

**Exam-style questions**
1. *Why is the mask applied before softmax, not by zeroing the weights after it?* Adding `-∞` gives `e^{-∞} = 0`, and the remaining weights still sum to 1. Zeroing after softmax would leave rows that do not sum to 1, unless they are renormalized.
2. *Without position information, can one attention layer at "today" tell "Maya helps Ravi today" from "Ravi helps Maya today"?* No. Each token keeps its key and value wherever it sits, so the same weights follow the words, and the sum of the same weighted values is identical. Adding `p_i` changes the keys and values by slot.
3. *Numeric.* For `d_k = 64` and raw scores [−8, 8], give A's weight before and after scaling. Answer: before, `e^{-8}/(e^{-8} + e^{8}) = 1/(1 + e^{16}) ≈ 1.1 × 10^{-7}`. After dividing by 8, the scores are [−1, 1] and `1/(1 + e^{2}) ≈ 0.119`. B stays the winner, and A now contributes.

---

## Lecture 14: Self-attention (full calculation) and multi-head attention

**Aim**: Assemble single-head attention into the matrix form `E' = E + softmax(QK^T/√d_k + M)VW_O`, then split it into parallel heads and train real one-head and four-head models on TinyStories.

**Running example(s)**
- `attention.html` `s15` and `s16`: the ten-token river sentence, an 18-step numerical walkthrough, and then the matrix form on the **seven-token prefix**, where row 7 is bank throughout.
- `part3.html` `s01` and `s02`: the same river prefix, with the final "the" as receiver. Head 1 might favour river ("setting clues?") and head 2 fisherman ("person clues?"). The coat story ("Maya wore a hooded red wool coat") is drawn with three illustrative heads for colour, material and detail. The two-head worksheet uses `T = 10`, width 4, and two heads of width 2.
- TinyStories (`s03` to `s06`): 6,000 documents split by story into 4,822 train, 586 validation and 592 test stories. The teaching sentence is **"Lily found a red ball."**, tokenized as `lily found a red ball .`. With 4 special IDs (`PAD 0, BOS 1, EOS 2, UNK 3`) the vocabulary has `C = 10`. The window is `w = 4`, giving 7 training pairs: the first is `[0,0,0,1] → 8` (lily), the fourth is `[1,8,7,5] → 9` (red).
- The trained model: `C = 4,000`, `T = 64`, `d_model = 64`, 4 heads of width 16, a 256-unit prediction MLP, batch 512, 6,000 updates, and seeds 11, 29, 47. Positions are a learned 64 × 64 table, not sinusoidal.

**Ideas in teaching order**
- (`s15`, `s16`) Every token runs the same steps, so stack the rows. `Q = EW_Q`, `S = QK^T/√d_k`, `A = softmax_row(S + M)`, `H = AV`, `ΔE = HW_O`, `E' = E + ΔE`.
- (`s18`, `s19`) Notation by space: `q, k` live in `ℝ^{d_k}`, `v, m` in `ℝ^{d_v}`, and only `e, Δe, e'` live in `ℝ^{d_model}`. Generation is predict, choose, append, repeat. Training is `L = −log p(y)` plus backprop.
- Why more than one head: one head uses one set of source weights for every coordinate of its message. Two heads can weight sources independently.
- Each head has its own `W_Q, W_K, W_V` on the same input rows. Concatenate the messages, project with `W_O`, and add to `E`.
- The data pipeline: whole stories, split before windowing, then tokenize, vocabulary, BOS/EOS once per story, left-padded windows, and `X [B,4]`, `y [B]`.
- The model: token lookup plus a learned position row, Q/K/V reshaped into 4 heads of 16, scores `/√16`, a causal-plus-PAD mask, `A @ V`, join, `W_O`, residual, then the final row into the MLP and the vocabulary.
- Generation: BOS plus the prompt, left-padded to 64, greedy with PAD, BOS and UNK excluded, stopping at EOS.
- Results: four heads beat one head, and both beat the fixed-window MLP, at equal attention parameter counts.

**Key equations**
- `$E=\begin{bmatrix}e_1\\\vdots\\e_T\end{bmatrix},\quad Q=EW_Q,\;K=EW_K,\;V=EW_V$`, with `E` of shape `T × d_model`.
- `$S=\dfrac{QK^\top}{\sqrt{d_k}},\quad A=\operatorname{softmax}_{\text{row}}(S+M),\quad H=AV$`. `S`, `M` and `A` are `T × T`. `H` is `T × d_v`.
- `$\Delta E=HW_O,\qquad E'=E+\Delta E$`.
- Multi-head (part3 `s02-cap-map`): head `h` gives `$H^{(h)}=A^{(h)}V^{(h)}$` of shape [10×2]. Concatenating two [10×2] gives [10×4], and `[10×4] × W_O [4×4]` is `ΔE`.
- Perplexity: `$\exp(\text{test loss})$`. "If every true next token gets probability ¼, perplexity is 4."

**Worked numbers worth reusing**
1. Why two heads (`s01-v-independent`, optional box): river's value is [10, 1] and fisherman's is [2, 8]. One head with weights [0.8, 0.2] returns `[8.4, 2.4]`. Two heads: the setting head with [0.8, 0.2] takes coordinate 1 and gets 8.4. The person head with [0.2, 0.8] takes coordinate 2 and gets `0.2·1 + 0.8·8 = 6.6`.
2. Lily windows (`s03-data-pairs-tensors`): `all_X` is 7 × 4. The rows are `[0,0,0,1]→8, [0,0,1,8]→7, [0,1,8,7]→5, [1,8,7,5]→9, [8,7,5,9]→6, [7,5,9,6]→4, [5,9,6,4]→2`. That is 6 ordinary tokens plus EOS, so 7 targets. The training split has `964,338 + 4,822 = 969,160` targets.
3. Shapes (`s04`): `X [2,64]`, giving token rows `[2,64,64]`. Each of Q, K, V is `[2,4,64,16]`, the scores are `[2,4,64,64]`, the messages `[2,4,64,16]`, and `joined [2,64,64]`.
4. Test results (`s05-cap-scores`, mean ± SD over 3 seeds):

   | Model | Test loss | Perplexity | Parameters | Time per seed |
   |---|---|---|---|---|
   | Fixed-window MLP | 3.943 ± 0.008 | 51.59 ± 0.40 | 2,332,832 | 58.2 s |
   | One-head attention | 3.445 ± 0.011 | 31.34 ± 0.33 | 1,321,120 | 65.6 s |
   | Four-head attention | 3.360 ± 0.021 | 28.78 ± 0.61 | 1,321,120 | 69.7 s |

   Four heads cut mean perplexity by 8.2% versus one head.
5. Owl prompt (`s06-cap-samples`, greedy, 24 tokens). The prompt is "once upon a time, there was a wise old owl who lived in a big tree. the owl". The four-head continuation is "lived in a big tree and the owl. the owl was very curious and he loved to play with his friends. one". The one-head continuation is "was very happy and had a lot of fun. the little girl was so happy and she was so happy to have a".

**Code** (verbatim, consecutive frames `s04-cap-embed` to `s04-cap-join` of part3)
```python
Q = model.W_Q(E).reshape(B, T, 4, 16).transpose(1, 2)
K = model.W_K(E).reshape(B, T, 4, 16).transpose(1, 2)
V = model.W_V(E).reshape(B, T, 4, 16).transpose(1, 2)
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(16)
future = torch.ones(T, T, dtype=torch.bool, device=E.device).triu(1)
real = X.ne(0)
mask = future[None] | ((~real[:, None, :]) & real[:, :, None])
A = torch.softmax(scores.masked_fill(mask[:, None], float("-inf")), dim=-1)
messages = A @ V
joined = messages.transpose(1, 2).reshape(B, T, 64)
updated = E + model.W_O(joined)
```
```python
# attention.html s16: single head, matrix form
A = (Q @ K.T / d_k**0.5 + mask).softmax(dim=-1)
E_new = E + (A @ V) @ W_O
```

**Figures and widgets worth reusing**
- `#s15-frame1` (attention.html): an 18-step "Next step" walkthrough of one forward pass with every number visible.
- `#s16-frame3-routing` (attention.html): the matrix pipeline with dimensions always in view, and clickable score cells.
- `figures/multihead/s01-v-independent.svg`: one receiver, two queries, two messages (river versus fisherman).
- `figures/multihead/s02-cap-map.svg`: the full two-head shape map, `[10×4]` to two heads to concat to `W_O` to `E'`.
- `figures/multihead/s03-data-pairs-tensors.svg` (the Lily `all_X`/`all_y` table) and `figures/multihead/s05-cap-scores.svg` (the results table).
- The live three-model comparison app is `word-lab/` (https://nipunbatra.github.io/attention/word-lab/). It runs WebGPU/WASM in the browser.
- Deep links:
  - https://nipunbatra.github.io/attention/attention.html#s16-frame3-routing
  - https://nipunbatra.github.io/attention/part3.html#s01-v-independent
  - https://nipunbatra.github.io/attention/part3.html#s02-cap-map
  - https://nipunbatra.github.io/attention/part3.html#s04-cap-weights
  - https://nipunbatra.github.io/attention/part3.html#s05-cap-scores

**The lecture's own summary lines**
- "Each token matches its query against its own and earlier keys, mixes the corresponding values, and receives a context-dependent update through a residual addition." (`s19`, "Intuitive")
- "Same receiver, two queries. Each head has its own keys, values and weights."
- "Concatenation joins the coordinates. The output projection learns how both heads contribute to the update."
- "Four heads keep 'owl' in view here, but all three outputs still have flaws."

**Connections**
- Builds on: L12 and L13 (single head, mask, position), L11 (tokenization, windows, cross-entropy, greedy decoding), and L10 (the train/validation/test split, checkpoint selection by validation loss).
- Reused by: L15 recaps "multiple heads … Concatenate messages. Project with W_O" and adds multi-head cross-attention. L16 ViT and L22 (live-coding a ViT in PyTorch) reuse the same block.

**Exam-style questions**
1. *Two heads of width 2 and one head of width 4 give the same total width. What can two heads do that one cannot?* Each head computes its own softmax over sources, so one head can favour river while another favours fisherman. One head applies the same source weights to every coordinate of its message.
2. *Why split TinyStories by story before making windows?* Overlapping windows from one story would otherwise land in both train and test. That leaks text and makes held-out loss look better than it is.
3. *Numeric.* Four-head test loss is 3.360 nats. Give the perplexity, and the tensor shape of the attention scores for a batch of 2 windows. Answer: `exp(3.360) ≈ 28.8` (the slide's 28.78 averages the per-seed perplexities 28.40, 29.49 and 28.46). The scores are `[B, heads, T, T] = [2, 4, 64, 64]`.

---

## Lecture 15: From Attention to Applications

**Aim**: Show how the same attention blocks become three architectures (decoder-only, encoder and encoder-decoder) by changing which positions may attend to which, which output embeddings the task reads, and what the training target is.

**Running example(s)**
- One sentence throughout: **"Raghav goes to school in Delhi."** Six word-sized teaching tokens, with punctuation omitted.
- Hindi target: **"राघव दिल्ली में स्कूल जाता है।"**. The start token is `[शुरू]` and the end token `[समाप्त]` (display labels for BOS/EOS).
- Decoder example: "Raghav goes to" predicts "school" (word 4). The message `m_3` is received at "to".
- Vocabulary head example: `d = 8`, `|V| = 50,000`. The logits are Raghav 1.0, bicycle −2.0, **goes 5.5**, ##ing −1.0, school 1.5, banana −3.0, to 3.0, and −8 for each of the other 49,993.
- NER: labels PERSON, ORGANIZATION, PLACE, OTHER. The output is `Raghav = PERSON`, `Delhi = PLACE`, and OTHER for the rest.
- Sentence classification: "The movie was really awful" gives NEGATIVE (mean pooling or CLS). The topic task (EDUCATION, TRAVEL, OTHER) uses an 8 → 6 → 3 classifier on the updated CLS.
- Cross-attention for predicting राघव from `[शुरू]`: the illustrative weights over the six English tokens are `[0.72, 0.04, 0.03, 0.09, 0.04, 0.08]`.
- The deck has 89 main frames plus an appendix (97 PDF pages) and is designed for 45 to 60 minutes. Every weight and label is authored; no model inference was run.

**Ideas in teaching order**
- Same input, different requested outputs (a continuation, a label per word, a Hindi translation). Three questions return throughout: *which positions can attend to each other, which embeddings the task uses, what the training target is*.
- Part I, decoder-only: a causal 6 × 6 mask. Read the last updated embedding through a vocabulary head (`1 × 8` times `8 × 50,000`), apply softmax over all 50,000 tokens, then greedy, append, repeat. Examples are text and code completion.
- Part II, encoder: a whole-input mask (every token reads every token). One shared classifier `z_i = e_i W + b` at all six positions gives NER. Six tokens give six contextual embeddings (BERT is the example family).
- Sentence labels: mean pooling `c = (e_1 + … + e_5)/5`, or a `[CLS]` token with its own trainable embedding-table row, learned through the loss. "Readout" is the choice of which encoder outputs to use.
- Part III, encoder-decoder with a fixed context: `c = Pool(e_1,…,e_6)` conditions a causal decoder, `p(y_t | y_<t, c)`. Three ways to feed `c`: add `u = cW_P` to every target embedding, concatenate `[e_t ∥ c]` then apply `W_F`, or prepend `u` before `[शुरू]`.
- The bottleneck: if pooling loses "Delhi", the decoder cannot revisit the source.
- Part IV, cross-attention: keep `e_1…e_6`. The query comes from the target, after causal self-attention at `[शुरू]`. Keys and values come from the encoder outputs. The message `c_1` is projected, added, then the MLP and vocabulary head predict राघव. A seven-step walkthrough generates the whole Hindi sentence.
- One decoder block holds two attention operations: causal self-attention, then cross-attention, then the FFN. Multi-head cross-attention follows the same rules.
- Part V and VI: three allowed-access patterns (full 6 × 6, causal 3 × 3, rectangular Hindi × English), the model families, a task-matching table and a decision tree.

**Key equations** (row-vector notation; `E` holds the final source embeddings, `D` the current target embeddings)
- Recap: `$\Delta e_i=m_iW_O,\quad e'_i=e_i+\Delta e_i$`, followed by the MLP and its residual. Multi-head messages are concatenated before `W_O`.
- Encoder self-attention: `$q_1=e_1W_Q,\;K=EW_K,\;V=EW_V,\;m_1=\operatorname{softmax}(q_1K^\top/\sqrt{d_k})V$`.
- NER head: `$z_i=e_iW+b$`, then softmax over 4 labels at each `i` (`W` is 8 × 4 in the shape slides).
- Fixed context: `$c=\operatorname{Pool}(e_1,\ldots,e_6)$`, `$p(y_t\mid y_{<t},c)$`.
- Cross-attention: `$q_t=e_tW_Q,\;k_i=e_iW_K,\;v_i=e_iW_V,\;\alpha_{ti}=\operatorname{softmax}_i\!\big(q_tk_i^\top/\sqrt{d_k}\big),\;c_t=\sum_i\alpha_{ti}v_i$`. "cₜ is the cross-attention message, just as mᵢ was the self-attention message."
- Shapes: `$E\in\mathbb R^{n\times d}$`, `$D\in\mathbb R^{m\times d}$`, `$A=\operatorname{softmax}(QK^\top/\sqrt{d_k})\in\mathbb R^{m\times n}$`, `$C=AV\in\mathbb R^{m\times d_v}$`. Per head, `$C_j=\operatorname{softmax}(Q_jK_j^\top/\sqrt{d_k})V_j$` with `$Q_j=DW_Q^{(j)}$` and `$K_j=EW_K^{(j)}$`.

**Worked numbers worth reusing**
1. Vocabulary softmax (`vocab-softmax`): goes 84.58%, to 6.94%, school 1.55%, Raghav 0.94%, ##ing 0.13%, bicycle 0.05%, banana 0.02%. The other 49,993 tokens together get 5.80% (each has logit −8). Greedy picks "goes", giving "Raghav goes".
2. CLS classifier (`readout-cls-classify`): logits [2, 1, 0] give `[0.665, 0.245, 0.090]` for EDUCATION, TRAVEL, OTHER. The CLS loss example (`cls-learning`): `−log 0.8 ≈ 0.223`.
3. Cross-attention for राघव (`cross-raghav-weights`): the scaled scores are `log(0.72), log(0.04), …`, so the softmax returns exactly `[0.72, 0.04, 0.03, 0.09, 0.04, 0.08]`.
4. Appendix (`cross-numerical`, `cross-numerical-values`): the scaled scores `[0.2, 0.1, 0.3, 2.0]` for Raghav, goes, school, Delhi give a softmax of `[0.110, 0.100, 0.122, 0.668]`. With `V = [[1,0],[0,1],[1,1],[2,1]]`, `c_t = [0.110 + 0.122 + 2(0.668), 0.100 + 0.122 + 0.668] ≈ [1.57, 0.89]`.
5. Shape counts (`readout-*`, `cross-attention-shapes`):
   - NER: six words, width 8, an `8 × 4` classifier, so `6 × 4` scores.
   - CLS: seven rows of 8, one selected `1 × 8` row, then `1 × 3` scores.
   - Cross-attention with `m = 4` target positions and `n = 6` source positions: `A` is `4 × 6`, and every row sums to 1.

**Code**: The deck shows no code. Its formulas are on slides `encoder-qkv`, `cross-equation` and `multihead-cross`, quoted above. Pseudo-steps from the guide: `c = Pool(E)`, `u = c W_P`, prepend `u` before `[शुरू]`.

**Figures and widgets worth reusing** (all are inline SVGs in `interactives/from-attention-to-applications/index.html`; the deck has no image files)
- `#attention-recap`: one diagram of positional encoding, self-attention and multiple heads, giving a contextualized embedding per token.
- `#ner-shared`: one shared classifier over six positions.
- `#source-bottleneck`: everything squeezed through the pooled `c`.
- `#cross-source-match`: target Q against English K/V.
- `#source-lookup-0` to `#source-lookup-6`: an interactive seven-step Hindi generation with self-attention and cross-attention weights at each step.
- `#three-attention-patterns` (full, causal, rectangular) and `#model-families` (the three architecture portraits).
- Deep links:
  - https://nipunbatra.github.io/dl-teaching/interactives/from-attention-to-applications/#cls-readout
  - https://nipunbatra.github.io/dl-teaching/interactives/from-attention-to-applications/#source-bottleneck
  - https://nipunbatra.github.io/dl-teaching/interactives/from-attention-to-applications/#cross-source-match
  - https://nipunbatra.github.io/dl-teaching/interactives/from-attention-to-applications/#three-attention-patterns
  - https://nipunbatra.github.io/dl-teaching/interactives/from-attention-to-applications/#model-families

**The lecture's own summary lines**
- "Filled cell = connection allowed, not an attention weight."
- "Keys determine the weights; values supply the information sent to the target."
- "Source states stay fixed; the query-dependent message changes."
- The exit question's answer: "NER uses whole-input attention, all token states and token labels. Classification uses a whole-input readout and one sentence label. Translation uses source encoding, causal target states, cross-attention and next-target-token supervision."
- Decision tree: "Do I need to generate new tokens?" No leads to the encoder. Yes leads to "Separate source pathway?", where no gives decoder-only and yes gives encoder-decoder.

**Connections**
- Builds on: L12 to L14 (the `m_i`, `Δe_i`, `e'_i` notation is reused on purpose; multi-head concatenation before `W_O`; causal masks; positions) and L11's vocabulary softmax and greedy decoding.
- Reused by:
  - L16 ViT: "full attention, embedding updates and CLS readout" is this lecture's encoder plus CLS, with patches as tokens. The exit slide says "Next lecture: what if our tokens are not words?".
  - L17 CLIP: two encoders read out to one vector each, the "retrieval" readout here.
  - L18 and L19 VLMs: "visual prefixes, cross-attention" generalize the prepend and cross-attention designs.
  - L26 object detection: DETR-style set prediction uses decoder queries that cross-attend to encoder features. This is a plain fact about DETR; the L26 deck is not checked here.

**Exam-style questions**
1. *Where do Q, K and V come from in cross-attention, and why can the source K/V be cached during generation?* Q comes from the current target embedding, after causal self-attention. K and V come from the final encoder embeddings. The source does not change between target steps, so its K/V can be computed once. Only the query, and so the weights and `c_t`, changes at each step.
2. *Why does a fixed pooled `c` create a bottleneck that cross-attention removes?* Every source detail must survive pooling into one fixed-width vector. A later step cannot re-query the original "Delhi" embedding. Cross-attention keeps all `n` source states, and each target query can make a fresh weighted lookup.
3. *Numeric.* Scaled scores `[0.2, 0.1, 0.3, 2.0]` and values `[[1,0],[0,1],[1,1],[2,1]]`: give the weights and `c_t`. Answer: `e^s ≈ [1.221, 1.105, 1.350, 7.389]`, which sum to 11.065, so `α ≈ [0.110, 0.100, 0.122, 0.668]`. Then `c_t ≈ [0.110 + 0.122 + 1.336, 0.100 + 0.122 + 0.668] = [1.57, 0.89]`.

---

## Module in one paragraph

The module keeps one objective, predicting the next token, and changes how a token gathers context. L11 builds a 27-character name model: look up 2-number rows, concatenate a 3-token window (6 numbers), apply 32 ReLU units, then 27 logits, softmax and cross-entropy, and generate by sampling and shifting. Its first-layer width grows with the window. L12 swaps the window for a weighted summary whose weights are computed from content. The query is matched to each key by a dot product, softmax turns the scores into weights, and the weights mix the values into a message `m_i`. `W_O` turns `m_i` into an update `Δe_i`, and the residual adds it: `e'_i = e_i + Δe_i`. In the hand-built "bank" example, river gets 0.414 of bank's attention in one sentence and cheque 0.400 in the other. L13 derives Q/K/V as shared projections, divides by `√d_k` against saturation, masks the future with `-∞`, predicts water (0.367) or teller (0.521) from the final "the", and adds position information, because attention alone is order-blind. L14 writes everything as `E' = E + softmax(QK^T/√d_k + M)VW_O`, splits it into heads, and measures four heads (perplexity 28.78) against one head (31.34) and an MLP (51.59) on TinyStories. L15 rearranges the same block into encoders (full mask, token or CLS readouts), decoder-only models (causal mask, vocabulary head) and encoder-decoders, where cross-attention lets each target query look up retained source keys and values.

## The three things a student must remember

1. **Queries and keys choose the weights; values carry the information.** For each receiver, softmax over `q_i·k_j/√d_k` gives weights that sum to 1. The weights mix `v_j` into a message, `W_O` maps it to `Δe_i`, and the residual adds it: `e'_i = e_i + Δe_i`.
2. **The mask and the readout define the architecture.** A full mask with per-token or CLS readout makes an encoder (labels). A causal mask with a last-position vocabulary head makes a decoder (generation). Target queries over source keys and values make cross-attention (translation).
3. **Attention is order-blind and position-agnostic until position is added.** Learned or sinusoidal rows are added to the input; ALiBi and RoPE inject relative position into the scores or into Q and K. The causal mask controls visibility, not order.
