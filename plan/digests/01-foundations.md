# Digest: Foundations module (schedule Lectures 1 to 4)

Sources read on `origin/master` of `dl-teaching` (commit 118c7bce): `lecture1/L1-probabilistic-view.typ`, `lecture2/L2-linear-to-mlp.typ`, cheat sheets `S01` to `S04`, the figure folders and their generator scripts, and the companion notebooks linked from the schedule. The published `L1.pdf` (110 pages, built 13 Aug 2026) and `L2.pdf` (72 pages, built 18 Aug 2026) match the source.

## How the decks map to the schedule (read this first)

- One Typst deck, `L1-probabilistic-view.typ` (title "Where Deep Learning Losses Come From", subtitle "Likelihood, cross-entropy, and regularization from probability"), serves schedule Lectures 1, 2 and 3. The deck has no internal session breaks. The split below follows the schedule titles and the 3-minute summary videos ("Where MSE and cross-entropy come from" for L2; "Outliers: how the noise model picks the loss" and "Priors: where regularization comes from" for L3):
  - **L1** = deck Parts I to III: "Three familiar losses" through "Answer: uniform support" (source lines 31 to 822).
  - **L2** = deck Part IV up to "Homoscedastic vs input-dependent uncertainty" (lines 824 to 1027), plus Part V "Classification: cross-entropy" (lines 1516 to 1711).
  - **L3** = the robust-regression block inside Part IV ("A few unusual observations..." to "Interactive: likelihood → loss", lines 1028 to 1515), plus Parts VI to VIII (Bayes, MAP, priors, full Bayes; lines 1712 to end).
- The cheat sheets split differently: S01 = primer + likelihood + NLL (matches L1); **S02 page 2 is MAP and regularization**, so S02 = losses + MAP; **S03 has no MAP at all**: it is output heads, stable logit losses, heteroscedastic and robust regression, PyTorch loss semantics and debugging. So "L3 = MAP" per the schedule is backed by S02, not S03.
- The decks use deck-era numbering: the L2 deck (schedule Lecture 4) says "Lecture 1 chose the loss; Lecture 2 expands the prediction rule", and its saturation slide says "Lecture 3 will connect it to the learning signal" (meaning backprop, now schedule Lecture 6).
- Neither deck shows any code on a slide. All plots in the L1 deck are drawn natively in Typst; the PNG/SVG files in `lecture1/figures/` are leftovers from the archived Marp deck (`_marp-archive/lecture1-L1-probabilistic-view.marp.md`) and are not used by the current deck. The L2 deck does use files from `lecture2/figures/`.

## Notation conventions to match (all four lectures)

- Data: $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$. Random target $Y$, observed $y$; random input $\mathbf{X}$, observed $\mathbf{x}$. A single fixed observation is $(\mathbf{x}^\star, y^\star)$. The L4 deck switches to row superscripts $\mathbf{x}^{(i)}, y^{(i)}$ for a data matrix.
- Parameters: bold $\boldsymbol{\theta}$ ("the model's learned weights and biases"). Model: $p_{\boldsymbol\theta}(y \mid \mathbf{x})$, read "model $\boldsymbol\theta$'s probability or density for $y$, given $\mathbf{x}$". Mean function $f_{\boldsymbol\theta}(\mathbf{x})$; augmented input $\tilde{\mathbf{x}}_i = (1, \mathbf{x}_i)$, $\boldsymbol\theta = (\theta_0, \theta_1, \dots, \theta_d)$.
- Likelihood: $L(\boldsymbol\theta; \mathcal{D}) = p(\mathcal{D} \mid \boldsymbol\theta)$, often just $L(\theta)$. Log-likelihood on the coin slides: $\ell(\theta) := \log L(\theta)$.
- Loss: per-example $\ell_i(\boldsymbol\theta) = -\log p_{\boldsymbol\theta}(y_i \mid \mathbf{x}_i)$; empirical risk $\hat R(\boldsymbol\theta)$; dataset losses written $\mathcal{L}$ ($\mathcal{L}_\text{BCE}$, $\mathcal{L}_\text{MAP}$); least-squares and logistic objectives written $J(\boldsymbol\theta)$. Note the clash: $\ell$ is the log-likelihood on the coin slides but the per-example NLL everywhere else.
- Residual $r_i = y_i - \hat y_i$; influence $\psi(r) = d\ell/dr$. Noise std $\sigma$, but $\sigma(z)$ is also the sigmoid. Prior std $\tau$, $\lambda = 1/(2\tau^2)$; Laplace noise scale $s_L$; Laplace prior scale $a$; Student-$t$ $(\nu, s)$.
- L4: $z = \mathbf{w}^\top\mathbf{x} + b$, activation $a = \phi(z)$, hidden $\mathbf{h} = \phi(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)$, logits $\mathbf{z} = \mathbf{W}_2\mathbf{h} + \mathbf{b}_2$, widths $d \to m \to K$; pre-activations $u_j$ in the XOR slides.

---

## Lecture 1: Where Deep Learning Losses Come From I (distributions, likelihood, and loss)

**Aim.** Show that a training loss is the negative log of a likelihood: a network outputs a distribution over $Y$, the observed data score candidate parameters, and logs turn the product into a trainable sum.

**Running examples.**
- Bernoulli(0.75) PMF; a fixed input $\mathbf{x}^\star$ with $p_{\boldsymbol\theta_A}(Y{=}1 \mid \mathbf{x}^\star) = 0.80$ and candidate $\boldsymbol\theta_B$ giving $0.30$.
- One Gaussian observation $y^\star = 1$ under $\mathcal{N}(\mu, 1)$: $L(1) \approx 0.399$, $L(0) \approx 0.242$.
- Rain: $P(R_1)=0.4$, $P(R_2 \mid R_1)=0.8$, $P(R_2 \mid \neg R_1)=0.2$. Smoke alarms: $P(A_1{=}1 \mid S)=0.9$, $P(A_2{=}1 \mid S)=0.6$.
- Coins: fair $\theta=0.5$ vs head-biased $\theta=0.8$ on data $H,T$ and $H,H$.
- The main coin sequence: $H,H,T,T,T,H,H,T,T,T$, so $n_H = 4$, $n_T = 6$; candidates $\theta=0.4$ and $\theta=0.7$.
- Two Gaussian measurements $y_1=1$, $y_2=2$, candidates $\mu=1.5$ and $\mu=0$. Poisson count $y=3$.

**Ideas in teaching order.**
- Three familiar losses (MSE, BCE, $\mathcal{L}_\text{data} + \lambda\|\boldsymbol\theta\|_2^2$), one objective: "Derive each term from an explicit likelihood or prior."
- Probability primer: support $\mathcal{S}_Y$, PMF vs PDF; "Density is not probability" (a density can exceed 1; only integrals are probabilities); Uniform (support matters); Gaussian (two knobs $\mu$, $\sigma$).
- Read $p_{\boldsymbol\theta}(y \mid \mathbf{x})$ one symbol at a time. A neural network maps $\mathbf{x}$ to distribution parameters; the point prediction is the mean (regression) or mode (classification).
- Prediction fixes the model and varies the outcome; likelihood fixes the observation and varies $\boldsymbol\theta$. Likelihood scores need not sum to 1 over parameters.
- A dataset needs a joint likelihood; the product needs conditional independence. Independent and identically distributed are separate claims (rain example fails independence; two alarms or two sensors fail identical).
- i.i.d. gives $p(\mathcal{D} \mid \boldsymbol\theta) = \prod_i p_{\boldsymbol\theta}(y_i \mid \mathbf{x}_i)$; MLE picks the argmax.
- Why logs: log is strictly increasing so the argmax is unchanged (Poisson example peaks at $\hat\lambda = 3$ in both curves), and products become stable sums (underflow example).
- Coin MLE by calculus: stationary point $n_H/(n_H+n_T) = 0.4$, second derivative confirms a maximum.
- Maximizing likelihood = minimizing NLL; empirical risk = average NLL; a Uniform model makes an out-of-support target cost $-\log 0 = +\infty$.

**Key equations.**
$$p_Y(y) \ge 0,\quad \int_{\mathcal{S}_Y} p_Y(y)\,dy = 1,\qquad P(a \le Y \le b) = \int_a^b p_Y(y)\,dy$$
$$L(\boldsymbol\theta;\mathcal{D}) = p(\mathcal{D}\mid\boldsymbol\theta) = \prod_{i=1}^n p_{\boldsymbol\theta}(y_i \mid \mathbf{x}_i)\quad\text{(conditional independence)}$$
$$\hat{\boldsymbol\theta}_\text{MLE} = \arg\max_{\boldsymbol\theta} p(\mathcal{D}\mid\boldsymbol\theta)$$
$$\log L(\theta) = n_H\log\theta + n_T\log(1-\theta),\qquad \theta^\star = \frac{n_H}{n_H+n_T}$$
$$\ell_i(\boldsymbol\theta) = -\log p_{\boldsymbol\theta}(y_i\mid\mathbf{x}_i),\qquad \hat R(\boldsymbol\theta) = \frac1n\sum_i \ell_i(\boldsymbol\theta) = -\frac1n\log p(\mathcal{D}\mid\boldsymbol\theta)$$

**Worked numbers worth reusing.**
1. "Work two tiny Bernoulli datasets": data $H,T$: $L(0.5)=0.25$, $L(0.8)=0.16$ (fair wins); data $H,H$: $L(0.5)=0.25$, $L(0.8)=0.64$ (biased wins).
2. "Coin flips: a likelihood you can compute" and "...calculate the log-likelihood": $L(\theta)=\theta^4(1-\theta)^6$; $L(0.4)\approx 0.00119$, $L(0.7)\approx 0.000175$ ("about $6.8\times$ larger at $\theta=0.4$"); $\ell(0.4)\approx -6.73$, $\ell(0.7)\approx -8.65$.
3. "The second derivative confirms a maximum": $-4/(0.4)^2 - 6/(0.6)^2 = -25 - 16.67 \approx -41.67 < 0$, so $\hat\theta_\text{MLE}=0.4$.
4. "Why logs? Products become stable sums": 200 outcomes each with probability $0.01$: $L = 0.01^{200} = 10^{-400}$, below the smallest float64 subnormal $4.94\times10^{-324}$, so it underflows to 0; $\log L = 200\log(0.01) \approx -921.03$.
5. "Worked example: dependence changes the product": $P(R_1,R_2) = 0.4\times0.8 = 0.32$; $P(R_2) = 0.4\times0.8 + 0.6\times0.2 = 0.44$; the independence product $0.4\times0.44 = 0.176 \ne 0.32$. (Also "Two Gaussian measurements": $L(1.5)\approx 0.352\times0.352 = 0.124$ vs $L(0)\approx 0.242\times0.054 = 0.013$.)

**Code.** No code on the slides. The linked notebook `notebooks/L01/00_likelihood_iid_worked_examples.ipynb` reproduces the coin example; verbatim:
```python
coin_data = torch.tensor([1, 1, 0, 0, 0, 1, 1, 0, 0, 0.0])

def coin_log_likelihood(theta_candidate):
    coin = D.Bernoulli(probs=theta_candidate)
    total = torch.tensor(0.0)
    for y_i in coin_data:
        total = total + coin.log_prob(y_i)
    return total
```

**Figures worth reusing.** The current deck draws its plots inline, so these are optional leftovers from the Marp version, generated by `lecture1/diagrams/make_figs.py` (and `figs_a.py`), each as `.svg` and `.png`:
- `lecture1/figures/net_to_distribution.png`: small network whose output is a bell labelled $p_\theta(y\mid x)$. Best one-glance picture of "network outputs a distribution".
- `lecture1/figures/density_area.png`: density with a shaded interval $P(a\le Y\le b)$.
- `lecture1/figures/uniform.png`: Uniform density with $-\log 0 = +\infty$ outside the support.
- `lecture1/figures/gaussian_params.png`: changing $\mu$ vs changing $\sigma$.

**The lecture's own summary lines.**
- "Likelihood is a *score as a function of* $\boldsymbol\theta$; the scores need not sum to $1$ over parameter settings."
- "Independence is what licenses multiplication."
- "Logs make optimization easier *and* prevent numerical underflow."
- "Minimizing empirical NLL is maximum likelihood; $1/n$ only rescales the objective."
- "Zero likelihood becomes an infinite training penalty."
- S01 banner: "Probability models possible outcomes. Likelihood scores parameters using the outcome that occurred."

**Connections.** Builds on basic probability (PMF, PDF, chain rule). The NLL $\ell_i$ and empirical risk $\hat R$ are what Lectures 5 and 7 optimize with minibatch gradients; the support idea returns in Lecture 9 ("Valid outputs by design: softplus to softmax"); surprise $-\log p$ returns as entropy, cross-entropy and KL in Lectures 23 and 24; $p(y\mid x)$ vs $p(x)$ is revisited at the start of Lecture 25.

**Three exam-style questions.**
1. *Conceptual.* For fixed $\mathbf{x}^\star$, $p_{\boldsymbol\theta_A}(Y{=}1\mid\mathbf{x}^\star)=0.8$ and $p_{\boldsymbol\theta_B}(Y{=}1\mid\mathbf{x}^\star)=0.3$. Why may $0.8$ and $0.3$ not sum to 1, while $0.8$ and $0.2$ must? *Answer:* $0.8 + 0.2$ is a distribution over outcomes $y$ for one fixed $\boldsymbol\theta$ (prediction). $0.8$ and $0.3$ score one fixed observation under two parameter settings (likelihood), which is a function of $\boldsymbol\theta$, not a distribution over it.
2. *Conceptual.* Two sensors measure the same temperature independently with $\sigma_1 \ne \sigma_2$. Is the likelihood a product? Are the data i.i.d.? *Answer:* Yes, a product (independent), but not identically distributed: each factor uses its own $\sigma_i$.
3. *Numeric.* Observed $H,H,T$. Compare $\theta=0.5$ and $\theta=0.8$ by likelihood and by NLL; give the MLE. *Answer:* $L(0.5)=0.125$, $L(0.8)=0.8\cdot0.8\cdot0.2=0.128$; NLL $=-\ln 0.125 \approx 2.079$ vs $-\ln 0.128 \approx 2.056$, so $0.8$ wins narrowly; $\hat\theta_\text{MLE} = 2/3$. (My example, built on the deck's coin template.)

---

## Lecture 2: Where Deep Learning Losses Come From II (MLE, regression, and classification)

**Aim.** Derive the two workhorse losses: a Gaussian observation model gives MSE (and least squares), and Bernoulli or categorical models with sigmoid or softmax give binary and multi-class cross-entropy.

**Running examples.**
- Regression as signal plus noise: synthetic scatter around the line $1.2 + 0.28x$ with vertical Gaussian bells drawn at $x = 1.5, 4.0, 6.5, 9.0$ ("What the assumption means visually").
- Linear regression $\hat y_i = \theta_0 + \theta_1 x_i$, then $\boldsymbol\theta^\top\tilde{\mathbf{x}}_i$.
- Logits $\mathbf{z} = (1.4, 0.6, -0.8)$ through softmax (bar chart, no numbers on the slide); softmax bars for $(2, 1, -0.5)$.
- One binary example $y=1$, $p=0.8$ (BCE vs MSE).
- True-class probabilities $p_y = 0.9, 0.1, 0.001$.

**Ideas in teaching order.**
- $y_i = f_{\boldsymbol\theta}(\mathbf{x}_i) + \epsilon_i$, $\epsilon_i\sim\mathcal{N}(0,\sigma^2)$: "Gaussian noise lives in the *output direction*, conditional on $\mathbf{x}$: a vertical bell at every input."
- Gaussian NLL is squared error plus a constant; "MSE = Gaussian NLL, up to constants."
- Linear mean: Gaussian MLE becomes ordinary least squares; the normal equations give a closed form ("neural networks keep the MSE objective but use iterative optimization").
- Why earlier courses ignored $\sigma^2$: with one fixed global $\sigma^2$ the $\frac n2\log\sigma^2$ term is constant.
- How $\sigma^2$ is chosen: global $\hat\sigma^2_\text{MLE}$ from residuals, or a heteroscedastic network predicting $(\mu_i, s_i)$ with $s_i = \log\sigma_i^2$; validate interval coverage.
- Classification outputs probabilities: logits, then sigmoid or softmax.
- Bernoulli $p^y(1-p)^{1-y}$ gives BCE; BCE matches a binary label better than MSE (MSE's Gaussian "permits all real $Y$").
- Logistic regression = Bernoulli MLE; softmax and categorical give cross-entropy $-\log p_\text{true class}$.
- Confident mistakes are expensive.

**Key equations.**
$$-\log p(\mathcal{D}\mid\boldsymbol\theta) = \frac{1}{2\sigma^2}\sum_i \big(y_i - f_{\boldsymbol\theta}(\mathbf{x}_i)\big)^2 + C$$
$$\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}\,\hat{\boldsymbol\theta} = \tilde{\mathbf{X}}^\top\mathbf{y},\qquad \hat{\boldsymbol\theta} = (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^\top\mathbf{y}$$
$$\mathcal{L}(\boldsymbol\theta;\sigma^2) = \frac{1}{2\sigma^2}\sum_i r_i^2 + \frac n2\log\sigma^2 + C,\qquad \hat\sigma^2_\text{MLE} = \frac1n\sum_i (y_i - \hat\mu_i)^2$$
$$-\log p_{\boldsymbol\theta}(y\mid\mathbf{x}) = -y\log p - (1-y)\log(1-p),\qquad p = \sigma(z) = \frac{1}{1+e^{-z}}$$
$$p_k = \frac{e^{z_k}}{\sum_j e^{z_j}},\qquad -\log p_{\boldsymbol\theta}(\mathbf{y}\mid\mathbf{x}) = -\sum_k y_k\log p_k = -\log p_\text{true class}$$
($\tilde{\mathbf{X}} = [\mathbf{1}\ \mathbf{X}]$ is the augmented data matrix; $s_i = \log\sigma_i^2$ is the predicted log-variance.)

**Worked numbers worth reusing.**
1. "Bernoulli NLL gives BCE": $y=1$, $p=0.8$ gives $-\log 0.8 \approx 0.22$; $p=0.2$ gives $-\log 0.2 \approx 1.61$.
2. "Why BCE matches a binary label better than MSE": $y=1$, $p=0.8$: $\ell_\text{MSE} = (1-0.8)^2 = 0.04$ vs $\ell_\text{BCE} = -\log 0.8 \approx 0.223$.
3. "Confident mistakes are expensive": $p_y = 0.9 \Rightarrow \ell\approx 0.105$; $0.1 \Rightarrow 2.303$; $0.001 \Rightarrow 6.908$.
4. "Why use logits?": $z\to-\infty$: $p\to0$; $z=0$: $p=0.5$; $z\to+\infty$: $p\to1$.
5. S03 cheat sheet "First prove one example by hand": binary $y=1,p=0.8$: $0.223$; 3-class $p_c=0.7$: $-\log 0.7\approx0.357$; Gaussian $r=2,\sigma=1$: $r^2/(2\sigma^2)=2$ plus constants.

**Code.** No code on the slides. The deck links only the two Lecture 3 notebooks; two unlinked notebooks in `notebooks/L01/` cover this lecture. Verbatim from `02_softmax_cross_entropy.ipynb` (stable softmax; the same notebook checks the gradient $\nabla_z\mathcal{L} = p - y$, which the current slides do not show):
```python
def softmax(z):
    z = z - z.max()          # numerically stable
    e = np.exp(z)
    return e / e.sum()
```
S03 gives the fused stable forms: $\ell(z,y) = \max(z,0) - zy + \log(1+\exp(-|z|))$ and $\ell(\mathbf{z},c) = -z_c + \operatorname{logsumexp}(\mathbf{z})$.

**Figures worth reusing.** As for L1, these come from the Marp-era generator `lecture1/diagrams/make_figs.py` (`.svg` + `.png`) and are not in the current deck:
- `lecture1/figures/regression_conditionals.png`: fitted line $f_\theta(x)$ with a sideways Gaussian at four inputs over the scatter. Best single picture of "MSE = Gaussian NLL".
- `lecture1/figures/softmax_pipeline.png`: input, neural net, logits $z$, softmax, probabilities summing to 1.
- `lecture1/figures/neglog_curve.png`: $-\log p_y$ with the three points $p=0.9, 0.1, 0.001$ labelled ($\ell\approx 0.11, 2.30, 6.91$).
- `lecture1/figures/gradient_bars.png`: bars for $p$, one-hot $y$ and $p - y$ = gradient (matches notebook 02, not a current slide).
- `lecture1/figures/sigmoid.png`: logistic sigmoid.

**The lecture's own summary lines.**
- "MSE = Gaussian NLL, up to constants."
- "Ordinary least squares is Gaussian maximum likelihood for a linear mean."
- "For point predictions, fixed-variance Gaussian MLE gives the same fitted model as MSE."
- "Predict *log-variance* so every real $s_i$ maps to a positive variance $e^{s_i}$."
- "Binary cross-entropy = Bernoulli NLL." / "MSE is possible; BCE is the negative log-likelihood matched to a binary observation."
- "Softmax guarantees $p_k>0$ and $\sum_k p_k=1$, as a categorical model requires."
- "The negative log makes a confident wrong prediction a large loss."

**Connections.** Builds on L1's $\ell_i = -\log p$. The linear model and logits become the output layer of the MLP in Lecture 4 (the L4 deck: "We will not re-derive the losses"). Log-variance and softmax parameterizations return in Lecture 9 (valid outputs by design). Softmax cross-entropy over a vocabulary is the next-token loss of Lecture 11; softmax reappears as attention weights in Lectures 12 to 14 and in CLIP's contrastive loss (Lecture 17). Lectures 23 and 24 recast the classification loss as a KL divergence ("classification loss and knowledge distillation as KL"). The VAE loss in Lecture 25 is "reconstruction plus KL", and its reconstruction term is a negative log-likelihood of exactly this kind (I did not check whether that deck uses a Gaussian or a Bernoulli decoder).

**Three exam-style questions.**
1. *Conceptual.* Why can an earlier ML course train linear regression with plain MSE and never mention $\sigma^2$? When must you keep the $\log\sigma^2$ term? *Answer:* with one fixed global $\sigma^2$, $\frac n2\log\sigma^2$ is constant and $1/(2\sigma^2)$ only rescales, so the argmin is unchanged. Keep it when $\sigma$ (or $\sigma_i$) is learned; otherwise the model could inflate variance freely.
2. *Conceptual.* For a label $y\in\{0,1\}$, what distribution does MSE implicitly assume, and why is BCE the better match? *Answer:* MSE is the NLL of $\mathcal{N}(p,\sigma^2)$, which puts density on every real $Y$; BCE is the Bernoulli NLL, which allows only 0 and 1.
3. *Numeric.* A 3-class network outputs logits $\mathbf{z} = (2, 1, -0.5)$ and the true class is 1. Find $p_1$ and the cross-entropy. *Answer:* $e^2 = 7.389$, $e^1 = 2.718$, $e^{-0.5} = 0.607$, sum $10.714$; $p_1 = 0.690$; $\ell = -z_1 + \log\sum_j e^{z_j} = -2 + 2.372 = 0.372$. (Logits from the deck's softmax bar chart; the numbers are my calculation.)

---

## Lecture 3: Likelihood to Loss in Practice (MAP, regularization, and robust observation models)

**Aim.** Show that the noise model picks how errors are penalized (Gaussian, Laplace, Student-$t$, Uniform), and that a prior on the parameters picks the regularizer (Gaussian prior gives $L_2$, Laplace prior gives $L_1$), so the training objective is NLL plus negative log-prior.

**Running examples.**
- **Robust 12-point line** ("SYNTHETIC · COMPUTED"): ten typical points $(-3.0,-2.8), (-2.4,-2.2), (-1.8,-1.9), (-1.2,-0.9), (-0.6,-0.5), (0.0,0.2), (0.7,0.8), (1.4,1.2), (2.0,2.2), (2.6,2.4)$ and two outliers $(-2.2, 3.3)$, $(2.2, -3.2)$. Fitted $(\theta_0,\theta_1)$: ten typical points only $(0.070, 0.955)$; Gaussian $(-0.040, 0.397)$; Laplace $(0.044, 0.906)$; Student-$t$ $\nu=3,s=1$ $(0.053, 0.882)$. Student-$t$ sweep: $\nu=1$ $(0.064,0.928)$; $\nu=3,s=0.5$ $(0.065,0.934)$; $\nu=3,s=1.5$ $(0.036,0.807)$; $\nu=30$ $(-0.014,0.581)$.
- **Three heads**: $\mathcal{D} = (H,H,H)$, $L(\theta)=\theta^3$, prior $p(\theta) = 6\theta(1-\theta)$.
- **Two coins**: fair $F$ with $P(H\mid F)=0.5$, biased $B$ with $P(H\mid B)=0.9$; priors $0.90/0.10$ and then $0.10/0.90$.
- **Polynomial overfitting**: eight points at $x = 0.1,\dots,0.9$ with $y = (0.36, 0.60, 0.81, 1.00, -0.03, -0.85, -1.13, -0.81)$, trend $\sin(2\pi x)$, degrees 1, 3, 7.
- **Four-point MAP regression**: $x = (-1, 0, 1, 2)$, $y = (-0.8, 1.8, 5.3, 7.9)$ from $f_\text{true}(x) = 2 + 3x$; noise $\sigma = 0.5$; prior $(b,w)\sim\mathcal{N}(\mathbf{0},\tau^2\mathbf{I})$ with $\tau = 0.5$, then $\tau = 0.1$.
- **Scalar shrinkage**: MLE $= 2$, $s = 0.6$, prior $\tau = 2$ or $0.5$; ridge vs $L_1$ with $z = 0.6$.

**Ideas in teaching order.**
- Two outliers flatten the MSE line; a robust noise model allows occasional large residuals.
- Compare tails on a log scale; at $r=4$ heavy tails are about $68\times$ less surprised.
- One observation's gradient: residual density, NLL, $\psi(r)$, then $\nabla_{\boldsymbol\theta}\ell_i = -\psi(r_i)\nabla_{\boldsymbol\theta}f_{\boldsymbol\theta}(\mathbf{x}_i)$. Gaussian influence grows with $|r|$, Laplace is constant, Student-$t$ eventually decreases.
- Bayes' rule with $\boldsymbol\theta$ for $A$ and $\mathcal{D}$ for $B$; three heads do not prove $\theta = 1$; a prior tempers it (posterior peak $0.8$).
- Two-coin Bayes in four steps: likelihoods, prior times likelihood, normalize by the evidence, change the prior. "posterior odds = prior odds × likelihood ratio".
- MAP $=$ NLL $+$ negative log-prior. Every learner has an inductive bias; a prior makes part of it explicit.
- Gaussian prior gives $L_2$ ($\lambda = 1/(2\tau^2)$); ridge shrinks by $\alpha = \tau^2/(\tau^2+s^2)$; Laplace prior gives $L_1$ and soft thresholding (exact zeros).
- Prior strength trades overfitting against underfitting.
- Full Bayes keeps $p(\boldsymbol\theta\mid\mathcal{D})$ and averages predictions; deep learning usually uses MLE or MAP, ensembles, dropout approximations or variational inference.

**Key equations.**
$$\nabla_{\boldsymbol\theta}\ell_i = -\psi(r_i)\,\nabla_{\boldsymbol\theta}f_{\boldsymbol\theta}(\mathbf{x}_i),\qquad \psi_\text{Gauss}=\frac{r}{\sigma^2},\ \psi_\text{Laplace}=\frac{\operatorname{sign}(r)}{s_L},\ \psi_t = \frac{(\nu+1)r}{\nu s^2 + r^2}$$
$$p(\boldsymbol\theta\mid\mathcal{D}) = \frac{p(\mathcal{D}\mid\boldsymbol\theta)\,p(\boldsymbol\theta)}{p(\mathcal{D})}$$
$$\hat{\boldsymbol\theta}_\text{MAP} = \arg\min_{\boldsymbol\theta}\Big(\underbrace{-\log p(\mathcal{D}\mid\boldsymbol\theta)}_\text{NLL} \underbrace{-\log p(\boldsymbol\theta)}_\text{negative log-prior}\Big)$$
$$\boldsymbol\theta\sim\mathcal{N}(\mathbf{0},\tau^2\mathbf{I}):\quad \mathcal{L}_\text{MAP} = \text{NLL} + \underbrace{\tfrac{1}{2\tau^2}}_{\lambda}\|\boldsymbol\theta\|_2^2 + C,\qquad \hat\theta_\text{MAP} = \frac{\tau^2}{\tau^2+s^2}\,\hat\theta_\text{MLE}$$
$$\text{Laplace prior: } \hat\theta_\text{MAP} = \operatorname{sign}(z)\max(|z|-\lambda, 0),\quad \lambda = s^2/a$$
$$p(y^\star\mid\mathbf{x}^\star,\mathcal{D}) = \int p_{\boldsymbol\theta}(y^\star\mid\mathbf{x}^\star)\,p(\boldsymbol\theta\mid\mathcal{D})\,d\boldsymbol\theta$$
($z = \hat\theta_\text{MLE}$ is the centre of a quadratic likelihood of width $s$; $a$ is the Laplace prior scale.)

**Worked numbers worth reusing.**
1. "A large residual is less surprising under heavy tails" and "Large residuals create different gradient magnitudes": at $r=4$ with unit scales, densities Gaussian $\approx 0.000134$, Laplace $\approx 0.00916$, Student-$t$ ($\nu=3$) $\approx 0.00916$ ("about $68\times$"); gradient factors $|\ell'(4)| = 4$, $1$, $16/19\approx0.84$.
2. "Step 1" to "Step 4" (two coins): $P(HHH\mid F) = 0.125$, $P(HHH\mid B) = 0.729$, ratio $\approx 5.83$. Prior $0.90/0.10$: weights $0.1125$, $0.0729$, evidence $0.1854$, posterior $P(F\mid\mathcal{D})\approx0.607$, $P(B\mid\mathcal{D})\approx0.393$. Prior $0.10/0.90$: evidence $0.6686$, posterior $0.019/0.981$.
3. "A prior tempers the conclusion from three flips": MLE $\theta=1$; posterior $\propto\theta^4(1-\theta)$ peaks at $\theta = 4/5 = 0.8$.
4. "The likelihood peaks near the generating line" to "The prior changes the fitted line": squared error of $(2,3)$ is $0.180$, of $(0,1)$ is $56.580$, of MLE $(2.07, 2.96)$ is $0.162$; MAP with $\tau=0.5$ is $\hat y = 1.79 + 2.62x$, with $\tau = 0.1$ is $\hat y = 0.44 + 0.68x$.
5. "Prior width controls how far MAP moves" and "Ridge shrinks; $L_1$ can threshold to exactly zero": MLE 2, $s=0.6$: $\tau=2$ ($\lambda=0.125$) gives MAP $\approx1.83$; $\tau=0.5$ ($\lambda=2$) gives $\approx0.82$. With $z=0.6$: ridge MAP $=0.3$; $L_1$ with $\lambda = 1$ gives $\operatorname{sign}(z)\max(|z|-1,0) = 0$.

**Code.** No code on the slides. The linked notebook `notebooks/L01/03_robust_linear_regression.ipynb` (a different synthetic set: 25 points on $1+2x$, outliers at indices 4 and 20 shifted by $\pm5$) fits each model; verbatim:
```python
def fit_line(model_name, target, steps=1200):
    theta = torch.zeros(2, requires_grad=True)  # [intercept, slope]
    optimizer = torch.optim.Adam([theta], lr=0.035)

    for _ in range(steps):
        mean = theta[0] + theta[1] * x
        loss = -observation_model(model_name, mean).log_prob(target).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Figures worth reusing.** Marp-era files, not used by the current deck:
- `lecture1/figures/residual_losses.png` (from `make_figs.py`, also `.svg`): $-\log p(r)$ for Gaussian $r^2$, Laplace $|r|$, Student-$t$, and Uniform barrier lines. The one-glance "noise model picks the loss".
- `lecture1/figures/prior_likelihood_posterior.png`: prior, likelihood (MLE) and posterior (MAP) contours in $(\theta_1,\theta_2)$. PNG only; I could not find a script on `origin/master` that generates it.
- `lecture1/figures/overfit_polynomial.png` (`make_figs.py`): degree-7 fit vs true trend.
- `lecture1/figures/posterior_samples.png` (`make_figs.py`): sampled posterior curves for the posterior predictive.
- Avoid `lecture1/figures/priors_shift_boundary.png`: it is meant to show a class prior moving a decision boundary, but in both panels ("balanced prior", "rare positive (0.1)") the dashed boundary sits at the plot edge, $x\approx7$, so it looks broken. It is also about class priors, not parameter priors.

**The lecture's own summary lines.**
- "Robust fitting keeps every observation, while limiting the gradient contribution of an extreme residual."
- "MAP objective: NLL $+$ negative log-prior. In ML language: data loss $+$ regularization."
- "$\lambda = 1/(2\tau^2)$: smaller $\tau$ → larger $\lambda$ → stronger shrinkage toward $\mathbf{0}$."
- "Regularization trades flexibility for stability; *more is not always better*."
- "Retrieval summary": Gaussian likelihood ⇒ MSE; Categorical likelihood ⇒ cross-entropy; Gaussian prior ⇒ $L_2$; Laplace prior ⇒ $L_1$; MLE + prior ⇒ MAP.
- "Loss ↔ likelihood; regularizer ↔ prior; prediction ↔ plug-in or posterior predictive."
- The standard objective: $\min_{\boldsymbol\theta}\frac{1}{|B|}\sum_{i\in B}-\log p_{\boldsymbol\theta}(y_i\mid\mathbf{x}_i) + \lambda\|\boldsymbol\theta\|_2^2$, with "DL doesn't discard probabilistic ML; it *scales* it" (slide uses a dash there).

**Connections.** Builds on L1's likelihood and L2's Gaussian regression. The $\psi(r)\nabla f$ chain rule previews backpropagation (Lecture 6). Weight decay as an $L_2$ prior, and the over/underfitting picture, return in Lecture 10 (generalization and regularization). The evidence $p(\mathcal{D})$, the posterior that is hard to integrate, and the "variational inference" bullet are exactly where Lectures 23 and 24 pick up ("Bayes for a tiny neural network and its intractable evidence ... VI for the network"); Lecture 25's VAE loss (reconstruction plus KL) comes from that ELBO.

**Three exam-style questions.**
1. *Conceptual.* Two red outliers flatten the MSE line from slope $0.955$ to $0.397$, but Laplace gives $0.906$. Explain using $\psi(r)$. *Answer:* Gaussian $\psi(r) = r/\sigma^2$ grows with the residual, so the two large residuals dominate the gradient; Laplace $\psi = \operatorname{sign}(r)/s_L$ caps each observation's pull at constant magnitude. All twelve points are kept in both fits.
2. *Conceptual.* Why can an $L_1$ penalty give exact zeros while $L_2$ only shrinks? *Answer:* The Laplace negative log-prior $|\theta|/a$ has a corner at 0; its subgradient $[-1,1]$ lets $\theta=0$ be optimal for every $|z|\le\lambda$. The Gaussian prior is smooth, giving $\alpha z$, which is zero only when $z = 0$.
3. *Numeric.* A scalar likelihood is centred at $\hat\theta_\text{MLE} = 2$ with width $s = 0.6$; the prior is $\mathcal{N}(0, 1)$. Find $\lambda$ and $\hat\theta_\text{MAP}$. *Answer:* $\lambda = 1/(2\cdot1^2) = 0.5$; $\alpha = 1/(1 + 0.36) = 0.735$; $\hat\theta_\text{MAP} \approx 1.47$ (between the deck's $1.83$ for $\tau=2$ and $0.82$ for $\tau=0.5$).

---

## Lecture 4: From Linear Models to Neural Networks (neurons, activations, multilayer networks, XOR, and approximation)

**Aim.** Keep the Lecture 1 to 3 losses and make $f_{\boldsymbol\theta}$ expressive: one weighted sum cannot bend, a neuron adds a nonlinearity, a hidden layer of ReLU features solves XOR, and enough hinges can approximate any continuous function on a bounded region.

**Running examples.**
- One weighted sum: $\mathbf{x} = (2,-1)$, $\mathbf{w} = (1.5, 0.5)$, $b = -0.2$, so $z = 2.3$.
- A batch: $\mathbf{X} = \begin{pmatrix}2&-1\\0&3\\-1&1\end{pmatrix}$, $\mathbf{y} = (2.4, 1.0, -1.1)^\top$.
- One neuron: $\mathbf{x}=(2,-1)$, $\mathbf{w}=(1,-1)$, $b=0$, ReLU.
- Three 1-D class scores $z_1 = x$, $z_2 = -x$, $z_3 = 0.5$.
- **The XOR network** (kept throughout): $h_1 = \text{ReLU}(x_1+x_2)$, $h_2 = \text{ReLU}(x_1+x_2-1)$, $z = 2h_1 - 4h_2 - 1$, predict 1 when $z\ge0$; $\mathbf{W}_1 = \begin{pmatrix}1&1\\1&1\end{pmatrix}$, $\mathbf{b}_1 = (0,-1)^\top$, $\mathbf{w}_\text{out} = (2,-4)^\top$, $b_\text{out} = -1$.
- **Tent from three hinges**: $f(x) = \text{ReLU}(x) - 2\,\text{ReLU}(x-1) + \text{ReLU}(x-2)$.
- **Universal approximation**: target $\sin x$ on $D = [-\pi,\pi]$, piecewise-linear interpolants with fixed knots.
- Depth: $h_2(x) = \text{ReLU}(2 - \text{ReLU}(2x+1))$; a 2-D fold $\text{ReLU}(x_1 + 0.65x_2 - 0.35)$.

**Ideas in teaching order.**
- "Lecture 1 chose the loss; Lecture 2 expands the prediction rule": $\mathbf{x}\to f_{\boldsymbol\theta}(\mathbf{x})\to$ output parameters $\to$ same loss.
- One weighted sum, batch form $\mathbf{X}\mathbf{w} + b\mathbf{1}$, bias as a column of ones; "The model can tilt and shift a line or plane, but it cannot bend."
- A neuron has two stages, $z$ then $a = \phi(z)$. Sigmoid and tanh are smooth switches; ReLU and GELU are ramps; saturation means a flat curve and a tiny derivative.
- Without activations, two linear layers collapse into one.
- A binary linear classifier draws one straight boundary; XOR defeats it. Multi-class linear scores give straight-edged regions; softmax preserves the winner; visible boundaries are ties among maximal scores only.
- An MLP hidden layer is a column of learned features; layer shapes and parameter count.
- The XOR network verified on all four inputs, in matrix form, and with probability and loss; the hidden map sends $(1,0)$ and $(0,1)$ to the same point. Four-point XOR differs from filled XOR.
- Universal approximation: a ReLU is a hinge; output weights set slope changes; three hinges make a tent; more hinges shrink the sup error; "expressivity ≠ trainability ≠ generalization".
- Width adds rows, depth adds columns; depth is composition (layer 2 sees $h_1$, not $x$). Then the same forward, backprop, optimize loop.

**Key equations.**
$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1} = \tilde{\mathbf{X}}\boldsymbol\theta,\qquad (n\times(d+1))((d+1)\times1)\to(n\times1)$$
$$z = \mathbf{w}^\top\mathbf{x} + b,\quad a = \phi(z);\qquad \sigma'(z) = \sigma(z)(1-\sigma(z));\qquad \text{GELU}(z)\approx z\,\sigma(1.702z)$$
$$\mathbf{W}_\text{eq} = \mathbf{W}_2\mathbf{W}_1,\qquad \mathbf{b}_\text{eq} = \mathbf{W}_2\mathbf{b}_1 + \mathbf{b}_2$$
$$\mathbf{h} = \phi(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1),\quad \mathbf{z} = \mathbf{W}_2\mathbf{h} + \mathbf{b}_2,\qquad \#\text{params} = m(d+1) + K(m+1)$$
$$\frac{p_i}{p_j} = e^{z_i - z_j},\qquad (\mathbf{w}_i - \mathbf{w}_j)^\top\mathbf{x} + (b_i - b_j) = 0$$
$$g(x) = c + \sum_j v_j\,\text{ReLU}(w_jx + b_j),\qquad \forall\epsilon>0\ \exists\text{ finite MLP } g:\ \sup_{x\in D}|f(x) - g(x)| < \epsilon$$

**Worked numbers worth reusing.**
1. "Calculate every contribution entering the node" and "One matrix multiplication predicts all three rows": $1.5(2) + 0.5(-1) + (-0.2)(1) = 3.0 - 0.5 - 0.2 = 2.3$; batch $\hat{\mathbf{y}} = (2.5, 1.5, -1.0) + (-0.2) = (2.3, 1.3, -1.2)$ vs targets $(2.4, 1.0, -1.1)$.
2. "Calculate the effect of saturation": $\sigma'(0) = 0.25$, $\sigma'(6)\approx0.0025$; for $\Delta z = 0.1$, $\Delta h\approx0.025$ vs $0.00025$, "100× less"; ReLU gives $\Delta h\approx0.1$.
3. "Verify the XOR network on all four inputs": $(0,0)$: $u_1=0,h_1=0,u_2=-1,h_2=0,z=-1\to0$; $(1,0)$ and $(0,1)$: $h=(1,0)$, $z=1\to1$; $(1,1)$: $h=(2,1)$, $z=-1\to0$. "Combine the two features...": at $(1,0)$, $z=1$, $p=\sigma(1)\approx0.731$, BCE for $y=1$ is $-\log 0.731\approx0.313$.
4. "Evaluate five explicit inputs...": scores $x, -x, 0.5$ at $x = -1, -0.5, 0, 0.5, 1$; visible boundaries at $x=\pm0.5$; the class 1/2 tie at $x=0$ is hidden (class 3 wins with $0.5$). "Softmax preserves the winner": $z_i=2, z_j=1$ gives $p_i\approx0.731$, $p_j\approx0.269$.
5. Universal approximation slides: $f(1.57)\approx1.000$, $g(1.57)\approx0.866$, gap $\approx0.134$; with more hinges the worst gap is $\approx0.033 < \epsilon = 0.05$; "More hinges reduce the largest approximation error": 2, 5, 15 hinges give $\epsilon_m\approx0.437, 0.134, 0.019$. MLP 3-5-2: $\mathbf{W}_1$ has $3\times5 = 15$ edges, $\mathbf{W}_2$ has $5\times2 = 10$.

**Code.** No code on the slides; the deck links two Colab notebooks. Verbatim from `notebooks/L02/02_mlp_xor.ipynb`:
```python
def two_relu_logit(X):
    s = X[:, 0] + X[:, 1]
    hidden = np.column_stack([relu(s), relu(s - 1)])
    z = 2 * hidden[:, 0] - 4 * hidden[:, 1] - 1
    return hidden, z
```
The same notebook asserts truth-table agreement $1.0$ on the four corners and $0.75$ on filled-square XOR. From `notebooks/L02/01_linear_regression_gd.ipynb` (data $y = 2x + 1 + \text{noise}$, $n=100$), verbatim:
```python
for step in range(300):
    yhat = w*x + b
    resid = y - yhat
    loss = np.mean(resid**2)
    losses.append(loss)
    gw = -2*np.mean(resid*x)
    gb = -2*np.mean(resid)
    w -= lr*gw
    b -= lr*gb
```

**Figures worth reusing.** All generated by `lecture2/diagrams/l2_figs.py` (`.svg` + `.png`, plus `.pdf` where noted); the first five are used in the deck:
- `lecture2/figures/xor.png` (also `.pdf`): the four XOR corners, two candidate lines, "no single line separates them". Best one-glance figure.
- `lecture2/figures/xor_scope_contrast.png` (`.pdf`): left, the two-ReLU rule correct on four Boolean inputs (class 1 inside the diagonal band); right, the same rule on filled XOR with misclassified areas hatched.
- `lecture2/figures/softmax_regions.png` (`.pdf`): three linear scores carve the plane into polygonal regions.
- `lecture2/figures/relu_2d_fold.png` (`.pdf`): one ReLU on the plane, hinge line, and the 3-D folded surface.
- `lecture2/figures/relu_multiclass_map.png` (`.pdf`): piecewise-linear class boundaries from ReLU features, with hidden-unit hinges dashed.
- `lecture2/figures/relu_build.png` (not in the deck): target $f(x)$ and a sum of ReLUs overlaid, with the individual ramps faint. Good for universal approximation. (Note: `lecture2/figures/feature_transform.png` is also unused and shows a different hidden map from the deck's XOR network, so avoid it.) `lecture2/figures/depth-feature-hierarchy-vision.png` is used in the deck and labelled "CONCEPTUAL SCHEMATIC"; it is not made by a script I found.

**The lecture's own summary lines.**
- "The model can tilt and shift a line or plane, but it cannot bend."
- "Without a $\phi$ node, the middle column adds parameters but no new bend or feature shape."
- "XOR requires a nonlinear feature transformation before a final straight separator can work."
- "Softmax changes score scale, not score order: $\arg\max_k p_k = \arg\max_k z_k$."
- "Predict $1$ when $z\ge0$. One ordinary output node correctly classifies all four XOR inputs."
- "Exact on the *four Boolean inputs* is not exact on filled regions."
- "expressivity ≠ trainability ≠ generalization"
- S04 banners: "The loss says what counts as a good prediction. The network says which predictions are possible." and "Affine map = select and shift a direction. Activation = introduce a bend. Composition = build a representation."
- Closing slide: "We now have $\mathbf{x}\to f_{\boldsymbol\theta}(\mathbf{x})\to\mathcal{L}$. Next: how does the loss send credit backward through that computation?"

**Connections.** Reuses Lectures 1 to 3 unchanged as the output head and loss (sigmoid + BCE on the XOR output; softmax for $K$ classes). Saturation ($\sigma'(6)\approx0.0025$) sets up vanishing gradients in backprop (Lecture 6) and optimization (Lectures 5, 7); the dense layer $\mathbf{W}\mathbf{x}+\mathbf{b}$ is the object of Lecture 8 (affine autograd, batched VJPs). The hidden-layer MLP is the model of Lecture 11 ("a hidden-layer MLP" in the schedule) and the feed-forward part of the Transformer blocks in Lectures 15 and 16. GELU is introduced here; I did not check whether later decks use it.

**Three exam-style questions.**
1. *Conceptual.* A colleague stacks five linear layers with no activations to "make the model deeper". What can it represent? *Answer:* one affine map, $\mathbf{W}_\text{eq} = \mathbf{W}_5\cdots\mathbf{W}_1$ and a single bias; the boundary for binary classification is still one hyperplane, so XOR stays unsolved.
2. *Conceptual.* The universal approximation theorem says a one-hidden-layer network can approximate $\sin x$ on $[-\pi,\pi]$ to any $\epsilon$. Name three things it does not promise. *Answer:* that gradient training will find that network, that finite data suffices, that the network is small, or that it generalizes outside $D$ (any three).
3. *Numeric.* For the deck's XOR network, compute $u_1, h_1, u_2, h_2, z$, $p = \sigma(z)$ and the BCE at $\mathbf{x} = (1,1)$ with $y = 0$; then count its parameters. *Answer:* $u_1 = 2$, $h_1 = 2$, $u_2 = 1$, $h_2 = 1$, $z = 4 - 4 - 1 = -1$, $\hat y = 0$; $p = \sigma(-1)\approx0.269$; BCE $= -\log(1-0.269) = -\log 0.731\approx0.313$. Parameters with $d=2$, $m=2$, $K=1$: $2(3) + 1(3) = 9$.

---

## Module in one paragraph

The Foundations module answers two questions that every later lecture assumes: where the loss comes from, and why the model is a neural network. Lectures 1 to 3 treat a network as a machine that outputs the parameters of a distribution $p_{\boldsymbol\theta}(y\mid\mathbf{x})$; the observed data score candidate parameters through the likelihood, i.i.d. turns it into a product, and the log turns it into a sum, so training is minimizing the average negative log-likelihood. The choice of observation model then fixes the loss (Gaussian gives MSE and least squares, Bernoulli with sigmoid gives BCE, categorical with softmax gives cross-entropy, Laplace and Student-$t$ give robust fits that limit an outlier's gradient), and a prior on the parameters fixes the regularizer (Gaussian prior gives $L_2$ with $\lambda = 1/(2\tau^2)$, Laplace prior gives $L_1$ with exact zeros), so the standard objective is NLL plus negative log-prior, a MAP estimate. Lecture 4 keeps those losses and changes only $f_{\boldsymbol\theta}$: a weighted sum cannot bend, stacked linear layers collapse, but affine maps separated by a nonlinearity build features, as the two-ReLU XOR network with output $z = 2h_1 - 4h_2 - 1$ shows, and sums of ReLU hinges can approximate any continuous function on a bounded region, which is a statement about capacity, not about training or generalization.

## The three things a student must remember

1. **A loss is a negative log-likelihood.** $\ell_i(\boldsymbol\theta) = -\log p_{\boldsymbol\theta}(y_i\mid\mathbf{x}_i)$: Gaussian gives MSE, Bernoulli gives BCE, categorical gives cross-entropy, and the noise model decides how hard large errors are punished.
2. **A regularizer is a negative log-prior.** MAP $=$ NLL $+$ $(-\log p(\boldsymbol\theta))$: a Gaussian prior gives $L_2$ ($\lambda = 1/(2\tau^2)$), a Laplace prior gives $L_1$.
3. **Depth needs nonlinearity.** Without $\phi$, layers collapse to one affine map; with ReLU, hidden units are learned features (XOR) and hinges that can approximate any continuous function on a bounded region, which does not guarantee that training finds it.
