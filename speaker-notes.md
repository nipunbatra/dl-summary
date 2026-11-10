# The course in one lecture: speaker notes

Planned main route: 90 minutes, including student responses. Optional slides are outside this budget.

## Pacing

| Part | Minutes |
|---|---:|
| Lecture map | 7 |
| Part 1 · Foundations | 11 |
| Part 2 · Gradients and optimization | 12 |
| Part 3 · Generalization | 9 |
| Part 4 · Attention and language | 12 |
| Part 5 · Vision and language | 13 |
| Part 6 · Generative modelling | 12 |
| Part 7 · Object detection | 8 |
| The whole course | 6 |

## title: The course in one lecture

**Build 1:** Every tile is one lecture and opens its slides. The seven rows are the seven parts of today.

- Ask: Which lecture do you remember best, and what did its model output?
- Expected: Any lecture works. Write two or three of the outputs the class names on the board (a class, the next token, a box); the next slide collects all of them.

**Point here:** The whole course fits on one page, and it is one method applied seven times.

**Transition:** First, the models themselves.

**Source:** Course page: nipunbatra.github.io/dl-2026/schedule.html

## ten-models: Ten models, built in twenty-six lectures

**Build 1:** The models we built, in order, each with what it returns. What do all ten have in common?

- Ask: What do these models have in common?
- Expected: They are all neural networks trained by gradient descent on a loss. Push for the sharper answer: every loss here is a negative log probability.

**Build 2:** So one recipe trained them all. The next slides run it once, on the smallest network we built.

- Ask: Squared error does not look like a negative log probability. Where does it fit?
- Expected: Squared error is the negative log likelihood of a Gaussian with a fixed variance (Lectures 1 and 2), so it is the same loss with a different output distribution.

**Point here:** What a model returns changes from lecture to lecture; how it is trained does not.

**Common misconception:** That each new architecture came with a new learning algorithm. Only the pieces plugged into the recipe changed.

**Transition:** Here is the recipe, one step per click.

## recipe: The recipe, on one example

**Build 1:** One training example from Lecture 4's XOR problem: the input x = (1, 0) and its target y = 1.

- Ask: What are the two things a training example gives us?
- Expected: An input x and the answer y we want for it.

**Build 2:** The model is a function with 9 parameters θ. Its two ReLU units give h = (1, 0), and the last layer turns them into a score, z = 1.00.

- Ask: Which of these numbers can training change?
- Expected: Only the parameters θ. The data and the form of the model stay fixed.

**Build 3:** A sigmoid turns the score into a probability. The network says y = 1 with p = 0.73.

- Ask: Why a sigmoid, and what would we use for ten classes?
- Expected: The output has to be a valid probability. A sigmoid gives one number between 0 and 1; a softmax does the same for many classes (Lecture 9).

**Build 4:** The loss is the negative log of the probability the network gave to the right answer: −ln 0.73 = 0.31.

- Ask: What would the loss be if the network had said p = 1? And p = 0.5?
- Expected: Zero for p = 1, and ln 2 = 0.69 for p = 0.5. The loss grows as the network gets less sure of the right answer.

**Build 5:** Backprop gives the gradient g of the loss for all 9 parameters. One step of size 0.1 against it moves p from 0.73 to 0.78 and the loss to 0.25.

- Ask: Which way does the last bias move, and why?
- Expected: Up, from −1 to −0.97: the gradient on the score is p − y = −0.27, and the step goes against the gradient, which raises z and so raises p.

**Build 6:** Training repeats that step. At test time the loop is gone: the trained network takes an input and returns a probability.

- Ask: What is missing from the bottom row that the top row has?
- Expected: The target, the loss and the update. At test time nobody tells the network the answer.

**Point here:** Data, model, output, loss and update form a loop; at test time only the first three run.

**Common misconception:** That the loss is used at test time. It only exists where we have the right answer.

**Transition:** One step barely moved the loss. What happens when the step is repeated?

**Source:** The network is the one Lecture 4 built by hand for XOR. Numbers: scripts/sci/p0.py.

## recipe-loop: The recipe, repeated

**Build 1:** The same step, now on all four XOR points, 300 times. Drag the number of steps and switch the step size η. Which of the three learns XOR?

- Ask: Before dragging: the four predictions start at 0.27, 0.73, 0.73, 0.27. Which of them are already on the right side of 0.5?
- Expected: All four are, but none is confident: the hand-built network of Lecture 4 separates XOR, and training makes it surer.

**Build 2:** The loop is the same in all three runs; only the step size differs. Part 2 comes back to that choice.

- Ask: Why does the large step end at exactly 0.69?
- Expected: 0.69 is ln 2, the loss of a network that says 0.5 for every input. After step 2 the second ReLU unit outputs 0 on all four inputs and gets no gradient again. One unit cannot draw XOR, so the best the network can do is say 0.5 everywhere.

**Point here:** Repeating one small step is all that training is, and the step size can make or break it.

**Common misconception:** That a larger step only gets there faster. Here it switches a unit off for good on step 2.

**Notebook:** Set η = 0.1 and drag slowly from 0: the loss on step 1 is 0.324, a little above the start (0.313), before it falls. Keep that for the question at the end of Part 2.

**Transition:** Six lines of PyTorch run this loop for every model in the course.

**Source:** The network is the one Lecture 4 built by hand for XOR. Numbers: scripts/sci/p0.py.

## recipe-card: The recipe as a card, and as code

**Build 1:** The card has one row per step of the flowchart. Beside it is the loop that ran in every lecture. The loader hands over a batch of pairs (x, y).

- Ask: Which row of the card has no line in this loop?
- Expected: At test time: it runs after the loop, with no loss and no update.

**Build 2:** The model turns the inputs into scores, one per class.

- Ask: What is the shape of logits for a batch of 32 images and 10 classes?
- Expected: 32 by 10: one score per class for every example.

**Build 3:** cross_entropy applies the softmax and takes −ln of the probability of the right class, so this one line is two rows of the card.

- Ask: Why does PyTorch not want probabilities here?
- Expected: Working from the scores is numerically safer than taking the log of a softmax (Lecture 9: logits of 1000 overflow).

**Build 4:** PyTorch adds new gradients to the old ones, so the loop clears them first.

- Ask: What happens if we forget this line?
- Expected: The gradients of every batch pile up, and the steps grow in a direction that mixes old batches with the new one.

**Build 5:** backward() is backprop. It fills in the gradient of the loss for every parameter.

- Ask: Which lecture built this line from scratch?
- Expected: Lecture 6, with a small autograd engine; Lecture 8 extended it to vectors and matrices.

**Build 6:** step() applies the update. With SGD or with Adam, this line stays the same.

- Ask: Where does the step size live?
- Expected: Inside the optimizer, set when it was created: opt = torch.optim.SGD(model.parameters(), lr=0.1).

**Point here:** The six rows of the card are the six things to ask about any model.

**Common misconception:** That this loop is specific to classifiers. Only the model and the loss line change for the other models.

**Transition:** With the card in hand, here is the route through the course.

## route: The route for today

**Build 1:** Each part changed some rows of the card. We take the parts in order and ask of each: which rows, and how?

- Ask: Which part changed the Update row?
- Expected: Part 2. It is the only part that changes nothing else: backprop and the optimizers work the same for every model that follows.

**Point here:** Seven parts, each read as a change to the same six rows.

**Transition:** Part 1: where the loss comes from.

## part-1: Part 1 · Where the loss comes from

**Build 1:** By the end of this part you can say where cross-entropy and squared error come from.

- Ask: Where did cross-entropy come from? Was it chosen because it works well?
- Expected: It was not picked from a list. It is the negative log-likelihood of a categorical distribution, and every other loss of the course is read off a distribution the same way.

**Transition:** Start with the smallest likelihood we wrote: ten tosses of a coin.

## nll-coin: The loss is a negative log-likelihood

**Build 1:** Lecture 1’s ten tosses: four heads, six tails. Drag θ, the coin’s chance of heads. Which θ would you bet on: 0.4, 0.7, or another?

- Ask: Which θ would you bet on, and why?
- Expected: 0.4, because four of the ten tosses were heads. Hold that guess; the next two clicks justify it.

**Build 2:** The likelihood scores each θ by how probable it makes these tosses. At 0.4 they are 6.82 times as likely as at 0.7.

- Ask: Why do we multiply the ten probabilities?
- Expected: The tosses are independent, so the probability of all ten is the product of the ten probabilities.

**Build 3:** Take the negative log. The product becomes a sum and the best θ becomes the lowest point: 6.73 nats at θ = 0.4.

- Ask: Why take the log at all, if the best θ does not move?
- Expected: The log is increasing, so the best θ stays put. A sum of logs does not underflow the way a product of many small numbers does, and its gradient is a sum over examples, which is what minibatches use.

**Point here:** Training a model is finding the θ under which the data are most probable, written as a loss: the negative log-likelihood.

**Common misconception:** That the likelihood is a distribution over θ. It is a score for each θ; its values need not sum to 1 over θ.

**Notebook:** Leave θ at 0.7 at click 1 and let the class bet. At click 2 drag to 0.4; at click 3 drag to either end to show how fast the loss grows.

**Transition:** A network does the same thing, one example at a time.

**Source:** The tosses HHTTTHHTTT are Lecture 1’s. Numbers: scripts/sci/p1.py.

## nll-loss: Cross-entropy is the negative log of the right class

**Build 1:** Two networks. One says p = 0.8 for a label y = 1. The other gives the logits shown, and class 1 is correct. What loss does each pay?

- Ask: What loss does each network pay?
- Expected: Each pays −ln of the probability it gave the right answer. Let them compute the first one.

**Build 2:** The Bernoulli’s negative log is binary cross-entropy: −ln 0.8 = 0.22. Had it said 0.2, it would pay 1.61.

- Ask: Why is a confident mistake so expensive?
- Expected: The negative log grows without bound as p goes to 0: p = 0.2 already costs 1.61.

**Build 3:** Softmax turns the logits into probabilities. Class 1 gets 0.69, so the loss is −ln 0.69 = 0.37. That is cross-entropy.

- Ask: How large can this loss get, and how small?
- Expected: Zero when the right class gets probability 1; no upper bound as it goes to 0. Ten equal guesses would give ln 10.

**Build 4:** Its gradient at the logits is p − y. A step against it raises the right class’s logit and lowers the others by what they took.

- Ask: Why is p − y a good learning signal?
- Expected: It is zero once p matches y, and each logit moves by how wrong its probability is. The same p − y comes back in every classifier of the course.

**Point here:** Binary and multi-class cross-entropy are the negative log-likelihoods of a Bernoulli and a categorical output.

**Common misconception:** That cross-entropy needs a separate log-softmax step in code. F.cross_entropy takes the logits and does both, which is also numerically safer (Lecture 9).

**Transition:** Line the cases up: each kind of target has its own distribution.

**Source:** The logits (2, 1, −0.5) are from Lecture 2’s softmax chart. Numbers: scripts/sci/p1.py.

## dist-loss: Each kind of target picks a distribution, and a loss

**Build 1:** The last two slides in one table: each kind of target gets a distribution, and its negative log is the loss. What goes in the last row?

- Ask: For a real-valued target, which distribution, which output layer, which loss?
- Expected: A Gaussian; no squashing; squared error.

**Build 2:** A real number gets a Gaussian with a fixed σ. The network outputs its mean, with nothing to squash it.

- Ask: Why no sigmoid or softmax on the last row?
- Expected: A mean can be any real number, so the output layer leaves the score as it is.

**Build 3:** Its negative log is the squared error over 2σ², plus a constant that never moves the fit. With r = 2 and σ = 1: 2 plus 0.92.

- Ask: So what does training with squared error assume about the noise?
- Expected: Gaussian noise with one fixed σ for every input. Outliers then pull hard on the fit; a Laplace noise model gives absolute error instead (Lecture 3).

**Point here:** Choosing the output distribution chooses the output layer and the loss together.

**Common misconception:** That squared error is a neutral default. It is the Gaussian’s negative log-likelihood, and it brings the Gaussian’s light tails with it.

**Transition:** The data gave us the loss. What if we also have a belief about the weights?

**Source:** Rows follow the retrieval summary of Lectures 2 and 3; the Gaussian example is from the S03 cheat sheet. Numbers: scripts/sci/p1.py.

## map-prior: A Gaussian prior is an L2 penalty

**Build 1:** Four points from 2 + 3x, noise σ = 0.5. Maximum likelihood gives the dashed line. With a prior θ ~ N(0, 0.5²), which way does it move?

- Ask: Which way does the line move, and which of b and w moves more?
- Expected: Both shrink toward 0. Ask for the reason before the click: the prior says weights near 0 are more probable.

**Build 2:** Toward zero. The prior adds its negative log to the loss, and at τ = 0.5 the fit is (1.79, 2.62). Drag τ.

- Ask: What does a small τ say about the weights, and what does it do to the fit?
- Expected: It says the weights are almost surely near 0. At τ = 0.05 the line is nearly flat, (0.13, 0.20): the prior outweighs four points.

**Build 3:** A narrow prior, τ = 0.1, pulls to (0.44, 0.68); a wide one gives maximum likelihood back.

- Ask: Where have you met this penalty under another name?
- Expected: Ridge regression and weight decay. A Laplace prior gives the L1 penalty instead, which can set weights exactly to zero.

**Point here:** The regularizer is the negative log of a prior: MAP is the negative log-likelihood plus the negative log-prior.

**Common misconception:** That regularization is a trick bolted onto the loss. It is the prior written in the same units as the loss.

**Notebook:** Click 2: drag τ from 10 down to 0.05 and watch the purple line flatten while λ grows.

**Transition:** The losses are settled. Now the model itself: one weighted sum is not enough.

**Source:** The four points and σ = 0.5 are Lecture 3’s MAP example. Numbers: scripts/sci/p1.py.

## mlp-xor: Two ReLU units solve XOR

**Build 1:** Lecture 4’s XOR. The two filled points have y = 1. Can one straight line put them on one side and the hollow points on the other?

- Ask: Can one straight line separate them?
- Expected: No. Any line that puts (0, 1) and (1, 0) on one side puts (0, 0) or (1, 1) there too.

**Build 2:** No line can. Two ReLU units move the points: (1, 0) and (0, 1) land on the same spot, h = (1, 0).

- Ask: Why does stacking two linear layers without the ReLU not help?
- Expected: Two linear layers multiply into one linear map, which is still one straight boundary. The ReLU is what bends.

**Build 3:** In h, one line does it: z is positive only for the filled points. This is the network the opening trained.

- Ask: Which parameters did we set by hand here, and which would training find?
- Expected: All nine were set by hand in Lecture 4. Training would find some network that works, not necessarily this one.

**Point here:** A hidden layer of ReLUs moves the data until one straight line is enough.

**Common misconception:** That more linear layers add power. Without a nonlinearity they collapse into one.

**Transition:** Two units handle four points. How far does this go?

**Source:** The XOR network is the one Lecture 4 built by hand. Numbers: scripts/sci/p1.py.

## p1-hinges: Enough hinges fit any curve

**Build 1:** Each ReLU is a hinge, and a sum of hinges is a bent line. Here it is fitted to sin x. Add hinges: what happens to the largest gap?

- Ask: What happens to the largest gap as hinges are added?
- Expected: It shrinks, roughly with the square of the hinge spacing. Under 0.05 needs nine hinges (gap 0.049).

**Build 2:** It shrinks: 0.437, 0.134, 0.019 for 2, 5, 15 hinges. Enough hinges fit any continuous curve on a bounded range; training still has to find them.

- Ask: What does this result not promise?
- Expected: It does not say that gradient descent will find those weights, that a finite data set is enough to pin them down, or that the fit is any good outside the range.

**Point here:** A one-hidden-layer ReLU network can approximate any continuous function on a bounded range, given enough units.

**Common misconception:** That this theorem explains why deep networks train well. It is a statement about capacity only.

**Notebook:** Click the three buttons in order; point at the red bar, the largest gap, each time.

**Transition:** Time to check the whole part with one number.

**Source:** Fits of sin x on [−π, π] with equally spaced hinges, as in Lecture 4. Numbers: scripts/sci/p1.py.

## check-1: Check: a loss of 2.30 at the first step

**Build 1:** Two ideas of this part meet here. Take a minute with your neighbour.

- Ask: Is something broken?
- Expected: No: −ln of the right class’s probability, with ten equal guesses, is ln 10 = 2.30.

**Build 2:** ln 10 is the loss of ten equal guesses. A first loss far above it would mean the starting logits are too large.

- Ask: What first loss would worry you?
- Expected: One much larger than 2.30: the network is confidently wrong before it has learned anything, usually from badly scaled initial weights. One much smaller would hint at a bug or a leak of labels.

**Point here:** Cross-entropy at the start of training should be close to ln K; that is a free sanity check.

**Transition:** The whole part on one card.

## recap-1: Part 1 in one slide

**Build 1:** Next: how the gradient of that loss is computed, for every weight at once.

- Ask: Which rows of the card did this part fill in?
- Expected: The model (an MLP), the output (softmax probabilities) and the loss (−ln of the right class). The update row is what Part 2 is about.

**Point here:** Part 1 fixed the model, the output and the loss of the recipe.

**Transition:** Part 2: how the weights move.

## part-2: Part 2 · How the weights move

**Build 1:** By the end you can run backprop by hand and say what each optimizer adds to the plain step.

- Ask: In θ ← θ − η ∇loss, which pieces do we get to choose?
- Expected: How the gradient is measured (on how much data) and the step: its size η, how it changes over time, and whether it uses memory of past gradients. Backprop itself is fixed: it computes the exact gradient of whatever loss we wrote.

**Transition:** First, where the gradient comes from.

## bp-graph: Backprop, one node at a time

**Build 1:** Lecture 6’s graph for one prediction, ŷ = wx + b. The forward pass works left to right and stores every value on the way.

- Ask: Which values does the backward pass need from the forward pass?
- Expected: The stored values: x for the multiply node, ŷ and y for the square. That is why the forward pass keeps them.

**Build 2:** The target is y = 10. Should w go up or down to lower the loss?

- Ask: Before computing anything: up or down for w?
- Expected: Up. The prediction 7 is below the target 10, and x is positive, so a larger w raises ŷ toward y.

**Build 3:** Backward starts at the loss and walks right to left. The square node passes back its local slope.

- Ask: Why do we start backward at the loss and not at w?
- Expected: The loss is the one scalar we want to reduce; one sweep from it gives the gradient for every input at once.

**Build 4:** An add node copies the gradient that arrives to both of its inputs.

- Ask: What does an add node do to the gradient?
- Expected: Its local slope is 1 for both inputs, so it hands the same number to each.

**Build 5:** A multiply node scales it by the other input. The gradient on w is negative, so w should go up.

- Ask: What would g_x be, and why do we not use it?
- Expected: −6 × w = −12. x is data, not a parameter, so nothing updates it.

**Point here:** Every node multiplies the gradient that arrives by its own local slope; one backward sweep gives every gradient.

**Common misconception:** That the gradient says how far to move. It gives a direction and a rate; the step size sets the distance.

**Transition:** Now take one step with these two gradients.

**Source:** Example: Lecture 6, the central graph. Numbers: scripts/sci/p2.py.

## bp-update: One step, in PyTorch

**Build 1:** The same graph in PyTorch. The forward line builds the graph and stores its values; the target is 10.

- Ask: What does requires_grad=True ask PyTorch to do?
- Expected: To record every operation on that tensor, so backward() can replay the graph.

**Build 2:** backward() runs the sweep from the last slide and leaves each gradient in .grad.

- Ask: Why would a training loop call zero_grad() before this line?
- Expected: backward() adds into .grad. Without clearing it, the gradients of earlier steps pile up.

**Build 3:** The step moves each parameter against its gradient, scaled by the step size η = 0.01.

- Ask: Why is the update inside torch.no_grad()?
- Expected: The update is not part of the model. Recording it would add the step itself to the graph.

**Build 4:** Run the forward pass again with the new values. The prediction moved toward the target.

- Ask: Would a step ten times larger, η = 0.1, do better?
- Expected: No. It moves w to 3.80 and b to 1.60, so ŷ jumps past the target to 13.00 and the loss is 9.00 again.

**Point here:** One step is: forward, backward, then move every parameter a little against its gradient.

**Common misconception:** That a larger step always learns faster. Here ten times the step overshoots and the loss does not fall at all.

**Transition:** This gradient used one example. With many examples, how much data should measure it?

**Source:** Example: Lecture 6. Numbers: scripts/sci/p2.py.

## sgd: A gradient from part of the data

**Build 1:** Lecture 5’s line fit, one step of size 0.1 from (b, w) = (2, −1). With all four points the loss falls. Which single point would you rather not draw?

- Ask: Which point pulls hardest, and which is the unlucky one?
- Expected: Point 4 pulls hardest (gradient (−6, −12)). Point 2 is the unlucky one: it is nearly fitted already, so its step barely helps the rest.

**Build 2:** Click point 2: its own loss falls, but the loss on all four rises. The grey arrows are the other one-point steps. Where is their average?

- Ask: How can a correct gradient of point 2 raise the total loss, and where is the average of the four arrows?
- Expected: It is the exact gradient of point 2’s loss, not of the total. Lowering b helps points 1 and 2 but hurts points 3 and 4, which sit far above the line. The four grey and purple arrows average to the full-batch step.

**Build 3:** One draw can go uphill while the average goes downhill. A larger batch shrinks the spread.

- Ask: Why not always use all the data, then?
- Expected: One full gradient costs a pass over all the data. With millions of examples, a mini-batch of a few hundred gives a good enough direction many times more cheaply.

**Point here:** A mini-batch gives a cheap estimate of the gradient that is right on average, so single steps can go uphill.

**Common misconception:** That SGD is wrong whenever the loss goes up on a step. One draw follows one point’s surface; only the average follows the full loss.

**Notebook:** Start on “all four points”, then click point 2 and point 4 in turn. Read the loss on all four points each time.

**Transition:** Even an exact gradient can make slow progress. Lecture 7’s ravine shows why.

**Source:** Example A of Lecture 5. Numbers: scripts/sci/p2.py.

## opt-play: Same ravine, four optimizers

**Build 1:** Lecture 7’s ravine is 100 times steeper across θ₂ than along θ₁. Each method took 55 steps from the same start. Why does gradient descent zigzag?

- Ask: One step size has to serve both directions. What goes wrong?
- Expected: A step small enough to stay stable across the steep θ₂ is tiny along the flat θ₁, so gradient descent crosses the valley again and again and creeps along it.

**Build 2:** With a memory of past gradients, all three beat the plain step. Each uses its own step size from Lecture 7.

- Ask: What does each method remember between steps?
- Expected: Momentum keeps a running average of the gradients (direction). RMSProp keeps a running average of their squares (the size of each coordinate). Adam keeps both, with a correction for starting at zero.

**Point here:** Momentum remembers direction, RMSProp remembers scale, and Adam remembers both.

**Common misconception:** That Adam needs no tuning. Each method here has its own step size from the lecture; with the same η they would rank differently.

**Notebook:** The figure opens at 55 steps. Pick gradient descent, drag back to 0 and forward again to watch it cross the valley; then do the same for momentum. The losses on the right: gradient descent 0.46, momentum 0.33, Adam 0.06, RMSProp 0.00.

**Transition:** Backprop on scalars gave these gradients. A real layer works on vectors.

**Source:** Example B of Lectures 5 and 7, with each method’s step size from Lecture 7. Numbers: scripts/sci/p2.py.

## vjp: Backprop through a layer multiplies by Jᵀ

**Build 1:** Lecture 8’s layer. The slopes on the arrows are the entries of the Jacobian J. If the gradient (3, 4) arrives at z, what reaches x?

- Ask: Which arrows carry gradient back to x₁?
- Expected: The two arrows leaving x₁: to z₁ with slope 1 and to z₂ with slope 2. x₁ was used twice, so both contributions add (the branch rule of Lecture 6).

**Build 2:** Each input adds up what every output sends back along its arrows. Written as matrices, that sum is Jᵀ times the gradient.

- Ask: Why the transpose?
- Expected: J has a row per output and a column per input. Going backward we sum over outputs for each input, which is a column of J, so a row of Jᵀ.

**Build 3:** Autograd computes Jᵀg directly and never builds J. For a layer of 4,096 units, J would have about 1.7 × 10⁷ entries.

- Ask: What does z.backward(g) need g for?
- Expected: z is a vector, so backward needs the gradient arriving at z (the seed). For a scalar loss, PyTorch seeds it with 1.

**Point here:** For a layer, backprop is a vector-Jacobian product: the gradient at the output times Jᵀ.

**Common misconception:** That autograd forms Jacobians. It only ever computes products Jᵀg, which cost about as much as the forward pass.

**Transition:** Backprop and the optimizer need one more thing: outputs that do not overflow.

**Source:** Example: Lecture 8 (the autograd deck). Numbers: scripts/sci/p2.py.

## logits: Softmax overflows; the loss takes logits

**Build 1:** Softmax turns logits into probabilities. For (2, 1, 0) it is easy. What should (1000, 999, 998) give?

- Ask: Before computing: is softmax(1000, 999, 998) different from softmax(2, 1, 0)?
- Expected: No. Only differences between logits matter, and both have gaps of 1.

**Build 2:** Computed directly in float32, e to the 1000 overflows to inf, and inf divided by inf is nan.

- Ask: Where exactly does it break?
- Expected: float32 holds numbers up to about e to the 88.7; e to the 1000 becomes inf, and inf / inf is nan.

**Build 3:** Subtracting the same number from every logit leaves softmax unchanged. Subtract the largest: the same three probabilities, with nothing to overflow.

- Ask: So what should a model output, and where should the softmax go?
- Expected: The raw logits. F.cross_entropy applies a log-softmax that subtracts the max for us; here it gives −ln 0.665 = 0.408 for class 1.

**Point here:** Keep the model’s output as logits and let the loss apply the softmax in a stable way.

**Common misconception:** That adding softmax as the last layer and then calling cross_entropy is fine. It applies softmax twice and the gradients shrink.

**Transition:** The last knob on the step is its size over time.

**Source:** Example: Lecture 9 (valid outputs). Computed in float32 by scripts/sci/p2.py.

## schedule: Large steps early, small steps late

**Build 1:** Lecture 9’s cosine schedule over T = 100 steps. Drag the step: halfway, η is 0.0505. Why would we want the step to shrink at all?

- Ask: What goes wrong with a constant large η late in training?
- Expected: The loss stops falling and bounces: near the minimum the gradient is mostly mini-batch noise, and a large step turns that noise into large moves.

**Build 2:** Large steps cover ground early; small steps settle late. The schedule changes only η, so it works with any optimizer.

- Ask: When would you add warmup in front of this curve?
- Expected: When the first steps are fragile, for example Adam on a large transformer: start η near zero and raise it over the first few hundred steps, then decay.

**Point here:** The optimizer chooses the direction of each step; the schedule chooses its size over time.

**Common misconception:** That a decaying η means the model learns less late in training. It means the noisy part of each step shrinks, so the parameters settle.

**Notebook:** Drag from 0 to 100: the curve is flat at both ends and steepest in the middle. Stop at 50 to read 0.0505.

**Transition:** Time to check the part with one question.

**Source:** Example: Lecture 9 (learning-rate schedules). Numbers: scripts/sci/p2.py.

## check-2: Check: the loss went up

**Build 1:** Two ideas from this part answer it. Talk to your neighbour for a minute.

- Ask: Is the code broken?
- Expected: Not necessarily: a one-point gradient is an unbiased but noisy estimate of the full gradient. The full-batch step from the same start lowers the loss to 5.22.

**Build 2:** One noisy step is not a trend.

- Ask: What would you print to tell noise from a real problem?
- Expected: The loss on all the training data (or a running average of the batch losses) every few hundred steps. If that trend rises, lower η or check the code; one batch is too noisy to judge.

**Point here:** A mini-batch step can go uphill; the trend over many steps is what to watch.

**Common misconception:** That every SGD step must lower the loss.

**Transition:** Here is the part on the recipe card.

**Source:** Example A of Lecture 5. Numbers: scripts/sci/p2.py.

## recap-2: Part 2 in one slide

**Build 1:** Next: Part 3, where a low training loss turns out to be the wrong thing to celebrate.

- Ask: Which row of the card did this part change, and which rows did it leave alone?
- Expected: Only Update. The data, model, output, loss and test-time rows are the same as in Part 1; the logits slide only moved where the softmax is applied.

**Point here:** The Update row is the same for every later model: backprop on a mini-batch, an optimizer and a schedule.

**Transition:** Part 3: generalization.

## part-3: Part 3 · Doing well on data you have not seen

**Build 1:** By the end you can say why training error is the wrong judge, and which split chooses the strength of a regulariser.

- Ask: Your network reaches a training loss of nearly zero. Is it a good model?
- Expected: We cannot tell from the training loss. It has to be measured on data the network did not train on.

**Transition:** Start with the smallest model that shows the problem: a polynomial through ten points.

## overfit: Ten points, ten weights

**Build 1:** Ten points from sin(πx) plus noise of size 0.25, and nine validation points at the midpoints. A degree 9 polynomial has ten weights. What will its training error be?

- Ask: Predict the training error and the validation error of the degree ${P.degree} fit.
- Expected: Training error zero (ten equations, ten unknowns); the validation error large.

**Build 2:** With ten weights and ten points it passes through every training point, so the training error is 0.00.

- Ask: Is a training error of zero good news?
- Expected: No. The fit has matched the noise as well as the curve; the training error cannot tell the two apart.

**Build 3:** Between the points it swings far from sin(πx): the validation error is 1.70, about 27 times the noise variance 0.0625.

- Ask: Where does the large error come from?
- Expected: From the swings between the points and near the ends. The weights have to be huge to make them: ‖w‖ = 465.

**Point here:** A model that can match every training point can be very wrong between them; only held-out data shows it.

**Common misconception:** That a lower training error always means a better model.

**Transition:** The large weights point to the fix: make large weights cost something.

**Source:** The ten-point lab of Lecture 10 (seed 21), ported from its deck. Numbers: scripts/sci/p3.py.

## reg-play: Make large weights cost something

**Build 1:** The loss gets a second term that grows with the weights. Drag λ from 0 upwards. Which column of the table should choose λ?

- Ask: Which column of the table may choose λ, and why not the others?
- Expected: Validation. Training error only rises as λ grows, and ‖w‖ is a check on the mechanism, not a target.

**Build 2:** Too little λ leaves the swings; too much flattens the curve. Only validation sees both mistakes.

- Ask: What happens at λ = 1?
- Expected: The fit is too flat to follow sin(πx): training 0.286 and validation 0.405. That is underfitting.

**Point here:** A penalty on ‖w‖ trades a little training error for a much smaller validation error, and validation sets its strength.

**Common misconception:** That the penalty helps the training error. It always makes the training error worse; it helps on new data.

**Notebook:** Start at λ = 0 (the interpolant, ‖w‖ = 465). Step right one notch at a time and read the validation column; it bottoms out at λ = 10⁻³, then rises again.

**Transition:** The penalty is one regulariser. Lecture 10 had three more, each at a different row of the recipe.

**Source:** The ten-point lab of Lecture 10 (seed 21), ported from its deck. Numbers: scripts/sci/p3.py.

## regs: Four ways to prefer a simpler fit

**Build 1:** Weight decay is the penalty of the last slide, seen in the update: shrink θ first, then take the usual step.

- Ask: Where does the factor (1 − ηλ) come from?
- Expected: From the gradient of (λ/2)‖θ‖², which is λθ. One step: 0.98 × 5 = 4.90, then − 0.1 × 3 gives 4.60.

**Build 2:** Dropout zeroes hidden units at random in training and divides the survivors by the keep rate q, so the average is unchanged.

- Ask: Why divide by q during training?
- Expected: So the expected value stays the same: E[h̃₃] = 0.5 × 12 + 0.5 × 0 = 6. At test time the full h is used with no rescaling.

**Build 3:** Label smoothing moves a little of the target to the wrong classes, so the loss stops pushing once \qcol{p_b} reaches 0.92.

- Ask: What does the plain one-hot target ask of the logits?
- Expected: An infinite gap: p_b → 1 needs the logits to grow without end. With smoothing the push stops at a gap of ln(0.92/0.02) = 3.83.

**Build 4:** Augmentation changes the Data row: each epoch sees a fresh crop and flip of every image.

- Ask: Which transforms would change the label?
- Expected: A flip that turns a 6 into a 9, or a left turn into a right turn. The transform must keep the label true, or the label must move with it.

**Point here:** Each regulariser adds a preference the ten points alone did not give, at a different row: the update, the hidden units, the target, the data.

**Common misconception:** That dropout stays on at test time. model.eval() turns it into the identity.

**Transition:** Which of these helped on real images?

**Source:** Worked numbers as in Lecture 10. The crop is the lecture’s v2.RandomCrop(32, padding=4) with a horizontal flip. Numbers: scripts/sci/p3.py.

## regs-cifar: One change at a time, on CIFAR-10

**Build 1:** Lecture 10 trained one small CNN five times, changing one thing each time. Which change do you expect to help most?

- Ask: Make a guess before the bars appear: which of the four regularisers wins?
- Expected: Most students say dropout. Augmentation wins by a wide margin.

**Build 2:** With only 2,000 images, new views of the data did more than any change to the loss or the network.

- Ask: Does this prove weight decay is useless?
- Expected: No. This is one seed and one setting each (λ = 0.01, ε = 0.1); another λ might help, and the small gaps may not survive a second seed.

**Point here:** When data are scarce, more valid views of the data is the strongest regulariser in this run.

**Common misconception:** That a regulariser which did not help at one setting never helps. Its strength was not tuned here.

**Transition:** Every number on this slide was a test accuracy. Who was allowed to look at it, and when?

**Source:** Recorded: Lecture 10, “Controlled CIFAR-10 ablations” (lecture7/diagrams/cifar_regularization_experiment.py, one seed), not retrained.

## splits: Train fits, validation chooses, test reports

**Build 1:** The training set has one job: gradient steps on it fit the weights θ.

- Ask: Why can the training set not choose λ?
- Expected: Training error is lowest at λ = 0 by construction; it would always pick the interpolant.

**Build 2:** Every choice we make (λ, when to stop, which regulariser) is made on validation. Here it picked λ = 10⁻³.

- Ask: Is validation error an honest estimate after we used it to choose?
- Expected: Slightly optimistic: we picked the λ that happened to do best on those nine points. That is why a third split exists.

**Build 3:** The test set is read once, at the end, for the report. If it changes a knob, it has become a second validation set.

- Ask: Why is the test error higher than the validation error here?
- Expected: Validation chose λ to suit its own nine points; the 101 test points are new. Test 0.124 against validation 0.065 is the honest gap.

**Point here:** Training fits θ, validation chooses everything else, and the test set reports once.

**Common misconception:** That reporting the best test score over many runs is fine. Choosing by test turns it into validation.

**Transition:** Now use all of it on a curve you will see in every project.

**Source:** The ten-point lab of Lecture 10 (seed 21), ported from its deck. Numbers: scripts/sci/p3.py.

## check-3: Check: the curves part ways

**Build 1:** Gradient descent on the ten-point lab, from w = 0. Training error falls from 0.336 to 0.012; validation error turns up after step 5,000.

- Ask: Training error keeps falling while validation error rises. What do you do?
- Expected: Early stopping: keep the checkpoint with the lowest validation error (step 5,000), then, if needed, add weight decay, dropout or augmentation, with validation choosing the strength.

**Build 2:** Restoring step 5,000 gives test error 0.125, as good as the penalty’s 0.124. Both choices were made on validation.

- Ask: Could we have chosen the step by the test error instead?
- Expected: No. Then the test error would no longer be an honest report; it would be a second validation set.

**Point here:** When the curves part, the model has started to fit the noise: stop at the best validation step, then regularise with validation as the judge.

**Common misconception:** That the last epoch is the model to keep. The best validation checkpoint is.

**Transition:** Here is Part 3 on one card.

**Source:** The ten-point lab of Lecture 10 (seed 21), ported from its deck. Numbers: scripts/sci/p3.py.

## recap-3: Part 3 in one slide

**Build 1:** Next: the model has been an MLP. Language needs one that can look back at earlier tokens.

- Ask: Which rows of the card did Part 3 change, and which did it leave alone?
- Expected: Data (splits and augmentation), Loss (the penalty and smoothed targets) and At test time (dropout off, test read once). The model and the update rule stayed the same, apart from the shrink factor weight decay adds.

**Point here:** Generalization changed three rows of the recipe and left the model alone.

**Transition:** Part 4: attention and language.

## part-4: Part 4 · Predicting the next token

**Build 1:** By the end you can say how a token reads the others, and why training hides the future.

- Ask: In Lecture 11, what did the name model see, and what did it output?
- Expected: The last three characters, and a probability for each of the 27 characters that could come next.

**Transition:** Start with the data and the output of that first model.

## next-token: One name, six guesses, a loss for each

**Build 1:** The name aabid, read three characters at a time. Each window gives one training pair. Given a a b, what should the model output?

- Ask: How many training pairs does aabid give, and why one more than its length?
- Expected: Six: one per character, plus one whose target is the boundary that ends the name.

**Build 2:** A probability for each of the 27 characters. The target i gets 0.117; the boundary -, which ends the name, gets 0.46.

- Ask: Why is i not the top bar? Is the model wrong?
- Expected: No. Many names end after a a b, so the boundary is the more common next character. Before training, i got only 0.052.

**Build 3:** The loss is the negative log of the target's probability, in nats. Each of the six pairs adds one such term.

- Ask: What would the loss be if the model gave i a probability of 1?
- Expected: Zero. The loss only shrinks to zero when the model is sure of the right answer.

**Build 4:** At test time there is no target, so we sample. The model starts a new name with - s a. Drag the temperature T.

- Ask: What does a small T do to the bars, and what does a large one do?
- Expected: Small T sharpens: n goes from 0.16 at T = 1 to 0.34 at T = 0.5, and T near 0 is the greedy choice. Large T flattens: 0.11 at T = 1.5, and the rare characters (the rest bar) gain.

**Point here:** The data supplies its own targets: every position of every name is a training pair, and the loss is −ln of the target's probability.

**Common misconception:** That temperature is used in training. It only changes how we sample at test time; the loss is computed at T = 1.

**Notebook:** Leave T at 1, then drag to 0.5 and to 1.5. The dashed outlines stay at T = 1, so the class can see the bars sharpen and flatten. "rest" is the other 19 characters together.

**Transition:** The window was three characters. How does a model read a whole sentence?

**Source:** The character model is the one Lecture 11 trained on Indian names (part1.html). Numbers: scripts/sci/p4.py.

## attn-bank: One word, two sentences

**Build 1:** Both sentences give bank the same lookup row. How can the model tell a river bank from a bank that takes cheques?

- Ask: What does an embedding table return for bank in the two sentences?
- Expected: The same row both times: a lookup cannot see the sentence.

**Build 2:** Bank's query is matched with every key up to itself. In A, river gets 0.41 and fisherman 0.23. Switch to B.

- Ask: In B, which two words does bank read most?
- Expected: cheque (0.40) and deposited (0.26).

**Build 3:** So bank's new row moves toward water in A and toward money in B. Lecture 12 chose the matrices by hand to show this.

- Ask: Where do the query, key and value of a word come from?
- Expected: From the word's own row times three learned matrices, W_Q, W_K and W_V, the same for every position (Lecture 13).

**Point here:** Attention gives each token a weighted average of the others, with weights computed from the content of the sentence.

**Common misconception:** That attention weights are learned parameters. The matrices W_Q, W_K and W_V are learned; the weights are computed afresh for every sentence.

**Notebook:** Click B, then A again. The words change under the bars, and the tallest bar moves to the word that tells the two banks apart.

**Transition:** Now the whole layer, for every token at once, in the lecture's own code.

**Source:** The bank model is the hand-built one of Lectures 12 and 13 (attention.html). Numbers: scripts/sci/p4.py.

## attn-eq: One attention layer, a piece per click

**Build 1:** Stack the token rows into E. One product scores every query against every key.

- Ask: What is the shape of S for a sentence of T tokens?
- Expected: T by T: one score for every pair of positions.

**Build 2:** Divide by √dk. Unscaled, the weaker key gets 1.13 × 10-7; scaled, it gets 0.12. The order is kept.

- Ask: Does the scaling change which key wins?
- Expected: No. It only stops the softmax from saturating: [0.12, 0.88] instead of a near one-hot.

**Build 3:** The mask adds −∞ wherever j > i, so those weights come out exactly 0 and the rest still sum to 1.

- Ask: Why add −∞ before the softmax, not zero the weights after it?
- Expected: e to the −∞ is exactly 0 and the softmax still normalises. Zeroing afterwards leaves rows that do not sum to 1.

**Build 4:** The softmax makes each row weights that sum to 1. They mix the values into a message mi of two numbers.

- Ask: What does each row of A add up to?
- Expected: One. Each message is a weighted average of the values.

**Build 5:** WO turns the message into an update of width 4, and the residual adds it. Bank's water coordinate goes from 0.70 to 2.21.

- Ask: Why add the update to e instead of replacing e?
- Expected: The residual keeps what the token already was and adds what it learned from the others. In A, bank keeps its row and gains 1.51 on the water axis.

**Point here:** One attention layer is two lines: weights from queries and keys, scaled and masked, then a mix of values added back to the rows.

**Common misconception:** That the √d_k scaling fixes overflow. Overflow is handled by subtracting the maximum inside the softmax; the scaling keeps the softmax from saturating.

**Transition:** Does this layer predict better than the window of Lecture 11? Lecture 14 trained both.

**Source:** The lecture writes the same steps as two lines: A = (Q @ K.T / d_k**0.5 + mask).softmax(dim=-1) and E_new = E + (A @ V) @ W_O. Numbers: scripts/sci/p4.py.

## attn-ppl: Perplexity on stories the models never saw

**Build 1:** If every true next token got probability 0.25, the perplexity would be 4: as unsure as a choice among four. Which model scores lowest?

- Ask: The MLP has the most parameters. Should it win?
- Expected: No. Its window concatenates the last tokens, so its first layer grows with the window, and it still sees less context. Attention shares its weights across positions.

**Build 2:** Four heads: 28.78, with 1,321,120 parameters against the MLP's 2,332,832. Attention reads further for fewer weights.

- Ask: Why do one head and four heads have the same parameter count?
- Expected: Four heads of width 16 split the same 64 columns of W_Q, W_K and W_V that one head of width 64 uses. Each head gets its own softmax, so they can read different words: perplexity falls by 8.2%.

**Point here:** Perplexity is the exponential of the held-out loss; attention beat the fixed window with fewer parameters.

**Common misconception:** That more parameters always mean a better language model. What the parameters can see matters more here.

**Transition:** The same block, wired three ways, gives the three families of Lecture 15.

**Source:** TinyStories, 4000 tokens, context 64, mean of three seeds; recorded from Lecture 14 (part3.html, capstone-evidence.json), not retrained.

## families: One block, three families

**Build 1:** One sentence, “Raghav goes to school in Delhi”. A filled cell lets that row read that column. Which family can write text one word at a time?

- Ask: Which of the three can generate text, and which cannot?
- Expected: Decoder-only and encoder-decoder generate; the encoder sees the whole input, so it labels instead.

**Build 2:** The training target follows from the mask: next words where the future is hidden, labels where every word is seen.

- Ask: Why can an encoder not be trained to predict the next word?
- Expected: Every position already reads the next word, so predicting it would be copying (this is the mask question of the next slide).

**Build 3:** The mask and the readout choose the family; the attention block is the same in all three.

- Ask: Which would you pick for translating English to Hindi, and why not decoder-only?
- Expected: Encoder-decoder: the English is read in full by the encoder, and each Hindi step looks it up again by cross-attention. A decoder-only model can also translate if the English is put first in the prompt.

**Point here:** Decoder-only, encoder and encoder-decoder differ in who may attend to whom and what the loss reads, not in the attention block itself.

**Transition:** One question on the mask before we close the part.

**Source:** The sentence and its Hindi translation are Lecture 15’s running example (data/p4.json).

## check-4: Check: the forgotten mask

**Build 1:** Two ideas meet here: what the target of each position is, and what the mask hides.

- Ask: What is the target at position i, and which positions can it read without the mask?
- Expected: The target is token i + 1; without the mask, position i reads it directly.

**Build 2:** Row 5 puts 0.67 of its weight on later words: 0.48 on river, its own answer.

- Ask: What would the loss on held-out text look like at test time?
- Expected: Generation feeds one token at a time, so the later tokens are not there. The model was never trained to predict without them, and its text falls apart even though the training loss looked perfect.

**Point here:** The causal mask makes training match test time: each position sees only what it will have when it generates.

**Common misconception:** That a near-zero training loss means a good model. Here it means the task leaked into the input.

**Transition:** Here is the whole part on one card.

**Source:** The bank model is the hand-built one of Lectures 12 and 13 (attention.html). Numbers: scripts/sci/p4.py.

## recap-4: Part 4 in one slide

**Build 1:** Next: the same block, fed image patches.

- Ask: Which rows of the card are the same as in the MLP classifier of the opening?
- Expected: Loss and Update: still −ln of the probability of the right answer, still a gradient step. Data, Model, Output and At test time changed.

**Point here:** A language model is the classifier of the opening, run at every position, with attention to gather the context.

**Transition:** Part 5: what if the tokens are not words?

## part-5: Part 5 · Images as tokens, labels as language

**Build 1:** Aim: for each of the four models, say what it is trained on, what it outputs and what it does at test time.

- Ask: Part 4’s model read text tokens. What would you have to change to make it read a photo?
- Expected: Only the input: some way to turn the photo into a sequence of vectors. The attention blocks do not care where the rows came from.

**Transition:** Start with the photo every one of these lectures used.

## vit: A photo becomes a sequence of patch tokens

**Build 1:** Lecture 16’s photo at 224 × 224, cut into square patches; each patch is one token. Switch the patch size: what happens to the tokens and the attention cost?

- Ask: Why not one token per pixel?
- Expected: 50,176 tokens, and attention compares every pair: the cost grows with the square of the count. Patches of 16 give 196.

**Build 2:** At patch size 16, each patch is 16 × 16 × 3 = 768 numbers. One shared linear map turns every patch into 192 features.

- Ask: Does the linear map know where its patch came from?
- Expected: No. It is shared by every patch, so the same crop anywhere gives the same vector. That is why a position vector is added.

**Build 3:** A learned [CLS] row and a position vector per row give 197 × 192. Then twelve attention blocks with no mask, the encoder of Part 4.

- Ask: Why no causal mask here?
- Expected: The whole photo is there at once: no patch is a “future” one to hide. This is the encoder branch of Part 4, not the decoder.

**Build 4:** The final [CLS] row goes through a linear head with 1,000 outputs, then a softmax. The right breed comes first, far ahead of the next two.

- Ask: Where is the cross-entropy?
- Expected: On the 1,000 logits, against the photo’s class: the same −ln p of the right class as in Part 1. Only the input changed.

**Point here:** A ViT is Part 4’s encoder with a new Data row: patches instead of word tokens, and a [CLS] row read out for the class.

**Common misconception:** That the ViT needs a new kind of layer for images. Apart from the patch projection, every layer is the text encoder’s.

**Notebook:** Press 32, 16 and 8: the tokens go 49, 196, 784, and the scores per head grow by the square. The outlined patch is the one Lecture 16 followed (P63). Lecture 22 rebuilt this ViT live in PyTorch with AI tools, on MNIST; its recording is on the schedule.

**Transition:** Its 1,000 classes are a fixed list. CLIP lets words supply the classes.

**Source:** Photo: newfoundland_31 from Oxford-IIIT Pet (Parkhi, Vedaldi, Zisserman and Jawahar), CC BY-SA 4.0, as the model sees it. ViT-Tiny (vit_tiny_patch16_224); the prediction is Lecture 16’s saved run. Numbers: scripts/sci/p5.py.

## clip: CLIP: a softmax along each row of the batch

**Build 1:** Lecture 17’s batch: three photos and their three captions, each encoded to a unit vector. Which cells of this matrix should training push up?

- Ask: Which cells are the observed pairs?
- Expected: The diagonal: photo i was posted with caption i. Every other cell in its row is a competitor.

**Build 2:** Divide by the temperature τ = 0.5, then a softmax along each row: each photo picks a caption. Drag τ and watch the rows sharpen or flatten.

- Ask: What does a small τ do to a row?
- Expected: It sharpens it: at τ = 0.1 the loss is 0.03, at τ = 2.0 it is 0.94. The order of the captions never changes.

**Build 3:** The loss is −ln of the paired cell, averaged over the rows and again down the columns, where each caption picks a photo.

- Ask: Where have you seen −ln of the right entry before?
- Expected: Everywhere since Part 1: it is the cross-entropy of a 3-way classifier whose classes are the captions in the batch.

**Build 4:** Training moves both encoders and the temperature so that the diagonal wins. On this one batch the loss keeps falling.

- Ask: Where are the class labels?
- Expected: There are none. The pairing of each photo with its own caption is the only supervision.

**Point here:** CLIP’s loss is ordinary cross-entropy, where the classes are the captions in the batch, done once per row and once per column.

**Common misconception:** That the softmax is taken over all captions in the dataset. It runs over the captions in the batch only, which is why CLIP needs large batches.

**Notebook:** Start at τ = 0.5 (loss 0.56). Drag to 0.1: the rows are nearly one-hot and the loss falls to 0.03. Drag to 2.0: the rows flatten towards 1/3 and the loss rises towards ln 3.

**Transition:** After training, the same two encoders classify photos with no new training at all.

**Source:** The authored 3-D batch of Lecture 17 (cat, dog, car). The loss after 100 steps is the lecture’s run. Numbers: scripts/sci/p5.py.

## clip-use: At test time, each class is written as a caption

**Build 1:** The ViT could only answer with its stored classes. With CLIP we write each class as a caption, encode it, and compare. Which caption wins?

- Ask: How would you add a new class to the ViT? And to CLIP?
- Expected: The ViT needs a new output row and training on labelled photos. CLIP needs only a new caption: no training.

**Build 2:** The photo goes to the caption with the largest cosine. All three scores come from one image vector; the captions are encoded once.

- Ask: Why “a photo of a …” rather than the bare word?
- Expected: CLIP was trained on captions, so a caption-like sentence lands closer to where the photos are. The bare word works, a little worse.

**Build 3:** Now leave “dog” out of the list. CLIP still picks one of the captions we wrote, “a photo of a bird”, ahead of “a photo of a car” by 0.0005.

- Ask: Is the winner a confident answer?
- Expected: No. The softmax always spreads 1 over the captions we wrote, even when none is right. The cosines here barely differ.

**Point here:** At test time CLIP needs no class rows: each class is a caption, and the prediction is the closest caption.

**Common misconception:** That CLIP generates the label. It only ranks the captions we supply; it writes no text.

**Transition:** To answer in words, the image has to reach a model that writes text.

**Source:** Photo: newfoundland_31 from Oxford-IIIT Pet (Parkhi, Vedaldi, Zisserman and Jawahar), CC BY-SA 4.0. Cosines: frozen CLIP ViT-B/32, Lecture 17’s saved measurements. Numbers: scripts/sci/p5.py.

## vlm: A VLM puts the image in front of a language model

**Build 1:** Lectures 18 and 19 kept the language model and gave it the image. For one photo and a 13-token prompt, how many positions does it read?

- Ask: What does the projector change, and what does it leave alone?
- Expected: It changes the width, 192 to 768, so the rows fit the language model. It keeps their number: 196 stay 196, and none becomes a word.

**Build 2:** Every patch row becomes a position in front of the prompt. Lecture 19’s eight learned summary vectors read all 196 rows and shorten the prefix.

- Ask: Are the summaries a choice of patches?
- Expected: No. Each is a learned query that mixes all the rows by attention, so every summary depends on the whole image.

**Build 3:** Training on the reference answer “A black dog.”: the loss is −ln p of each answer token, given the image and the tokens before it.

- Ask: Why is the prompt not graded?
- Expected: It is given, not predicted. Only the answer tokens get labels; the prompt’s are set to −100 and skipped.

**Point here:** A VLM is Part 4’s language model with image rows in front; it is trained with the same next-token loss, on the answer.

**Common misconception:** That the model reads the image in words. The image rows are vectors at earlier positions; they never become token IDs.

**Transition:** All these models are hard to question. The last one is built so that a person can check it.

**Source:** DeiT-Tiny rows and DistilGPT2 in Lectures 18 and 19’s notebook; the answer probabilities are the lecture’s chosen numbers. Numbers: scripts/sci/p5.py.

## cbm: A concept bottleneck: the class is read from concepts alone

**Build 1:** Lecture 20’s worksheet: three concepts, then one linear head W. The model says horse. A person sees stripes: correct it to 1. What happens?

- Ask: Which numbers does the class head see?
- Expected: Only the three concept values. That is the bottleneck: the photo reaches the class only through them.

**Build 2:** Only the stripes column of W moves the scores. Nothing is retrained: the same head, given the corrected concept, now says zebra.

- Ask: Would the same edit change a multi-task model’s answer?
- Expected: No. There the concept head is a side branch, and the class head never reads it. In a bottleneck the concepts are the only path.

**Build 3:** Training needs a label for every concept as well as the class: a concept loss on ĉ plus the usual class cross-entropy.

- Ask: What does each training scheme do with these two losses?
- Expected: Independent fits g on the concepts and f on the true concepts; sequential fits f on g’s outputs; joint minimises both losses together.

**Point here:** A concept bottleneck model changes the Output row: it outputs concepts first, and a person can correct them at test time.

**Common misconception:** That correcting a concept always helps. If the head has a wrong-signed weight, a correct edit can make the answer worse (Lecture 20, the billing example).

**Notebook:** Press the two buttons: the stripes bar turns amber and the class bars swap from horse to zebra.

**Transition:** Does this hold on real photos, with real concept predictors?

**Source:** The zebra, horse and giraffe worksheet of Lecture 20; weights from the cbm repository. Numbers: scripts/sci/p5.py.

## p5-cub: On real birds: accuracy, then accuracy after corrections

**Build 1:** Lecture 20’s run on CUB: 200 bird species, 112 concepts in 28 groups. Which model would you deploy if a bird expert will correct its concepts?

- Ask: Joint and standard are equally accurate. What does joint give that standard cannot?
- Expected: A place to intervene: its concepts can be read and corrected. The standard model has no concepts to correct.

**Build 2:** Correct one group of concepts after another. The least accurate model gains the most: its head was trained on true concepts, so it trusts corrected ones.

- Ask: Why does joint gain the least?
- Expected: Its head learned to read the predictor’s soft, drifted scores. True 0 and 1 values are not what it was trained on.

**Point here:** Accuracy alone does not rank concept bottleneck models; how much a correction helps depends on how the head was trained.

**Common misconception:** That the most accurate model is always the best choice. With an expert in the loop, the independent model wins.

**Transition:** One question that needs every model of this part.

**Source:** CUB, frozen CLIP ViT-B/32 features, one run (Lecture 20, from the cbm repository). Numbers: scripts/sci/p5.py.

## check-5: Check: which of these need class labels?

**Build 1:** Which of these four models need a class label on every training photo? Think about what the Data row of each held.

- Ask: Which need class labels to train?
- Expected: ViT and the concept bottleneck model. CLIP learns from captions, the VLM from answers in words.

**Build 2:** The Data row is where these models differ most; every loss is still −ln p.

- Ask: Which of them can add a new class without any training?
- Expected: CLIP: write the new class as a caption. A VLM can also be asked about it in words. The ViT and the bottleneck need new rows in their heads.

**Point here:** Replacing class labels by text is what lets CLIP and the VLM train on photos from the web.

**Transition:** Part 5 in one slide.

## recap-5: Part 5 in one slide

**Build 1:** Next: so far every model predicted a label or a token; next, models of the data itself.

- Ask: Which row of this card is different from the ViT’s card?
- Expected: Data (captions instead of class labels), Output and Loss (the batch’s captions are the classes), and At test time (captions written for the classes).

**Point here:** Part 5 changed every row except Update.

**Transition:** Part 6: models of the data itself.

## part-6: Part 6 · Learning a distribution

**Build 1:** By the end you can say why the VAE’s loss is a reconstruction error plus a KL.

- Ask: Every loss so far scored one prediction against one target. What would a loss between two whole distributions look like?
- Expected: Something that is zero when the two distributions agree and grows as they differ. That is the KL divergence, and Part 1’s cross-entropy is already one.

**Transition:** 6.1: the KL divergence, on Lecture 23’s weather.

## kl: The KL divergence: the extra cost of the wrong code

**Build 1:** Lecture 23’s weather. Our days follow p; we borrow the code another town built for its weather, q. How many extra bits a day does that cost?

- Ask: Before any formula: which outcome does the borrowed code handle worst for us?
- Expected: Sunny. It is our most common day (0.50), but the other town’s code gives it a long codeword because sunny is rare there (0.25).

**Build 2:** H(p) is what the best code for our days costs. The KL is the extra we pay. Drag q towards p: at q = p the extra is zero.

- Ask: Can the KL ever be negative? When is it zero?
- Expected: Never negative: no code beats the one built for the true distribution. It is zero only when q = p, which the slider shows.

**Build 3:** Part 1’s loss is the same quantity. The label is p, one-hot on cat; the prediction gives cat 0.60. The label’s entropy is 0, so cross-entropy is a KL.

- Ask: Why is it KL(label ‖ prediction), and not the other way round?
- Expected: The label is where the data come from, so it sits on the left. The reverse would need ln(q/0) for every wrong class, which is infinite.

**Point here:** KL(p ‖ q) = H(p, q) − H(p) is the extra cost of using q when the data come from p, and the classification loss is one such KL.

**Common misconception:** That KL is a distance. It is not symmetric: KL(p ‖ q) and KL(q ‖ p) differ, and the order matters in variational inference.

**Notebook:** Drag the slider from 0 to the end: the KL falls from 0.25 to 0.00 bits as q reaches p. Leave it at 0 before the click to 6.2.

**Transition:** Variational inference uses the same KL with the order flipped, to a posterior we cannot compute.

**Source:** Weather and classifier from Lecture 23; coin and mixture fit from Lecture 24. Numbers: scripts/sci/p6.py.

## elbo: The ELBO: a lower bound we can raise

**Build 1:** Lecture 24’s coin: a Beta(5, 3) prior, then 17 heads in 30 tosses. For a coin the evidence can be computed, so we can check VI against it.

- Ask: For a network, why can we not compute this number?
- Expected: The integral runs over every θ. Lecture 23 put a grid of points on each of the seven weights of its tiny network and found it would take hours; a real network has millions of weights.

**Build 2:** Pick any q, here Beta(10, 10). The evidence splits into the ELBO, which we can compute, and the KL to the posterior, which we cannot.

- Ask: Why is the ELBO always below ln p(𝒟)?
- Expected: The KL is never negative (6.1), so the ELBO can only sit at or below the fixed total. For Beta(10, 10) it is 0.62 below.

**Build 3:** The ELBO needs only q, the likelihood and the prior: fit the data, and stay near the prior.

- Ask: Which term pulls q towards the data, and which holds it back?
- Expected: The first term rewards q for putting its mass where the data are likely; the KL to the prior charges it for moving away from Beta(5, 3).

**Build 4:** Drag q. The total never changes, so whatever the ELBO gains, the KL loses. Where does the gap close?

- Ask: What makes the gap exactly zero?
- Expected: q equal to the posterior, Beta(22, 16): there the KL is 0. The fit is −20.94 and the KL to the prior 0.46, which add up to ln p(𝒟).

**Point here:** ln p(𝒟) = ELBO + KL(q ‖ posterior) is a fixed total, so raising the ELBO, which we can compute, lowers the KL, which we cannot.

**Common misconception:** That the ELBO is maximised by fitting the data alone. A spike at the best r fits better but pays a larger KL to the prior; on the coin its ELBO is lower than the posterior’s.

**Notebook:** At click 4, drag slowly from the left: the purple curve slides onto the teal one, and the red gap shrinks to 0 at Beta(22, 16), then opens again past it.

**Transition:** To raise the ELBO by gradient steps we need its gradient, and it is an average over samples of q.

**Source:** Weather and classifier from Lecture 23; coin and mixture fit from Lecture 24. Numbers: scripts/sci/p6.py.

## reparam: Gradients through a sample: θ = μ + σε

**Build 1:** Lecture 24 fitted q by drawing θ from it and averaging the loss. The code ran, and q did not move. Where does the gradient stop?

- Ask: Which arrow in the top graph has no derivative?
- Expected: The one into the draw. “Sample from 𝒩(μ, σ²)” is not a function of μ and σ that autograd can differentiate, so nothing flows back to them.

**Build 2:** Move the randomness into ε, a fixed input with no parameters. Then θ = μ + σε is ordinary arithmetic, and backprop goes through it.

- Ask: Is θ still random?
- Expected: Yes, with exactly the same distribution. Only the place where the randomness enters has moved: into ε, which needs no gradient.

**Build 3:** Put in one draw. With rsample() the same fit ends at 𝒩(0.43, 0.72²), KL 0.056.

- Ask: What is the gradient of θ with respect to σ, in words?
- Expected: The noise itself: with ε = −0.75, θ moves by −0.75 for each unit of σ. Each sample gives its own gradient, and their average estimates the true one.

**Point here:** Writing θ = μ + σε moves the randomness into an input, so the gradient of the loss reaches μ and σ; in PyTorch that is rsample().

**Common misconception:** That q.sample() raises an error when used in a loss. It runs without complaint; the parameters simply get no gradient.

**Transition:** Now the tiny network of Lecture 23, fitted by this trick.

**Source:** Weather and classifier from Lecture 23; coin and mixture fit from Lecture 24. Numbers: scripts/sci/p6.py.

## bnn: A Bayesian network: predictions with error bars

**Build 1:** Lecture 23’s tiny network: seven weights, 17 points with a gap. VI fitted q, a mean and a width per weight. Where should it be least sure?

- Ask: Where should a network that knows what it does not know be least sure?
- Expected: In the gap between the two groups of points, and beyond the ends of the data, where many curves fit equally well.

**Build 2:** At test time we draw networks from q and average them. VI widens its band in the gap, but less than the reference does: reverse KL makes q too sure.

- Ask: Why is the VI band narrower than the reference?
- Expected: VI minimises KL(q ‖ posterior), which charges q for mass where the posterior has little, so q hugs one mode and stays narrow. A diagonal q cannot tilt to follow correlated weights either.

**Point here:** A Bayesian network trained by VI returns a distribution over predictions, and its spread is the uncertainty; mean-field VI underestimates it.

**Common misconception:** That the band is a confidence interval from repeated data. It is the spread of predictions from networks drawn from q, plus the noise.

**Transition:** Lecture 25 meets the same problem with images, where the unknown is a code instead of the weights.

**Source:** The tiny network of Lectures 23 and 24; bands from generative/lectures/lecture1/data/science.json. Numbers: scripts/sci/p6.py.

## ae-vae: An autoencoder, and the VAE that grows out of it

**Build 1:** Lecture 25’s autoencoder on MNIST test digit #90, our 3: 784 pixels squeezed into two numbers and rebuilt. What does its loss compare?

- Ask: What are the input and the target of an autoencoder?
- Expected: The same image: x goes in and x is the target. No labels anywhere.

**Build 2:** The loss is the squared error between the image and its rebuild, summed over all 784 pixels. Random codes still decode to junk: nothing decided where codes live.

- Ask: Why can this autoencoder not generate new digits?
- Expected: Nothing in its loss says where the codes should be. Its codes for real digits sit far from where 𝒩(0, I) draws land, and the decoder was never trained there.

**Build 3:** The VAE’s encoder returns a small cloud: a mean and a width for each code number. In training, the decoder gets a draw z = μ + σε.

- Ask: Why give each image a cloud of codes rather than one point?
- Expected: So that the decoder learns to draw something sensible everywhere near a code, not only at one point. With a penalty that holds the clouds near 𝒩(0, I), random codes then decode to digits.

**Point here:** The VAE keeps the autoencoder’s shape but makes the code a sampled draw from a cloud that the encoder predicts.

**Common misconception:** That an autoencoder with a small reconstruction error is a good generator. The best rebuilder in Lecture 25, with the longest code, gives static from random codes.

**Transition:** The three fixes give a loss, and it is the ELBO of 6.2.

**Source:** MNIST test digit #90 and the trained networks of Lecture 25 (generative/lectures/lecture2/data). Numbers and images: scripts/sci/p6.py.

## vae-loss: The VAE’s loss: reconstruction plus KL

**Build 1:** Lecture 25’s loss in its own four lines. The model returns the rebuilt image and the encoder’s cloud, its mean and log-variance.

- Ask: Which variable in the first line is the cloud of 6.5?
- Expected: mu and logvar: the encoder’s q(z | x). The draw z = mu + exp(logvar / 2) · eps happens inside model(x).

**Build 2:** Gaussian pixels with σ = 0.2 turn the log-likelihood into squared error times 12.5, plus a constant. Here is that term for our 3.

- Ask: Why does a squared error appear in a log-likelihood?
- Expected: The log of a Gaussian density is minus the squared error over 2σ², plus a constant: the same reason squared error is the Gaussian’s negative log-likelihood in Part 1. Here 1 / (2 · 0.2²) = 12.5.

**Build 3:** The KL to 𝒩(0, I) has a closed form for two Gaussians. It pulls every image’s cloud towards the prior.

- Ask: What does the KL term do to the codes of all the images together?
- Expected: Each cloud is pulled towards 𝒩(0, I), so together they fill the region that random draws land in. That is what lets the decoder generate.

**Build 4:** Together they are minus 6.2’s ELBO, for one image. It sits just below ln p(x); the gap is the KL to the true posterior.

- Ask: We minimise the loss. What happens to the gap, the KL to the posterior?
- Expected: ln p(x) does not depend on q, so pushing the ELBO up pushes the gap down. For our 3 it is 0.70 nats: the encoder’s q is close to the posterior.

**Build 5:** Weight the KL by β and train again. Click through the seven trained VAEs: when do the codes fill the circle, and when do samples look like digits?

- Ask: At which β do the samples look best, and what do the codes do at β = 0?
- Expected: Around β = 1 to 4. At β = 0 there is no pull: the KL per image is 68.3, 267 of the 400 codes leave the square and samples from 𝒩(0, I) are poor.

**Point here:** The VAE is trained on minus the ELBO for each image: a reconstruction term from Gaussian pixels plus the KL of the encoder’s q to 𝒩(0, I).

**Common misconception:** That the KL term is a regulariser bolted on by hand. It falls out of the ELBO of 6.2, with the code z in the place of the weights θ.

**Notebook:** At click 5, start at β = 1, then click 0 and 256: watch the codes spread off the square, then collapse onto the centre. Keep 256 for the check that follows.

**Transition:** One question on the two ends of that sweep.

**Source:** MNIST test digit #90 and the trained networks of Lecture 25 (generative/lectures/lecture2/data). Numbers and images: scripts/sci/p6.py.

## check-6: Check: the two ends of the β sweep

**Build 1:** Answer in pairs before the click. You need two ideas from this part: when a KL is zero, and what the decoder can learn from a code.

- Ask: With β = 256 the KL is 0.00. What do the samples look like, and why?
- Expected: All eight are one blurry digit. KL(q ‖ prior) = 0 only if q equals 𝒩(0, I) for every image, so z carries no information about x; the best the decoder can do is draw the average image.

**Build 2:** Rebuilds fail too: squared error rises from 30.77 at β = 1 to 52.90 at β = 256.

- Ask: Is this a failure of training?
- Expected: No: it is the best answer to the loss we wrote. A KL weight that large makes any information in the code cost more than it saves in reconstruction. This is called posterior collapse.

**Point here:** The KL term trades information in the code against closeness to the prior; with too much weight on it, the code is ignored.

**Transition:** Part 6 in one slide.

**Source:** MNIST test digit #90 and the trained networks of Lecture 25 (generative/lectures/lecture2/data). Numbers and images: scripts/sci/p6.py.

## recap-6: Part 6 in one slide

**Build 1:** Next: one more change of output, from one label to a list of objects.

- Ask: Which rows of the card are new compared with the classifier of the opening?
- Expected: Data has no labels; the model has an encoder and a decoder with a sampled code; the output is a distribution over codes and images; the loss is minus the ELBO; at test time we sample a code and decode it. Only the update is the same gradient step.

**Point here:** Generative modelling changed every row of the recipe except the update.

**Transition:** Part 7: object detection.

## part-7: Part 7 · From one label to a list of objects

**Build 1:** By the end you can say how a network with a fixed-size output returns a list whose length changes from photo to photo.

- Ask: A photo holds a dog and a cat. Which rows of the recipe must change to get both of them, with where they are?
- Expected: The Output (a list of boxes), and so the Data (people draw boxes) and the Loss. At test time there is new work: choosing which boxes to keep.

**Transition:** Start from what a classifier says about one photo.

## det-list: From one label to a list of boxes

**Build 1:** Lecture 26’s courtyard photo. Before any boxes: if we gave it to a classifier from Part 5, what would come back?

- Ask: What would an image classifier return for this photo?
- Expected: One class with a probability: a softmax over its classes, as in Part 5.

**Build 2:** One label for the whole photo, with its probability. The cat, the plants, and where anything sits, are all lost.

- Ask: What does that answer leave out?
- Expected: The second animal, the plants, how many objects there are, and where each one is.

**Build 3:** A detector returns a list: one row per object, each a class, a score and a box. Another photo gives a list of another length.

- Ask: How many numbers describe one detection, and what changes from photo to photo?
- Expected: A class, a score and four numbers for the box. What changes is the number of rows: four here.

**Point here:** The Output row changes: a list of boxes, each with a class and a score, whose length depends on the photo.

**Common misconception:** That a detector runs a classifier once per object. It is one network, run once, that writes a box and scores in every cell of a grid.

**Transition:** To train boxes or to judge them, we first need to say when two boxes agree.

**Source:** Photo: Lecture 26’s courtyard. The photographs are course-owned generated images (see SOURCES.md); every box and score drawn on them is computed. Classifier: ResNet-50 on ImageNet, recorded from Lecture 26 (data/models.json). Boxes and scores: YOLOv8n, recorded from Lecture 26 (data/science.json). Numbers: scripts/sci/p7.py.

## iou: Do two boxes agree? Intersection over union

**Build 1:** The annotated dog and a guess the sliders move. Lecture 26’s guess sat 60 px right and 40 px down. What IoU would you give it?

- Ask: What IoU does a guess 60 px right and 40 px down get?
- Expected: 0.62: overlap 139,965 over union 224,645. Most of the class will guess higher.

**Build 2:** A small shift already costs a lot. IoU decides three things today: which boxes NMS deletes, which detections count as right, and in training which cell owns a box.

- Ask: Where else will IoU be used in this part?
- Expected: In NMS, to find duplicates (7.4), and in evaluation, to call a detection right (7.5).

**Point here:** IoU is overlap over union: one number for how well two boxes agree, used in training, in NMS and in evaluation.

**Common misconception:** That a box 60 px off is nearly perfect. Both boxes are 182,305 px², and the shift already drops IoU to 0.62.

**Notebook:** Start at 0 and 0: IoU 1. Drag right to 60 and down to 40: 0.62. Then drag right all the way left, until the boxes part: 0.

**Transition:** Now train a network to write boxes: one target per grid cell.

**Source:** Photo: Lecture 26’s courtyard. The photographs are course-owned generated images (see SOURCES.md); every box and score drawn on them is computed. The dog box: Lecture 26’s annotation; the guess: its box moved by hand. Numbers: scripts/sci/p7.py.

## det-loss: The loss, one grid cell at a time

**Build 1:** A 3 × 2 grid of 320-pixel cells. The dog’s centre lies in cell (1, 1), so that cell’s target holds its box. The untrained network writes this.

- Ask: Which cell is responsible for the dog, and why that one?
- Expected: Cell (1, 1): it holds the centre of the dog’s box, (587.5, 335.5).

**Build 2:** Square each difference and add: the loss of the dog’s cell. An empty cell pays only for its object? score.

- Ask: Which entry of the dog’s record costs the most?
- Expected: cat: the target is 0 and the untrained network says 0.52, a squared error of 0.27.

**Build 3:** Every cell’s loss is written in its cell. Gradient steps on their sum move the network’s table toward the target table.

- Ask: Why do the four empty cells cost anything at all?
- Expected: Their target object? is 0, and the untrained network says about 0.49 everywhere, so each pays 0.24. Their box and class entries are skipped.

**Point here:** The Data row now holds boxes, and the Loss row compares a target record with the network’s record in every cell. Lecture 26 used squared error throughout; real detectors use binary cross-entropy (Part 1’s −ln p) for object? and the class scores, and a box loss for the box.

**Common misconception:** That every cell is trained to find the dog. Only the cell holding its centre gets the dog’s box; the others learn object? = 0.

**Transition:** After training, every cell writes a box. Most of them are not wanted.

**Source:** Photo: Lecture 26’s courtyard. The photographs are course-owned generated images (see SOURCES.md); every box and score drawn on them is computed. The network: Lecture 26’s small grid CNN; its table before training and its loss after training recorded from data/models.json. Numbers: scripts/sci/p7.py.

## nms: Keep the confident boxes, then drop the duplicates

**Build 1:** YOLOv8n writes a box and scores in every cell of three grids. 31 boxes pass a score of 0.25, several per object. How do we keep one each?

- Ask: Why does one dog get many boxes?
- Expected: Neighbouring cells all see the dog and each writes a box for it: 10 dog boxes pass 0.25.

**Build 2:** Non-maximum suppression (NMS) keeps the best box and deletes its near copies, one class at a time. Press NMS in the figure.

- Ask: Why only boxes of the same class?
- Expected: A person on a bicycle: the two boxes overlap and both are right. NMS run per class keeps both.

**Build 3:** Four detections survive, one per object: the list of 7.1. A fixed number of records became a list of any length.

- Ask: What happens as the score threshold rises?
- Expected: Fewer boxes pass, and NMS still leaves one per object. At 0.30 the small potted plant (0.29) is gone; at 0.90 only the cat is left.

**Point here:** At test time two choices turn a fixed grid of records into a list: a score threshold, and NMS with an IoU threshold.

**Common misconception:** That NMS keeps only the single best box. It loops, so every object keeps its own best box, and boxes of other classes are never compared.

**Notebook:** Start with NMS off at 0.25: 31 boxes. Press NMS: 4. Then drag the threshold up and down, with NMS on and off.

**Transition:** Is that list any good? Count what it got right.

**Source:** Photo: Lecture 26’s courtyard. The photographs are course-owned generated images (see SOURCES.md); every box and score drawn on them is computed. Boxes and scores: YOLOv8n, recorded from Lecture 26 (data/science.json). Class-aware NMS rerun here at every threshold. Numbers: scripts/sci/p7.py.

## det-eval: How good is the list? Precision, recall and AP

**Build 1:** Lecture 26’s teaching example: five dogs, eight ranked boxes, each a TP or FP by IoU ≥ 0.5. Keep more boxes, one at a time: where does the point go?

- Ask: With only the top box kept, what are precision and recall?
- Expected: Precision 1/1 = 1, recall 1/5 = 0.2: right, but most dogs are missed.

**Build 2:** Each threshold is one point on the curve. Recall can only grow as boxes are added; precision falls whenever a wrong box comes in.

- Ask: Why can recall never reach 1 here?
- Expected: One dog, the small one far away, has no box at all, so no threshold can find it.

**Build 3:** One number for the whole curve: the area under its upper envelope, the average precision (AP).

- Ask: When would AP be 1?
- Expected: Only if every dog is found and every true positive is ranked above every false positive.

**Point here:** At test time we choose the threshold, and it trades precision against recall; AP scores the whole ranking at once.

**Common misconception:** That accuracy would do. A true negative is “no box and no object”, and every empty patch of grass is one, so detection counts only TP, FP and FN.

**Notebook:** Drag from the top box to all eight, reading precision and recall at each step. Stop at all eight: the lecture’s point.

**Transition:** One question that needs IoU and NMS together.

**Source:** The example: Lecture 26’s dog park, five annotated dogs and eight made-up ranked boxes (data/dog-park.json); TP, FP, precision, recall and AP recomputed here. Numbers: scripts/sci/p7.py.

## check-7: Check: two boxes on one dog

**Build 1:** Two ideas at once, IoU and NMS. Work it out with your neighbour before the next click.

- Ask: How many dogs are reported, and what is the precision?
- Expected: Two dogs; precision 0.5.

**Build 2:** NMS let the duplicate through, and evaluation counted it against us. An NMS threshold below 0.62 would delete it and give precision 1.

- Ask: What would you change to fix it, and what could go wrong?
- Expected: Lower the NMS threshold below 0.62. In a crowd that can delete a real neighbour whose box overlaps that much.

**Point here:** A duplicate that NMS misses costs precision; a threshold that is too low costs recall in crowds.

**Transition:** The part in one slide.

**Source:** Photo: Lecture 26’s courtyard. The photographs are course-owned generated images (see SOURCES.md); every box and score drawn on them is computed. The two scores are made up for this question; the IoU is the guess of 7.2. Numbers: scripts/sci/p7.py.

## recap-7: Part 7 in one slide

**Build 1:** Next: all seven parts side by side.

- Ask: Which row of the recipe did detection leave alone?
- Expected: The Update: the same gradient step as every other model in the course. Data, Output, Loss and At test time all changed.

**Point here:** Detection is the course’s recipe with a new output: the training loop is unchanged, and the new work is in the targets and at test time.

**Transition:** Now all seven parts, side by side.

## p7-code: IoU and NMS in PyTorch (optional)

**Build 1:** Why does nms run once per class in a real detector?

- Ask: Why does nms run once per class in a real detector?
- Expected: So that boxes of different classes never delete each other: torchvision’s batched_nms does exactly that.

**Point here:** Lecture 26’s two tools in a few lines each; the courtyard’s boxes go through them in scripts/sci/p7.py.

**Source:** Code: Lecture 26 (iou-code, nms-code).

## threads-ce: The same loss, five times

**Build 1:** Five models from five parts of the course. Each gave a probability p to what was observed. Which one was penalised most?

- Ask: Without computing anything: how do you rank the five?
- Expected: By p alone. The smallest p gets the largest penalty, whatever the model is.

**Build 2:** Each loss is −ln p, so all five sit on one curve.

- Ask: Where on this curve is a model that is certain and right? Certain and wrong?
- Expected: Certain and right is p = 1, where the loss is 0. Certain and wrong is p near 0, where the curve rises without limit.

**Build 3:** The loss reads one number, the probability given to what was observed.

- Ask: Squared error and the ELBO are not on this slide. Do they belong on the curve?
- Expected: They are the same idea with a density in place of a probability: squared error is −ln of a Gaussian density (Lectures 1 and 2), and the ELBO is a bound on ln p(𝒟) (Lecture 24). A density can exceed 1, so those losses can be negative and do not sit on this curve.

**Point here:** Across the course the loss was −ln of the probability given to what was observed.

**Common misconception:** That CLIP, a language model and a classifier use different losses. They differ in what the probability is over.

**Notebook:** Each row of the table opens the slide its number came from.

**Transition:** The loss is one thread. The attention block is another.

**Source:** Every number is read from this deck’s Parts 1, 4, 5 and 6 (scripts/sci/p8.py).

## threads-attn: The same block, in four models

**Build 1:** Part 4 built one block: self-attention under a mask. A language model hides the future and predicts the next token at every position.

- Ask: Why must a language model hide the future during training?
- Expected: The target at each position is the next token. If the block could see it, the model would copy it and learn nothing.

**Build 2:** Lecture 16 changed what a token is. A photo became 196 patches and a [CLS] token; nothing is hidden, and the class is read at [CLS].

- Ask: Why does the Vision Transformer need no mask?
- Expected: It predicts one class for the whole photo, so every patch may look at every other patch. There is no future to leak.

**Build 3:** CLIP runs two encoders, one for the photo and one for the caption, and compares the two vectors they return.

- Ask: Which of CLIP’s two encoders uses a causal mask?
- Expected: The text encoder, which reads the caption left to right and returns the vector at its last token. The image encoder is a Vision Transformer read at [CLS].

**Build 4:** A vision-language model puts the 196 patch rows in front of the question, in one sequence. The answer comes out at the last position, one token at a time.

- Ask: What does the projector do before the patch rows enter the language model?
- Expected: It changes their width from 192 to 768 so they fit the language model. Their number stays 196.

**Point here:** One block served four models. They differ in what a token is, which mask is used and where the output is read.

**Common misconception:** That vision models need a different architecture from language models. After Lecture 16 the block is the same.

**Transition:** Two threads, then. Now every part against the six rows of the card.

**Source:** Parts 4 and 5 of this deck; Lectures 12 to 19.

## parts-rows: Seven parts, six rows

**Build 1:** A bar marks a row of the card that the part changed. Read down the Update column: why is it almost empty?

- Ask: Which rows changed most often?
- Expected: Data, Output, Loss and At test time: five parts each. A new task mostly means new data, a new output and a new use of it.

**Build 2:** This is why one training loop could serve every model of the course.

- Ask: Part 6 sent the gradient through a sample, θ = μ + σε. Is that a new update rule?
- Expected: No. The reparameterisation changes how the loss is written so that backprop can reach μ and σ. The step is still a gradient step, which is why Part 6 has no bar in that column.

**Point here:** New tasks changed the data, the output and the loss. The update was built once, in Part 2.

**Common misconception:** That each architecture needs its own optimizer. Adam trained the classifier, the language model and the VAE alike.

**Notebook:** Each part’s name opens its recap slide.

**Transition:** The same view, one model at a time.

## master: Ten models, one card each

**Build 1:** Pick a model. Its card shows the six rows, and the tinted rows are the ones that differ from the MLP classifier’s. How many models tint the Update row?

- Ask: Which model has the fewest tinted rows, and why?
- Expected: The Vision Transformer, with three. It is a classifier like the MLP, so its loss and its decision are the MLP’s.

**Build 2:** Even there it is a gradient step. Everything else that changed is data, model, output, loss or use.

- Ask: Which models need no labels at all?
- Expected: The language models, CLIP and the VAE. Their targets come from the data itself: the next token, the paired caption, the image.

**Point here:** Every model of the course is the same card with some rows rewritten.

**Common misconception:** That a tinted row means a new idea each time. The tinted Loss rows of the language models, CLIP and the vision-language model are all −ln p of something observed.

**Notebook:** Click down the list in order. Stop at CLIP (no class labels) and at the VAE (no labels at all).

**Transition:** The card also works forwards, for a problem the course never covered.

## pipeline: The recipe, for a problem of your own

**Build 1:** The card as a flowchart. Each box is a question to answer before you write any code.

- Ask: Which box would you answer first?
- Expected: The first and the third: what is observed, and which distribution the model puts over it. The loss follows from those two.

**Build 2:** These are the answers the course gave. Now a new problem: forecast tomorrow’s PM2.5 and say how sure you are. What goes in each box?

- Ask: For the forecast, what is observed and what must the model output?
- Expected: Observed: tomorrow’s measured PM2.5, a number. Output: a distribution over that number, so a mean and a spread.

**Build 3:** One good answer. The Gaussian output carries the error bar, and its loss is squared error divided by the predicted spread, plus a term for the spread.

- Ask: How does the network keep the spread positive?
- Expected: A softplus or an exponential on the raw output, the “valid outputs by design” of Lecture 9.

**Point here:** A new problem is a new set of answers to the same six questions.

**Common misconception:** That uncertainty needs a Bayesian network. A Gaussian output head already gives an error bar for the noise in the data.

**Transition:** The course, once more, on one slide.

**Source:** Lectures 1 to 3 for the Gaussian likelihood; Lecture 9 for the positive spread.

## summary: The course on one slide

**Build 1:** Twenty-six lectures as one method. For your own problem, which row of the card comes first?

- Ask: Take a problem from your own work. What is observed, and which distribution over it would your model output?
- Expected: Any answer that names what is observed and a distribution over it: a class and a softmax, a number and a Gaussian, a sentence and a next-token distribution. The loss then follows as −ln p of what was observed.

**Point here:** Choose what is observed and a distribution over it. The loss and the update follow, as they did for all ten models.

**Common misconception:** That a new problem needs a new learning algorithm. It needs a new answer to the six rows.

**Transition:** The appendix has every recipe card and every lecture with its links.

## p8-card-mlp: Recipe card: MLP classifier (optional)

**Build 1:** Which rows differ from the MLP classifier’s card?

- Ask: Which rows differ from the MLP classifier’s card?
- Expected: None. This is the card the others are compared with.

## p8-card-charlm: Recipe card: character language model (optional)

**Build 1:** Which rows differ from the MLP classifier’s card?

- Ask: Which rows differ from the MLP classifier’s card?
- Expected: The tinted ones: Data, Model, Output, Loss, At test time.

## p8-card-gpt: Recipe card: transformer language model (optional)

**Build 1:** Which rows differ from the MLP classifier’s card?

- Ask: Which rows differ from the MLP classifier’s card?
- Expected: The tinted ones: Data, Model, Output, Loss, At test time.

## p8-card-vit: Recipe card: Vision Transformer (optional)

**Build 1:** Which rows differ from the MLP classifier’s card?

- Ask: Which rows differ from the MLP classifier’s card?
- Expected: The tinted ones: Data, Model, Output.

## p8-card-clip: Recipe card: CLIP (optional)

**Build 1:** Which rows differ from the MLP classifier’s card?

- Ask: Which rows differ from the MLP classifier’s card?
- Expected: The tinted ones: Data, Model, Output, Loss, At test time.

## p8-card-vlm: Recipe card: vision-language model (optional)

**Build 1:** Which rows differ from the MLP classifier’s card?

- Ask: Which rows differ from the MLP classifier’s card?
- Expected: The tinted ones: Data, Model, Output, Loss, At test time.

## p8-card-cbm: Recipe card: concept bottleneck model (optional)

**Build 1:** Which rows differ from the MLP classifier’s card?

- Ask: Which rows differ from the MLP classifier’s card?
- Expected: The tinted ones: Data, Model, Output, Loss, At test time.

## p8-card-bnn: Recipe card: Bayesian neural network (optional)

**Build 1:** Which rows differ from the MLP classifier’s card?

- Ask: Which rows differ from the MLP classifier’s card?
- Expected: The tinted ones: Data, Model, Output, Loss, Update, At test time.

## p8-card-vae: Recipe card: variational autoencoder (optional)

**Build 1:** Which rows differ from the MLP classifier’s card?

- Ask: Which rows differ from the MLP classifier’s card?
- Expected: The tinted ones: Data, Model, Output, Loss, Update, At test time.

## p8-card-det: Recipe card: object detector (optional)

**Build 1:** Which rows differ from the MLP classifier’s card?

- Ask: Which rows differ from the MLP classifier’s card?
- Expected: The tinted ones: Data, Model, Output, Loss, At test time.

## p8-lectures-1: Lectures 1 to 5, with their material (optional)

**Build 1:** Which of these would you reread first?

- Ask: Which of these would you reread first?
- Expected: Any. Each title opens the lecture’s row on the course page, where the short videos and the labs are also listed.

## p8-lectures-2: Lectures 6 to 10, with their material (optional)

**Build 1:** Which of these would you reread first?

- Ask: Which of these would you reread first?
- Expected: Any. Each title opens the lecture’s row on the course page, where the short videos and the labs are also listed.

## p8-lectures-3: Lectures 11 to 15, with their material (optional)

**Build 1:** Which of these would you reread first?

- Ask: Which of these would you reread first?
- Expected: Any. Each title opens the lecture’s row on the course page, where the short videos and the labs are also listed.

## p8-lectures-4: Lectures 16 to 20, with their material (optional)

**Build 1:** Which of these would you reread first?

- Ask: Which of these would you reread first?
- Expected: Any. Each title opens the lecture’s row on the course page, where the short videos and the labs are also listed.

## p8-lectures-5: Lectures 21 to 26, with their material (optional)

**Build 1:** Which of these would you reread first?

- Ask: Which of these would you reread first?
- Expected: Any. Each title opens the lecture’s row on the course page, where the short videos and the labs are also listed.
