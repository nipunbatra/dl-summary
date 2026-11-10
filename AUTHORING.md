# Authoring guide: the summary lecture (Lecture 27, ES 667 Deep Learning)

Read this whole file before writing a slide. Then read `parts/00-opening.js`: it is the finished exemplar, and every
part should read as if the same person wrote it.

## 1. What this deck is

A 90-minute revision lecture for students who sat through all 26 lectures. It has one idea:

> One recipe trained every model of the course: **Data, Model, Output, Loss, Update, At test time**.
> Each part of the course changed some of those six rows. The lecture walks the seven parts in order and asks of
> each: which rows did it change, and how?

The opening (Part 0, done) runs the recipe once on Lecture 4's XOR network. Parts 1 to 7 are the seven modules of the
schedule. The wrap (Part 8) lines up the ten models in one table. `plan/outline.md` has the plan of every part.

Because it is revision, a slide does not teach from zero. It **recalls**: the running example the students already
know from that lecture, its numbers, the one equation, and a question that makes them retrieve the idea before the
answer appears. Where the original lecture took ten slides to derive something, this deck shows the result on the same
example in one slide and links back to the lecture (`from:`).

## 2. What you own, and what you must not touch

You own exactly four things for your part `N` (file names are in `plan/outline.md`):

- `parts/NN-name.js`: your slides and your live figures
- `scripts/sci/pN.py`: computes or records every number your slides show
- `data/pN.json`: written by that script
- `figures/pN-*`: any image you copy in

Do not edit `kit.js`, `widgets.js`, `lecture.css`, `content.js`, `index.html`, `shared/`, `scripts/*.mjs`, another
part's files, `data/course.json` or anything outside `/Users/nipun/git/dl-summary`. Six other authors are writing the
other parts at the same time in this folder. If the kit lacks something, write a local helper in your part file and
local styles with `css('.pN-thing{…}')` (every class name starts with `pN-`). If you believe a kit change is needed,
say so in your report instead of making it.

Do not start, stop or restart any server. The deck is already served at `http://127.0.0.1:8790` (no caching).
Do not use git in this folder (no add, commit, stash, checkout). Do not install packages.

Source material is read-only:

- Your digest in `plan/digests/` has the running examples, numbers, equations, code and deep links of your lectures.
  Start there; it is usually all you need.
- `/Users/nipun/git/generative/lectures/` (KL and VI: `lecture1`; VAE: `lecture5`), `/Users/nipun/git/object-detection`
  and `/Users/nipun/git/cbm`: you may read files directly. Never write there.
- `/Users/nipun/git/dl-teaching`: read only through `git -C /Users/nipun/git/dl-teaching show origin/master:<path>`.
  Its working copy holds someone else's uncommitted work.
- `/Users/nipun/git/attention`: read only through `git -C /Users/nipun/git/attention show origin/main:<path>`.

The source decks number their lectures differently (the VAE deck calls itself Lecture 5, the detection deck Lecture
22). On a slide, always use the schedule's numbers: the ones in `data/course.json` and in `plan/outline.md`.

## 3. The shape of a part

Every part has the same shape, in this order:

1. `partBreak(k, title, lead, how, {line, ask, expected, transition})`: the aim in one sentence (`line`), a one or two
   sentence `lead`, and three or four steps `how`. Its id is `part-k`.
2. Four to seven content slides (`S_`), with the ids given in `plan/outline.md`. Before each, name the recipe row it
   is about: `chap('3.2','Loss: a penalty on the weights')`. The kicker then shows it.
3. `check-k`: one question for the class that needs two ideas of the part, at build 1; `answer(2, …)` at build 2.
4. `recap(k, did, next, {card, title, ask, expected, point, transition})`: two or three lines of what the part did,
   beside the recipe card of the part's model. Its id is `recap-k`. The card comes from `RECIPES` in `kit.js`
   (`card: RECIPES.gpt`, or `{...RECIPES.mlp, step:'…'}` to reword a row). The rows your part changed are tinted
   automatically from `PARTS`.

Stay inside your part's click budget (`plan/outline.md`): one click is about 30 seconds in class. If an idea does not
fit, do not squeeze it: put it on an appendix slide with `O('pN-something', title, {...})` (off the main route).
Slide ids listed in the outline are fixed, because the wrap links to them. Any extra id starts with `pN-`.

## 4. The kit (import only from `../kit.js`)

`kit.js` re-exports `shared/author.js` and `widgets.js`; read all three (they are short) and `lecture.css`.

Slides

- `S_(id, section, title, {builds, body, lines[], ask[], expected[], point, misconception, transition, demo, source, from})`.
  `lines`, `ask` and `expected` need exactly one entry per build. `lines[k]` is the story line under the slide at
  click k. `from: 12` or `from: [25,'#fix-loss/4']` puts a "From Lecture n" link top right (deep links to slides you
  know exist are better; your digest lists them).
- `O(id, title, {...})`: an appendix slide. `partBreak`, `recap`, `chap`, `strip`: see section 3.
- Text: `p(s)` (lead, 28px), `ps(s)` (26px), `small(s)` (caption), `takeaway(s)`, `answer(build, s)`,
  `kt()` / `kp()` / `kl()` colour a word as target (teal), prediction (purple), loss (red).
- Layout: `cols(left, right, '1fr 1.2fr')`, `at(build, html)` (appears from that click, keeps its space),
  `only([builds], html)` (shown only then), `stack(...)`, `sm(...)`, `table(headers, rows)`.
- Maths: `m(tex)` inline, `eq(tex)` display, written with `R` (`String.raw`): ``m(R`\qcol{\hat y}`)``.
  Colour macros: `\qcol` prediction (purple), `\pcol` target (teal), `\lcol` loss (red), `\dcol` data (blue),
  `\ncol` noise (amber). Also `\E`, `\KL`, `\N`, `\ELBO`. KaTeX throws on an error, which fails the deck.
- Numbers: `num(v, 2)` for text (true minus sign), `tn(v, 2)` inside TeX (ASCII minus), `big(12600)`,
  `word(7)` / `Word(7)` for small counts in prose.
- Code: `codeSteps([[0,'line shown from the start'],[2,'line shown from click 2'], …])`, wrapped in
  `<div class="code-sm">…</div>` when it sits in a column. `code(text)` for a static block.
- The recipe: `card(fill, {hi, title, sm})`, `RECIPES`, `SLOTS`, `PARTS`, `MODELS`.
- Data: `const D = need(await data('pN'), 'key1', 'key2');` at the top of the file (top-level await works).

Figures (`widgets.js`): `fig(W, H, inner, alt)` wraps SVG built from `label`, `line`, `arrow`, `path`, `dot`, `rect`,
`pill`; `panel({...})` gives axes and scales; `bars(items, {...})`, `grid(M, {...})`, `chart(rows, {...})` (flowchart),
`photo(name, W, H, inner, alt, {view})` for `figures/<name>.jpg` with SVG drawn on top. Colours: `COL.target`,
`COL.guess`, `COL.loss`, `COL.data`, `COL.amber`, `COL.green`, `COL.ink`, and `MUTED`, `AX`.

- Inside an SVG, reveals are `gat(build, svg)` and `gonly(build, svg)`. `at()` and `only()` are HTML and break out
  of the SVG.
- SVG text scales with the figure. Size the viewBox to the column the figure sits in (full width is 1152 units,
  the usable height of a slide is about 450) and keep label sizes at 19 to 24.

Live figures. A slide that argues about a parameter (a step size, a temperature, a threshold, a weight) gets a
control that changes it. The pattern is in `parts/00-opening.js` (`p0-loop`):

- The slide places `widget('pN-name', 'alt text')`. The part file calls `register('pN-name', (el, ctx) => {…})`.
- First line inside: `if (ctx.printing) { el.innerHTML = <a static figure at a sensible setting>; return; }`.
- Otherwise build `'<div class="w-fig">' + svg + '</div><div class="w-controls">' + buttons + slider(...) + '</div>'`,
  keep the unchanging part of the SVG outside a `<g class="live"></g>` and redraw only that group in `show()`.
- Sliders: `input` → `show()`, `change` → `blur()`. Buttons: toggle `aria-pressed`, `blur()`, `show()`. Call
  `show()` once. Blurring matters: the arrow keys must go back to moving through the slides.
- Precompute what the control sweeps over in `pN.py` (a curve per setting, or a function the JS can evaluate
  exactly, such as a softmax). JS may do arithmetic on data from the JSON; it may not hold constants of its own.
- If the figure asks a question ("which one learns?"), a later click answers it in words with `answer()`.

## 5. Numbers

No number is typed into a part file. Every number a slide shows is read from `data/pN.json`, and `scripts/sci/pN.py`
writes that file. Follow `scripts/sci/p0.py` for the layout (docstring with the run command, `OUT = …/data/pN.json`,
asserts on the values the slides rely on, rounded floats).

- Recompute what is cheap: closed forms, a softmax, a small fit, a few hundred gradient steps. Use the same example
  and settings as the lecture (they are in your digest) and assert that you reproduce the lecture's numbers.
- Record what is expensive (a trained model, a dataset run, a detector's output): put the number in `pN.py` as a
  constant with a comment naming the lecture and file it came from, and write it to the JSON under `recorded`.
  Do not download models or datasets.
- If you cannot reproduce a number of the lecture, keep the lecture's number as recorded and say so in your report.
- Interpreters: `/Users/nipun/git/clip-lab/.venv/bin/python` (torch, numpy) or `~/base/bin/python` (numpy, scipy,
  PIL, matplotlib, torch). MNIST is at `/Users/nipun/git/psdv-teaching/datasets`.
- Python writes float dictionary keys as `'0.01'`, `'0.1'`, `'1.0'`. In JS take keys from `Object.keys(...)`.
- Counts in prose come from data too: `${Word(D.items.length)} boxes`, never a typed "Four boxes".

## 6. How Nipun wants slides

Pace and structure

- One move per click. A slide makes one point; a click adds one thing.
- Recall first. Where it fits, build 1 poses the question on the lecture's own example and the next clicks answer it.
- One running example per idea, with its numbers. After an equation appears, the next click puts the example's
  numbers into it.
- A loss slide shows the prediction in purple (`\qcol`), the target in teal (`\pcol`) and the loss in red (`\lcol`),
  with numbers.
- Code is minimal PyTorch, at most 12 lines, one line per click beside the figure or number that line computes.
- A flowchart (`chart`) for any pipeline of more than two stages.
- Caveats, credits and "strictly speaking" go in the notes (`point`, `misconception`, `source`), off the slide.
- No non-vegetarian examples (no meat, fish or eggs in any analogy or illustration).
- The CLIP slides never say "zero-shot". Say what happens: each class is written as a caption.

Density (checked by `scripts/density.mjs`)

- A story line is at most 30 words. A state (body + story line + half the SVG labels) is at most 100 words; aim
  under 70. A code block is at most 12 lines.
- The body never repeats the story line. The story line says what to look at or asks the question; the body shows it.
- Nothing may overflow the slide, and no two labels may touch. Leave air around a figure.

Words

- Plain, concrete sentences in the voice of a teacher talking to the class ("we", "you").
- No slogan fragments ("One loss. Every model."), no forced groups of three, no "X, not Y" endings, no rhetorical
  "It's not just A, it's B", no decorative bold, no emojis, no title case in headings.
- No em dashes or en dashes anywhere, on slides or in notes. Write ranges as "1 to 4". Curly quotes are fine.
- Use the lecture's own terms and keep one name per thing across your part.
- Every number is explained on the slide or in the story line, and shown to about 2 decimals.
- Shared terms across the deck: the six rows are called Data, Model, Output, Loss, Update, At test time; `θ` for all
  parameters; `η` is "the step size"; `ln` for the natural log and "nats" for its unit; "the recipe" for the six rows.

Notes (they are the lecturer's script; fill them in for every build)

- `ask[k]`: what to ask the class at click k. `expected[k]`: the answer you hope for.
- `point`: the one sentence the slide exists for. `misconception`: the common wrong belief, when there is one.
- `transition`: the sentence that leads to the next slide. `demo`: what to do with a live figure.
- `source`: one short line under the slide: where the example comes from and which script made the numbers
  (and the credit for a photo).

## 7. Checking your work (required before you report)

From `/Users/nipun/git/dl-summary` (every Bash command should start with `cd /Users/nipun/git/dl-summary;`):

```
node --check parts/NN-name.js
DECK_PART=NN node scripts/verify.mjs          # every click rendered; screenshots in qa/parts/NN/
DECK_PART=NN node scripts/density.mjs         # word counts per click
```

`verify` must print `"passed": true`. `density` must print `over a limit: none`. Then **look at every screenshot**
in `qa/parts/NN/` with the Read tool, at every click, and fix what you see: text over a figure, labels that touch,
a figure that is too small to read from the back of a room, an answer visible before its click, a card that wraps
badly, a slide that looks empty at build 1. Repeat until a careful reader would find nothing to fix. A part that
passes the scripts but was never looked at is not finished.

Things that have gone wrong before

- `builds` must equal the lengths of `lines`, `ask` and `expected`, or the deck throws.
- A slide must fit at its final build; `at()` keeps the space of hidden things, so check build 1 for big holes.
- `verify` fails on SVG text that crosses the viewBox edge, on any `h2, p, pre, .display, table, li` outside the
  60px side margins, on content taller than the slide, and on a missing image.
- `register()` throws on a name used twice; a `widget()` with no `register()` throws at mount.
- A lone "·" or "→" between spaces counts as a word in the density check.
- `table()` does not escape its cells: pass HTML you trust, and escape anything else with `esc()`.
- To view your part in a browser: `http://127.0.0.1:8790/?all&part=NN#slide-id/2`.

## 8. Your report

End with a short report (plain text, no file):

1. The slides in order: id, builds, one line each. The total number of clicks against your budget.
2. The live figures: name, what the control changes.
3. The recap card: which `RECIPES` entry, and the exact text of any row you reworded.
4. Numbers: which were recomputed and matched the lecture, which are recorded, and any that differ from the digest.
5. The last output of `verify` and `density`, and a statement that you looked at every screenshot.
6. Anything you were unsure about, anything left undone, and any kit change you would ask for.
