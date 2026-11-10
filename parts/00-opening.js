// The opening: the cover, the models the course built, and the one recipe that trained them all, run on Lecture 4's
// XOR network. Every number comes from data/p0.json (scripts/sci/p0.py) or data/course.json (scripts/sci/course.py).
import {S_,p,cols,card,codeSteps,answer,m,R,data,need,course,meta,MODELS,word,Word,css,kt,kp,kl,
 fig,label,line,path,dot,panel,chart,widget,register,slider,num,COL,MUTED,AX,esc} from '../kit.js';

const D=need(await data('p0'),'xor','loop'),X=D.xor,L=D.loop;
const SCHED='https://nipunbatra.github.io/dl-2026/schedule.html';
const ETAS=Object.keys(L.loss),ETA0=ETAS.find(k=>+k===L.default);        // the keys as the script wrote them: '0.01', '0.1', '1.0'
const END=L.steps,endLoss=k=>L.loss[k][END],endP=k=>L.probs[k][END];
const SOURCE='The network is the one Lecture 4 built by hand for XOR. Numbers: scripts/sci/p0.py.';

css(`.p0-models .tile b{margin:5px 0 0}
.p0-route td:first-child{white-space:nowrap}
.p0-route a{color:var(--ink);text-decoration:none;font-weight:650}.p0-route a:hover{text-decoration:underline}`);

// ---------------------------------------------------------------------------------------- the cover
/** The schedule as a picture: one row per module, one tile per lecture, each tile a link to its row on the course page. */
function courseMap(){
 const tw=46,th=38,gap=8,rh=52,fs=22,left=318;
 const rows=[...course.modules.map(mod=>[mod.title,mod.lectures,false]),['Today',[meta.number],true]];
 const W=left+Math.max(...rows.map(r=>r[1].length))*(tw+gap)-gap+4;
 const rowId=n=>course.lectures.find(l=>l.n.includes(n)).n[0];         // Lectures 23 and 24 share one row of the schedule
 const g=rows.map(([name,ns,now],i)=>{
  const y=i*rh+(now?12:0)+2;
  return label(left-16,y+th/2+fs*.36,esc(name),{size:fs,anchor:'end',fill:now?COL.guess:COL.ink,weight:now?700:''})+ns.map((n,j)=>{
   const x=left+j*(tw+gap);
   const t=`<rect x="${x}" y="${y}" width="${tw}" height="${th}" rx="8" fill="${now?COL.guess:'#fff'}" stroke="${now?COL.guess:AX}" stroke-width="2"/>`+
    label(x+tw/2,y+th/2+fs*.36,n,{size:fs,anchor:'middle',fill:now?'#fff':COL.ink,weight:now?700:''});
   return now?t:`<a href="${SCHED}#lecture-${rowId(n)}" target="_blank" rel="noopener"><title>Lecture ${n} on the course page</title>${t}</a>`;}).join('');
 }).join('');
 return fig(W,rows.length*rh+8,g,`The ${word(course.count)} lectures of the course in their ${word(course.modules.length)} modules, and today's lecture, number ${meta.number}`);
}

S_('title','map',meta.title,{builds:1,
 body:`<div class="cover" style="grid-template-columns:1fr 1.35fr"><div>${p(`${Word(course.count)} lectures built ${word(MODELS.length)} models, and one recipe trained every one of them. Today we write that recipe down once, then see what each part of the course changed in it.`)}
  <p class="who">${meta.series} · Lecture ${meta.number}<br>Nipun Batra · IIT Gandhinagar</p></div>${courseMap()}</div>`,
 lines:[`Every tile is one lecture and opens its slides. The ${word(course.modules.length)} rows are the ${word(course.modules.length)} parts of today.`],
 ask:['Which lecture do you remember best, and what did its model output?'],
 expected:['Any lecture works. Write two or three of the outputs the class names on the board (a class, the next token, a box); the next slide collects all of them.'],
 point:'The whole course fits on one page, and it is one method applied seven times.',
 transition:'First, the models themselves.',source:'Course page: nipunbatra.github.io/dl-2026/schedule.html'});

// ---------------------------------------------------------------------------------------- the models
S_('ten-models','map',`${Word(MODELS.length)} models, built in ${word(course.count)} lectures`,{builds:2,
 body:`<div class="tiles p0-models" style="grid-template-columns:repeat(5,1fr)">${MODELS.map(M=>`<div class="tile">${M.name}<b>${M.out}</b></div>`).join('')}</div>
  ${answer(2,`Each one outputs a probability distribution. Its loss is the negative log of the probability given to what was observed.`)}`,
 lines:[`The models we built, in order, each with what it returns. What do all ${word(MODELS.length)} have in common?`,
  'So one recipe trained them all. The next slides run it once, on the smallest network we built.'],
 ask:['What do these models have in common?','Squared error does not look like a negative log probability. Where does it fit?'],
 expected:['They are all neural networks trained by gradient descent on a loss. Push for the sharper answer: every loss here is a negative log probability.',
  'Squared error is the negative log likelihood of a Gaussian with a fixed variance (Lectures 1 and 2), so it is the same loss with a different output distribution.'],
 point:'What a model returns changes from lecture to lecture. The way we train it stays the same.',
 misconception:'That each new architecture came with a new learning algorithm. Only the pieces plugged into the recipe changed.',
 transition:'Here is the recipe, one step per click.'});

// ---------------------------------------------------------------------------------------- the recipe, once
const sg=v=>v<0?`− ${Math.abs(v)}`:`+ ${v}`,T0=X.theta0;
const pEnd=endP(ETA0)[3],xEnd=L.inputs[3];
S_('recipe','map','The recipe, on one example',{builds:6,
 body:chart([
  {name:'Training: one step, on one example',steps:[
   {k:'Data',t:`x = (${X.x.join(', ')})\ny = ${X.y}`,col:COL.target},
   {k:'Model',t:`z = ${T0.w2[0]}h₁ ${sg(T0.w2[1])}h₂ ${sg(T0.b2)}\n= ${num(X.z)}`,at:2},
   {k:'Output',t:`p = σ(z) = ${num(X.p)}`,col:COL.guess,at:3},
   {k:'Loss',t:`−ln p = ${num(X.loss)}`,col:COL.loss,at:4},
   {k:'Update',t:`θ ← θ − ${X.eta} × g`,n:`last bias: ${num(T0.b2,0)} → ${num(X.after.b2)}`,at:5}]},
  {name:`At test time, after ${L.steps} such steps on the four XOR points`,at:6,steps:[
   {k:'Input',t:`x = (${xEnd.join(', ')})`},
   {k:'The trained model',t:`p = ${num(pEnd)}`,col:COL.guess},
   {k:'Decision',t:`p ${pEnd<.5?'<':'>'} 0.5, so class ${pEnd<.5?0:1}`}]}],
  {W:1152,gx:34,gy:92,fs:21,loop:[0,4,1,'repeat with the new θ',5],alt:'The recipe as a flowchart: data, model, output, loss and update in a loop, and the trained model used at test time'}),
 lines:[`One training example from Lecture 4's XOR problem: the input x = (${X.x.join(', ')}) and its ${kt('target y = '+X.y)}.`,
  `The model is a function with ${X.n_params} parameters θ. Its two ReLU units give h = (${X.h.map(v=>num(v,0)).join(', ')}), and the last layer turns them into a score, z = ${num(X.z)}.`,
  `A sigmoid turns the score into a probability. The network says y = 1 with ${kp('p = '+num(X.p))}.`,
  `The loss is the negative log of the probability the network gave to the right answer: ${kl(`−ln ${num(X.p)} = ${num(X.loss)}`)}.`,
  `Backprop gives the gradient g of the loss for all ${X.n_params} parameters. One step of size ${X.eta} against it moves p from ${num(X.p)} to ${num(X.after.p)} and the loss to ${num(X.after.loss)}.`,
  'Training repeats that step. At test time the loop is gone: the trained network takes an input and returns a probability.'],
 ask:['What are the two things a training example gives us?','Which of these numbers can training change?','Why a sigmoid, and what would we use for ten classes?',
  'What would the loss be if the network had said p = 1? And p = 0.5?','Which way does the last bias move, and why?','What is missing from the bottom row that the top row has?'],
 expected:['An input x and the answer y we want for it.','Only the parameters θ. The data and the form of the model stay fixed.',
  'The output has to be a valid probability. A sigmoid gives one number between 0 and 1; a softmax does the same for many classes (Lecture 9).',
  `Zero for p = 1, and ln 2 = ${num(L.ln2)} for p = 0.5. The loss grows as the network gets less sure of the right answer.`,
  `Up, from ${num(T0.b2,0)} to ${num(X.after.b2)}: the gradient on the score is p − y = ${num(X.dz)}, and the step goes against the gradient, which raises z and so raises p.`,
  'The target, the loss and the update. At test time nobody tells the network the answer.'],
 point:'Data, model, output, loss and update form a loop; at test time only the first three run.',
 misconception:'That the loss is used at test time. It only exists where we have the right answer.',
 transition:'One step barely moved the loss. What happens when we repeat it?',source:SOURCE});

// ---------------------------------------------------------------------------------------- the recipe, repeated
const YMAX=.8,LW=1152,LH=250,ETA_COL={[ETAS[0]]:COL.data,[ETAS[1]]:COL.guess,[ETAS[2]]:COL.amber};
const P=panel({W:700,H:LH,L:58,R:112,T:44,B:40,xMin:0,xMax:L.steps,yMin:0,yMax:YMAX,ticksX:[0,100,200,300],ticksY:[0,.2,.4,.6,.8],
 fmtX:v=>v?v:'step 0',yLabel:'mean loss on the four XOR points'});
const curve=(k,upto=L.steps)=>L.loss[k].slice(0,upto+1).map((v,t)=>[P.x(t),P.y(Math.min(v,YMAX))]);
const TX=[790,934,1074];                                                 // the table beside the plot: input, target, prediction
const loopStatic=P.axes+ETAS.map(k=>path(curve(k),{stroke:ETA_COL[k],w:2,op:.3})+label(P.x(L.steps)+16,P.y(endLoss(k))+7,`η = ${+k}`,{size:21,fill:COL.ink})).join('')+
 label(TX[0],34,'input x',{size:21,anchor:'middle',weight:650})+label(TX[1],34,'target y',{size:21,anchor:'middle',weight:650})+label(TX[2],34,'prediction p',{size:21,anchor:'middle',weight:650})+
 line(TX[0]-70,46,TX[2]+70,46,{stroke:AX})+
 L.inputs.map((x,i)=>label(TX[0],82+i*38,`(${x.join(', ')})`,{size:23,anchor:'middle',fill:COL.ink})+label(TX[1],82+i*38,L.targets[i],{size:23,anchor:'middle',fill:COL.target,weight:650})).join('');
function loopLive(k,t){
 const v=L.loss[k][t],pk=L.peak[k],y=P.y(Math.min(v,YMAX));
 return path(curve(k,t),{stroke:ETA_COL[k],w:4.5})+line(P.x(t),P.y(0),P.x(t),P.y(YMAX),{stroke:AX,w:1.5,dash:'4 5'})+dot(P.x(t),y,8,{fill:ETA_COL[k]})+
  (pk.loss>YMAX?label(P.x(14),P.y(.5),`rises to ${num(pk.loss)} at step ${pk.step}, above this chart`,{size:19,fill:COL.ink,halo:true}):'')+
  L.probs[k][t].map((q,i)=>label(TX[2],82+i*38,num(q),{size:23,anchor:'middle',fill:COL.guess,weight:650})).join('')+
  line(TX[0]-70,206,TX[2]+70,206,{stroke:AX})+label(TX[0]-70,238,`mean loss after ${t} step${t===1?'':'s'}`,{size:21,fill:COL.ink})+label(TX[2]+70,238,num(v),{size:23,anchor:'end',fill:COL.loss,weight:650});
}
const loopAlt='The loss of the XOR network over 300 steps at three step sizes, beside its four predictions';
register('p0-loop',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${LW} ${LH}" role="img" aria-label="${loopAlt}">${loopStatic}<g class="live">${inner}</g></svg></div>`;
 if(ctx.printing){el.innerHTML=frame(loopLive(ETA0,L.steps));return;}
 el.innerHTML=frame('')+`<div class="w-controls"><span>step size</span>${ETAS.map(k=>`<button type="button" data-eta="${k}" aria-pressed="${k===ETA0}">η = ${+k}</button>`).join('')}${slider('t','steps taken',0,L.steps,1,0)}</div>`;
 const live=el.querySelector('.live'),range=el.querySelector('[data-k="t"]'),out=el.querySelector('[data-o="t"]');
 let k=ETA0;
 const show=()=>{out.textContent=range.value;live.innerHTML=loopLive(k,+range.value);};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());
 for(const b of el.querySelectorAll('button[data-eta]'))b.addEventListener('click',()=>{k=b.dataset.eta;
  for(const o of el.querySelectorAll('button[data-eta]'))o.setAttribute('aria-pressed',o===b);b.blur();show();});
 show();
});

const big3=ETAS[2];
S_('recipe-loop','map','The recipe, repeated',{builds:2,
 body:`${widget('p0-loop',loopAlt)}
  ${answer(2,`Only η = ${+ETA0} learns XOR: its loss falls from ${num(L.loss[ETA0][0])} to ${num(endLoss(ETA0))}. At ${+ETAS[0]} it only reaches ${num(endLoss(ETAS[0]))}. At ${+big3} it jumps to ${num(L.peak[big3].loss)}, then stays at ${num(endLoss(big3))} with every prediction ${num(endP(big3)[0],1)}.`)}`,
 lines:[`The same step, now on all four XOR points, ${L.steps} times. Drag the number of steps and switch the step size η. Which of the three learns XOR?`,
  'The loop is the same in all three runs; only the step size differs. Part 2 comes back to that choice.'],
 ask:[`Before dragging: the four predictions start at ${L.probs[ETA0][0].map(v=>num(v)).join(', ')}. Which of them are already on the right side of 0.5?`,
  `Why does the large step end at exactly ${num(L.ln2)}?`],
 expected:['All four are, but none is confident: the hand-built network of Lecture 4 separates XOR, and training makes it surer.',
  `${num(L.ln2)} is ln 2, the loss of a network that says 0.5 for every input. After step ${L.h2_off_from[big3]} the second ReLU unit outputs 0 on all four inputs and never gets a gradient again. One unit cannot draw XOR, so the best the network can do is say 0.5 everywhere.`],
 point:'Training is one small step repeated many times, and the step size decides whether it works.',
 misconception:`That a larger step only gets there faster. Here it switches a unit off for good on step ${L.h2_off_from[big3]}.`,
 demo:`Set η = ${+ETA0} and drag slowly from 0: the loss on step 1 is ${num(L.loss[ETA0][1],3)}, a little above the start (${num(L.loss[ETA0][0],3)}), before it falls. Keep that for the question at the end of Part 2.`,
 transition:'Six lines of PyTorch run this loop for every model in the course.',source:SOURCE});

// ---------------------------------------------------------------------------------------- the recipe as a card and as code
const RECIPE={data:`pairs ${m('(x,\\,y)')}`,model:`scores ${m(R`z=f_\theta(x)`)}`,out:`probabilities ${m(R`\qcol{p}=\operatorname{softmax}(z)`)}`,
 loss:m(R`\lcol{-\ln p_{\pcol{y}}}`)+', the right class',step:m(R`\theta\leftarrow\theta-\eta\,\nabla_{\!\theta}\,\lcol{\text{loss}}`),test:`the class with the largest ${m(R`\qcol{p}`)}`};
S_('recipe-card','map','The recipe as a card, and as code',{builds:6,
 body:cols(card(RECIPE,{title:'The recipe, for a classifier'}),`<div class="code-sm">${codeSteps([[0,'for x, y in loader:'],[2,'    logits = model(x)'],
  [3,'    loss = F.cross_entropy(logits, y)'],[4,'    opt.zero_grad()'],[5,'    loss.backward()'],[6,'    opt.step()']])}</div>`,'1.05fr 1fr'),
 lines:['The card has one row per step of the flowchart. Beside it is the loop that ran in every lecture. The loader hands over a batch of pairs (x, y).',
  'The model turns the inputs into scores, one per class.',
  'cross_entropy applies the softmax and takes −ln of the probability of the right class, so this one line is two rows of the card.',
  'PyTorch adds new gradients to the old ones, so the loop clears them first.',
  'backward() is backprop. It fills in the gradient of the loss for every parameter.',
  'step() applies the update. With SGD or with Adam, this line stays the same.'],
 ask:['Which row of the card has no line in this loop?','What is the shape of logits for a batch of 32 images and 10 classes?','Why does PyTorch not want probabilities here?',
  'What happens if we forget this line?','Which lecture built this line from scratch?','Where does the step size live?'],
 expected:['At test time: it runs after the loop, with no loss and no update.','32 by 10: one score per class for every example.',
  'Working from the scores is numerically safer than taking the log of a softmax (Lecture 9: logits of 1000 overflow).',
  'The gradients of every batch pile up, and the steps grow in a direction that mixes old batches with the new one.',
  'Lecture 6, with a small autograd engine; Lecture 8 extended it to vectors and matrices.',
  'Inside the optimizer, set when it was created: opt = torch.optim.SGD(model.parameters(), lr=0.1).'],
 point:'The six rows of the card are the six things to ask about any model.',
 misconception:'That this loop is specific to classifiers. Only the model and the loss line change for the other models.',
 transition:'Next, the route we take through the course today.'});

// ---------------------------------------------------------------------------------------- the route
const ROUTE=[['Foundations','the loss, read off a likelihood'],['Gradients and optimization','how the update is computed'],
 ['Generalization','how the model does on new data'],['Attention and language','a model for sequences of tokens'],
 ['Vision and language','images as tokens, captions as labels'],['Generative modelling','distributions over weights and codes'],
 ['Object detection','the output becomes a list of boxes']];
if(ROUTE.length!==course.modules.length)throw new Error('The route needs one row per module');
S_('route','map','The route for today',{builds:1,
 body:`<div class="table-sm p0-route"><div class="table-scroll"><table><thead><tr><th>Part</th><th>What it changed in the recipe</th></tr></thead><tbody>${
  ROUTE.map(([name,what],i)=>`<tr><td><a href="#part-${i+1}">${i+1} · ${name}</a></td><td>${what}</td></tr>`).join('')}</tbody></table></div></div>`,
 lines:['Each part changed some rows of the card. We take the parts in order and ask of each: which rows, and how?'],
 ask:['Which part changed the Update row?'],
 expected:['Part 2. It is the only part that changes nothing else: backprop and the optimizers work the same for every model that follows.'],
 point:'We read each of the seven parts as a change to the same six rows.',
 transition:'Part 1: where the loss comes from.'});
