// Part 2: Gradients and optimization (Lectures 5 to 9). The Update row: backprop gives the gradient, a mini-batch
// estimates it, momentum and Adam shape the step, a schedule shrinks it. Every number comes from data/p2.json
// (scripts/sci/p2.py), which recomputes the lectures' examples and checks them against the lectures' values.
import {S_,partBreak,recap,chap,cols,at,only,eq,sm,R,num,tn,kt,kp,kl,answer,codeSteps,code,css,RECIPES,data,need,course,
 fig,label,line,arrow,path,dot,panel,gat,widget,register,slider,COL,MUTED,AX,esc} from '../kit.js';

const D=need(await data('p2'),'bp','sgd','opt','vjp','softmax','sched');
const SCRIPT='Numbers: scripts/sci/p2.py.';
/** A link to one slide of a lecture's PDF deck, picked by the link's label on the course page (Lectures 8 and 9 have two decks). */
const deck=(n,lab,page)=>{const l=course.lectures.find(x=>x.n.includes(n)),a=l&&l.links.find(x=>x.label===lab);
 if(!a)throw new Error(`No link "${lab}" for Lecture ${n}`);return {href:`${a.href}#page=${page}`,label:`From Lecture ${n}`};};
const GRAD=COL.amber;                                                     // gradients flowing backward, as in Lecture 6
const vec=(a,n=2)=>`(${a.map(v=>num(v,n)).join(', ')})`;
const clip=(id,x0,y0,x1,y1)=>`<clipPath id="${id}"><rect x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}"/></clipPath>`;

css(`.p2-tab table{font-size:24px}.p2-tab th,.p2-tab td{padding:8px 16px;text-align:right;font-variant-numeric:tabular-nums}
.p2-tab th:first-child,.p2-tab td:first-child{text-align:left}
.p2-sm table{font-size:23px}.p2-sm td,.p2-sm th{padding:7px 14px;text-align:right;font-variant-numeric:tabular-nums}.p2-sm td:first-child,.p2-sm th:first-child{text-align:left}
.p2-q{font-size:30px;line-height:1.45;max-width:1000px}
.p2-stack{display:flex;flex-direction:column;gap:14px}`);

// ---------------------------------------------------------------------------------------- the part opens
partBreak(2,'How the weights move',
 'The Update row needs a gradient and a step size. These lectures were about getting both right.',
 ['Backpropagation computes the gradient, one node at a time.','A mini-batch gives a cheap, noisy estimate of it.',
  'Momentum and Adam shape each step.','A schedule shrinks the step size over time.'],
 {line:'By the end you can run backprop by hand and say what each optimizer adds to the plain step.',
  ask:'In θ ← θ − η ∇loss, which pieces do we get to choose?',
  expected:'How the gradient is measured (on how much data) and the step: its size η, how it changes over time, and whether it uses memory of past gradients. Backprop itself is fixed: it computes the exact gradient of whatever loss we wrote.',
  transition:'First, where the gradient comes from.'});

// ---------------------------------------------------------------------------------------- 2.1 backprop on one graph
const B=D.bp;
chap('2.1','Update: the gradient by backprop');
const NODE={w:[100,74],x:[100,236],mul:[340,155],add:[610,155],b:[610,292],sq:[880,155],y:[880,292],L:[1080,155]};
const box=(c,t,{w=132,h=54,col=AX,fill='#fff',tc=COL.ink,bold=true}={})=>`<rect x="${c[0]-w/2}" y="${c[1]-h/2}" width="${w}" height="${h}" rx="10" fill="${fill}" stroke="${col}" stroke-width="2.5"/>`+
 label(c[0],c[1]+8,t,{size:23,anchor:'middle',fill:tc,weight:bold?650:''});
const op=(c,t)=>`<circle cx="${c[0]}" cy="${c[1]}" r="32" fill="#EEF3FD" stroke="${COL.data}" stroke-width="2.5"/>`+label(c[0],c[1]+10,t,{size:30,anchor:'middle',fill:COL.data,weight:650});
const gsub=v=>`g<tspan dy="6" font-size="16">${v}</tspan><tspan dy="-6"> </tspan>`;
const N=NODE,fwd=(x,y,s,col=COL.ink)=>label(x,y,s,{size:22,anchor:'middle',fill:col,weight:650}),bwd=(x,y,s)=>label(x,y,s,{size:22,anchor:'middle',fill:GRAD,weight:700});
const bpFig=fig(1152,330,
 arrow(N.w[0]+66,N.w[1]+10,N.mul[0]-32,N.mul[1]-12,{stroke:MUTED,w:2.5,head:12})+arrow(N.x[0]+66,N.x[1]-10,N.mul[0]-32,N.mul[1]+12,{stroke:MUTED,w:2.5,head:12})+
 arrow(N.mul[0]+34,N.mul[1],N.add[0]-34,N.add[1],{stroke:MUTED,w:2.5,head:12})+arrow(N.b[0],N.b[1]-27,N.add[0],N.add[1]+34,{stroke:MUTED,w:2.5,head:12})+
 arrow(N.add[0]+34,N.add[1],N.sq[0]-82,N.sq[1],{stroke:MUTED,w:2.5,head:12})+arrow(N.y[0],N.y[1]-27,N.sq[0],N.sq[1]+29,{stroke:MUTED,w:2.5,head:12})+
 arrow(N.sq[0]+82,N.sq[1],N.L[0]-56,N.L[1],{stroke:MUTED,w:2.5,head:12})+
 box(N.w,`w = ${B.w}`)+box(N.x,`x = ${B.x}`)+box(N.b,`b = ${B.b}`)+box(N.y,`y = ${B.y}`,{col:COL.target,tc:COL.target})+
 op(N.mul,'×')+op(N.add,'+')+box(N.sq,'(ŷ − y)²',{w:160,col:COL.data,fill:'#EEF3FD',tc:COL.data})+box(N.L,'L',{w:108,col:COL.loss,tc:COL.loss})+
 fwd((N.mul[0]+N.add[0])/2,N.mul[1]-16,`wx = ${num(B.m,0)}`)+fwd((N.add[0]+N.sq[0]-48)/2,N.add[1]-16,`ŷ = ${num(B.yhat,0)}`,COL.guess)+
 gat(2,label(N.L[0],N.L[1]-44,`L = ${num(B.loss,0)}`,{size:23,anchor:'middle',fill:COL.loss,weight:700}))+
 gat(3,bwd((N.add[0]+N.sq[0]-48)/2,N.add[1]+34,`${num(B.g_yhat,0)}`))+
 gat(4,bwd((N.mul[0]+N.add[0])/2,N.mul[1]+34,`${num(B.g_yhat,0)}`)+bwd(N.b[0]+128,N.b[1]+8,`${gsub('b')}= ${num(B.g_b,0)}`))+
 gat(5,bwd(N.w[0],N.w[1]-40,`${gsub('w')}= ${num(B.g_w,0)}`)),
 'Lecture 6’s computation graph: w times x, plus b, gives the prediction; the squared error against y gives the loss. Gradients flow back along the arrows.');
const bpEq=[R`\qcol{\hat y}=wx+b=${B.w}\cdot ${B.x}+${B.b}=${tn(B.yhat,0)}`,
 R`\lcol{L}=(\qcol{\hat y}-\pcol{y})^2=(${tn(B.yhat,0)}-${B.y})^2=${tn(B.loss,0)}`,
 R`\frac{\partial \lcol{L}}{\partial \qcol{\hat y}}=2(\qcol{\hat y}-\pcol{y})=2(${tn(B.yhat,0)}-${B.y})=${tn(B.g_yhat,0)}`,
 R`g_b=\underbrace{${tn(B.g_yhat,0)}}_{\text{arrives}}\times\underbrace{1}_{\text{local}}=${tn(B.g_b,0)}`,
 R`g_w=\underbrace{${tn(B.g_yhat,0)}}_{\text{arrives}}\times\underbrace{x}_{\text{local}}=${tn(B.g_yhat,0)}\times ${B.x}=${tn(B.g_w,0)}`];
S_('bp-graph','grad','Backprop, one node at a time',{builds:5,from:deck(6,'Slides',110),
 body:`${bpFig}<div class="state-slot">${bpEq.map((t,i)=>only([i+1],eq(t))).join('')}</div>`,
 lines:['Lecture 6’s graph for one prediction, ŷ = wx + b. The forward pass works left to right and stores every value on the way.',
  `The target is ${kt('y = '+B.y)}. Should w go up or down to lower the loss?`,
  'Backward starts at the loss and walks right to left. The square node passes back its local slope.',
  'An add node copies the gradient that arrives to both of its inputs.',
  'A multiply node scales it by the other input. The gradient on w is negative, so w should go up.'],
 ask:['Which values does the backward pass need from the forward pass?','Before computing anything: up or down for w?',
  'Why do we start backward at the loss and not at w?','What does an add node do to the gradient?','What would g_x be, and why do we not use it?'],
 expected:['The stored values: x for the multiply node, ŷ and y for the square. That is why the forward pass keeps them.',
  `Up. The prediction ${num(B.yhat,0)} is below the target ${B.y}, and x is positive, so a larger w raises ŷ toward y.`,
  'The loss is the one scalar we want to reduce; one sweep from it gives the gradient for every input at once.',
  'Its local slope is 1 for both inputs, so it hands the same number to each.',
  `${num(B.g_yhat,0)} × w = ${num(B.g_x,0)}. x is data, not a parameter, so nothing updates it.`],
 point:'Every node multiplies the gradient that arrives by its own local slope; one backward sweep gives every gradient.',
 misconception:'That the gradient says how far to move. It gives a direction and a rate; the step size sets the distance.',
 transition:'Now take one step with these two gradients.',source:`Example: Lecture 6, the central graph. ${SCRIPT}`});

// ---------------------------------------------------------------------------------------- 2.2 one step in PyTorch
const A1=B.after,BG=B.big;
chap('2.2','Update: one step');
const nn=v=>num(v,Number.isInteger(v)?0:2);
const cell=(b,s,cls='')=>`<td${cls?` class="${cls}"`:''}><span data-at="${b}">${s}</span></td>`;
S_('bp-update','grad','One step, in PyTorch',{builds:4,from:deck(6,'Slides',120),
 body:cols(`<div class="code-sm">${codeSteps([[0,`w = torch.tensor(${B.w}., requires_grad=True)`],[0,`b = torch.tensor(${B.b}., requires_grad=True)`],
  [0,`loss = (w * ${B.x} + b - ${B.y}) ** 2`],[2,'loss.backward()'],[3,'with torch.no_grad():'],[3,`    w -= ${B.eta} * w.grad`],[3,`    b -= ${B.eta} * b.grad`],
  [4,`loss = (w * ${B.x} + b - ${B.y}) ** 2`]])}</div>`,
  `<div class="p2-tab"><div class="table-scroll"><table><thead><tr><th></th><th>before</th><th><span data-at="2">gradient</span></th><th><span data-at="3">after one step</span></th></tr></thead><tbody>
  <tr><td>w</td><td>${nn(B.w)}</td>${cell(2,nn(B.g_w),'loss')}${cell(3,nn(A1.w))}</tr>
  <tr><td>b</td><td>${nn(B.b)}</td>${cell(2,nn(B.g_b),'loss')}${cell(3,nn(A1.b))}</tr>
  <tr><td>${kp('ŷ')}</td><td class="q">${nn(B.yhat)}</td><td></td>${cell(4,nn(A1.yhat),'q')}</tr>
  <tr><td>${kl('loss')}</td><td class="loss">${nn(B.loss)}</td><td></td>${cell(4,nn(A1.loss),'loss')}</tr></tbody></table></div></div>`,'1.15fr 1fr'),
 lines:[`The same graph in PyTorch. The forward line builds the graph and stores its values; the ${kt('target is '+B.y)}.`,
  'backward() runs the sweep from the last slide and leaves each gradient in .grad.',
  `The step moves each parameter against its gradient, scaled by the step size η = ${B.eta}.`,
  'Run the forward pass again with the new values. The prediction moved toward the target.'],
 ask:['What does requires_grad=True ask PyTorch to do?','Why would a training loop call zero_grad() before this line?',
  'Why is the update inside torch.no_grad()?',`Would a step ten times larger, η = ${BG.eta}, do better?`],
 expected:['To record every operation on that tensor, so backward() can replay the graph.',
  'backward() adds into .grad. Without clearing it, the gradients of earlier steps pile up.',
  'The update is not part of the model. Recording it would add the step itself to the graph.',
  `No. It moves w to ${num(BG.w)} and b to ${num(BG.b)}, so ŷ jumps past the target to ${num(BG.yhat)} and the loss is ${num(BG.loss)} again.`],
 point:'One step is: forward, backward, then move every parameter a little against its gradient.',
 misconception:'That a larger step always learns faster. Here ten times the step overshoots and the loss does not fall at all.',
 transition:'This gradient used one example. With many examples, how much data should measure it?',source:`Example: Lecture 6. ${SCRIPT}`});

// ---------------------------------------------------------------------------------------- 2.3 a noisy gradient
const A=D.sgd,CH=A.choices,KEYS=A.order;
chap('2.3','Update: a noisy gradient');
const HS=350,PA=panel({W:345,H:HS,L:66,R:14,T:34,B:50,xMin:1.35,xMax:2.85,yMin:-1.25,yMax:.45,ticksX:[1.5,2,2.5],ticksY:[-1,-.5,0],xLabel:'b',yLabel:'w'});
const PL=panel({ox:385,W:420,H:HS,L:56,R:14,T:34,B:50,xMin:-1.6,xMax:2.6,yMin:-2.5,yMax:7,ticksX:[-1,0,1,2],ticksY:[0,3,6],xLabel:'x',yLabel:'y'});
const LOSS0=String(A.levels.find(c=>Math.abs(c-A.loss0)<1e-9));
const inBox=(P,o)=>([u,v])=>u>=o.xMin&&u<=o.xMax&&v>=o.yMin&&v<=o.yMax;
const sgdBox={xMin:1.35,xMax:2.85,yMin:-1.25,yMax:.45};
const lblAt=A.contours[LOSS0].filter(inBox(PA,sgdBox)).reduce((a,q)=>q[0]>a[0]?q:a);  // where the start contour leaves the frame on the right
const th=t=>[PA.x(t[0]),PA.y(t[1])];
const sgdStatic=clip('p2-sgd-clip',PA.x(1.35),PA.y(.45),PA.x(2.85),PA.y(-1.25))+PA.axes+PL.axes+
 `<g clip-path="url(#p2-sgd-clip)">${Object.entries(A.contours).map(([c,pts])=>path(pts.map(th),c===LOSS0?{stroke:COL.loss,w:2.5,dash:'7 6'}:{stroke:AX,w:1.5})).join('')}</g>`+
 label(th(lblAt)[0]-8,th(lblAt)[1]-12,`loss ${num(A.loss0)}`,{size:20,anchor:'end',fill:COL.loss,halo:true})+
 label(PA.x(1.42),PA.y(.45)+26,`minimum ${vec(A.theta_star,1)} ↖`,{size:19,fill:MUTED,halo:true})+
 A.x.map((x,i)=>label(PL.x(x)-12,PL.y(A.y[i])-14,i+1,{size:20,anchor:'end',fill:COL.data,weight:650})).join('')+
 path([[PL.x(-1.6),PL.y(A.theta0[0]-1.6*A.theta0[1])],[PL.x(2.6),PL.y(A.theta0[0]+2.6*A.theta0[1])]],{stroke:MUTED,w:2.5,dash:'8 6'})+
 label(PL.x(2.6),PL.y(A.theta0[0]+2.6*A.theta0[1])+28,'before',{size:20,anchor:'end',fill:MUTED});
function sgdLive(k){
 const c=CH[k],t1=c.theta1,yl=x=>t1[0]+t1[1]*x,[x0,y0]=th(A.theta0),[x1,y1]=th(t1),R0=840;
 return KEYS.filter(j=>j!==k).map(j=>{const [a,b]=th(CH[j].theta1);return arrow(x0,y0,a,b,{stroke:AX,w:2,head:10});}).join('')+
  dot(x0,y0,6,{fill:COL.ink})+arrow(x0,y0,x1,y1,{stroke:COL.guess,w:4,head:12})+label(x0-14,y0+30,'start',{size:20,anchor:'end',fill:COL.ink,halo:true})+
  A.x.map((x,i)=>dot(PL.x(x),PL.y(A.y[i]),c.points.includes(i+1)?10:7,{fill:c.points.includes(i+1)?COL.data:'#9DB6EE'})).join('')+
  path([[PL.x(-1.6),PL.y(yl(-1.6))],[PL.x(2.6),PL.y(yl(2.6))]],{stroke:COL.guess,w:3.5})+
  label(R0,52,'the step used',{size:20})+label(R0,82,k==='all'?'all four points':`point ${k} only`,{size:25,fill:COL.ink,weight:650})+
  label(R0,128,'gradient',{size:20})+label(R0,158,vec(c.grad),{size:24,fill:GRAD,weight:650})+
  label(R0,204,'loss on all four points',{size:20})+label(R0,236,`${num(A.loss0)} → ${num(c.loss1)}`,{size:27,fill:COL.loss,weight:700})+
  (k==='all'?'':label(R0,282,`loss on point ${k}`,{size:20})+label(R0,312,`${num(c.own0)} → ${num(c.own1)}`,{size:24,fill:COL.loss,weight:650}));
}
const sgdAlt='Lecture 5’s four-point line fit: the step from the start in the (b, w) plane, and the fitted line before and after it';
register('p2-sgd',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 1152 ${HS}" role="img" aria-label="${esc(sgdAlt)}">${sgdStatic}<g class="live">${inner}</g></svg></div>`;
 if(ctx.printing){el.innerHTML=frame(sgdLive('2'));return;}
 el.innerHTML=frame('')+`<div class="w-controls"><span>the step uses</span>${KEYS.map(k=>`<button type="button" data-k="${k}" aria-pressed="${k==='all'}">${k==='all'?'all four points':'point '+k}</button>`).join('')}</div>`;
 const live=el.querySelector('.live');let k='all';
 const show=()=>{live.innerHTML=sgdLive(k);};
 for(const b of el.querySelectorAll('button[data-k]'))b.addEventListener('click',()=>{k=b.dataset.k;
  for(const o of el.querySelectorAll('button[data-k]'))o.setAttribute('aria-pressed',o===b);b.blur();show();});
 show();
});
const C2=CH['2'];
S_('sgd','grad','A gradient from part of the data',{builds:3,from:[5,'#page=47'],
 body:`${widget('p2-sgd',sgdAlt)}${answer(3,`Their average is exactly the full step, ${vec(A.full_step,3)}: each draw is noisy but right on average.`)}`,
 lines:[`Lecture 5’s line fit, one step of size ${A.eta} from (b, w) = ${vec(A.theta0,0)}. With all four points the loss falls. Which single point would you rather not draw?`,
  'Click point 2: its own loss falls, but the loss on all four rises. The grey arrows are the other one-point steps. Where is their average?',
  'One draw can go uphill while the average goes downhill. A larger batch shrinks the spread.'],
 ask:['Which point pulls hardest, and which is the unlucky one?','How can a correct gradient of point 2 raise the total loss, and where is the average of the four arrows?','Why not always use all the data, then?'],
 expected:[`Point 4 pulls hardest (gradient ${vec(A.per_point[3],0)}). Point 2 is the unlucky one: it is nearly fitted already, so its step barely helps the rest.`,
  'It is the exact gradient of point 2’s loss, not of the total. Lowering b helps points 1 and 2 but hurts points 3 and 4, which sit far above the line. The four grey and purple arrows average to the full-batch step.',
  'One full gradient costs a pass over all the data. With millions of examples, a mini-batch of a few hundred gives a good enough direction many times more cheaply.'],
 point:'A mini-batch gives a cheap estimate of the gradient that is right on average, but a single step can go uphill.',
 misconception:'That SGD is wrong whenever the loss goes up on a step. One draw follows one point’s surface; only the average follows the full loss.',
 demo:'Start on “all four points”, then click point 2 and point 4 in turn. Read the loss on all four points each time.',
 transition:'Even an exact gradient can make slow progress. Lecture 7’s ravine shows why.',source:`Example A of Lecture 5. ${SCRIPT}`});

// ---------------------------------------------------------------------------------------- 2.4 momentum and Adam
const O=D.opt,MS=O.methods,OCOL={gd:MUTED,momentum:COL.data,rmsprop:COL.amber,adam:COL.guess};
chap('2.4','Update: momentum and Adam');
const OB={xMin:-1.6,xMax:1.4,yMin:-1.45,yMax:.55};
const HO=320,PO=panel({W:780,H:HO,L:72,R:16,T:36,B:54,...OB,ticksX:[-1.5,-1,-.5,0,.5,1],ticksY:[-1,-.5,0,.5],xLabel:'θ₁',yLabel:'θ₂'});
const to=t=>[PO.x(t[0]),PO.y(t[1])];
const setting=M=>[`η = ${M.eta}`,M.key==='momentum'?`β = ${M.b1}`:M.b1!=null?`β₁ = ${M.b1}`:'',M.b2!=null?`β₂ = ${M.b2}`:''].filter(Boolean).join(', ');
const optStatic=clip('p2-opt-clip',PO.x(OB.xMin),PO.y(OB.yMax),PO.x(OB.xMax),PO.y(OB.yMin))+PO.axes+
 `<g clip-path="url(#p2-opt-clip)">${Object.values(O.contours).map(pts=>path(pts.map(to),{stroke:AX,w:1.5})).join('')}</g>`+
 dot(...to(O.theta_star),9,{fill:COL.green})+label(to(O.theta_star)[0],to(O.theta_star)[1]+34,'minimum',{size:20,anchor:'middle',fill:COL.green,weight:650,halo:true})+
 dot(...to(O.theta0),8,{fill:COL.ink})+label(to(O.theta0)[0]+14,to(O.theta0)[1]-12,`start, loss ${num(O.loss0)}`,{size:20,fill:COL.ink,halo:true})+
 label(820,48,'loss after',{size:20});
function optLive(k,t){
 const R0=820;
 return MS.map(M=>{const pts=O.paths[M.key].slice(0,t+1).map(to),on=M.key===k;
   return path(pts,{stroke:OCOL[M.key],w:on?4.5:2.2,op:on?1:.55})+(on?dot(...pts.at(-1),8,{fill:OCOL[M.key]}):'');}).join('')+
  label(R0+90,48,`${t} step${t===1?'':'s'}`,{size:20,fill:COL.ink,weight:650})+
  MS.map((M,i)=>{const on=M.key===k,y=92+i*42;return label(R0,y,M.name,{size:23,fill:OCOL[M.key],weight:on?750:''})+
   label(1140,y,num(O.loss[M.key][t]),{size:23,anchor:'end',fill:on?COL.loss:COL.ink,weight:on?750:''});}).join('')+
  line(R0,92+MS.length*42-24,1140,92+MS.length*42-24,{stroke:AX})+
  label(R0,92+MS.length*42+12,setting(MS.find(M=>M.key===k)),{size:20,fill:COL.ink});
}
const optAlt='Lecture 7’s ravine: gradient descent, momentum, RMSProp and Adam from the same start, with the loss of each after the chosen number of steps';
register('p2-opt',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 1152 ${HO}" role="img" aria-label="${esc(optAlt)}">${optStatic}<g class="live">${inner}</g></svg></div>`;
 if(ctx.printing){el.innerHTML=frame(optLive('adam',O.steps));return;}
 el.innerHTML=frame('')+`<div class="w-controls">${MS.map(M=>`<button type="button" data-m="${M.key}" aria-pressed="${M.key==='gd'}">${M.name}</button>`).join('')}${slider('t','steps',0,O.steps,1,O.steps)}</div>`;
 const live=el.querySelector('.live'),range=el.querySelector('[data-k="t"]'),out=el.querySelector('[data-o="t"]');let k='gd';
 const show=()=>{out.textContent=range.value;live.innerHTML=optLive(k,+range.value);};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());
 for(const b of el.querySelectorAll('button[data-m]'))b.addEventListener('click',()=>{k=b.dataset.m;
  for(const o of el.querySelectorAll('button[data-m]'))o.setAttribute('aria-pressed',o===b);b.blur();show();});
 show();
});
const OL=key=>num(O.loss[key][O.steps]);
S_('opt-play','grad','Same ravine, four optimizers',{builds:2,from:[7,'#page=105'],
 body:`${widget('p2-opt',optAlt)}${answer(2,`Momentum averages the pushes: across the valley they cancel. RMSProp scales each coordinate by its recent size. Adam does both.`)}`,
 lines:[`Lecture 7’s ravine is ${O.steeper} times steeper across θ₂ than along θ₁. Each method took ${O.steps} steps from the same start. Why does gradient descent zigzag?`,
  'With a memory of past gradients, all three beat the plain step. Each uses its own step size from Lecture 7.'],
 ask:['One step size has to serve both directions. What goes wrong?','What does each method remember between steps?'],
 expected:[`A step small enough to stay stable across the steep θ₂ is tiny along the flat θ₁, so gradient descent crosses the valley again and again and creeps along it.`,
  'Momentum keeps a running average of the gradients (direction). RMSProp keeps a running average of their squares (the size of each coordinate). Adam keeps both, with a correction for starting at zero.'],
 point:'Momentum remembers direction, RMSProp remembers scale, and Adam remembers both.',
 misconception:'That Adam needs no tuning. Each method here has its own step size from the lecture; with the same η they would rank differently.',
 demo:`The figure opens at ${O.steps} steps. Pick gradient descent, drag back to 0 and forward again to watch it cross the valley; then do the same for momentum. The right side lists the losses: gradient descent ${OL('gd')}, momentum ${OL('momentum')}, Adam ${OL('adam')}, RMSProp ${OL('rmsprop')}.`,
 transition:'Backprop on scalars gave these gradients. A real layer works on vectors.',source:`Example B of Lectures 5 and 7, with each method’s step size from Lecture 7. ${SCRIPT}`});

// ---------------------------------------------------------------------------------------- 2.5 autograd on vectors
const V=D.vjp,[J0,J1]=V.J,G=V.g;
chap('2.5','Update: autograd on vectors');
const VX=[[215,78],[215,262]],VZ=[[470,78],[470,262]];
const vnode=(c,t,col)=>`<circle cx="${c[0]}" cy="${c[1]}" r="34" fill="#fff" stroke="${col}" stroke-width="2.5"/>`+label(c[0],c[1]+9,t,{size:26,anchor:'middle',fill:col,weight:650});
const along=(a,b,f=.3)=>{const x0=a[0]+34,x1=b[0]-34;return [x0+f*(x1-x0),a[1]+f*(b[1]-a[1])+8];};
const wt=(x,y,v)=>label(x,y,num(v,0),{size:23,anchor:'middle',fill:COL.data,weight:700,halo:true});
const part=(i)=>`${num(G[0],0)}·${num(V.J[0][i],0)} + ${num(G[1],0)}·${V.J[1][i]<0?'('+num(V.J[1][i],0)+')':num(V.J[1][i],0)}`;
const vjpFig=fig(590,322,
 [[0,0],[1,0],[0,1],[1,1]].map(([i,j])=>line(VX[i][0]+34,VX[i][1],VZ[j][0]-34,VZ[j][1],{stroke:MUTED,w:2.5})).join('')+
 wt((VX[0][0]+VZ[0][0])/2,VX[0][1]-10,J0[0])+wt((VX[0][0]+VZ[0][0])/2,VX[1][1]+30,J1[1])+wt(...along(VX[0],VZ[1]),J1[0])+wt(...along(VX[1],VZ[0]),J0[1])+
 vnode(VX[0],'x₁',COL.data)+vnode(VX[1],'x₂',COL.data)+vnode(VZ[0],'z₁',COL.ink)+vnode(VZ[1],'z₂',COL.ink)+
 label((VX[0][0]+VZ[0][0])/2,22,'local slopes ∂z/∂x',{size:20,anchor:'middle',fill:COL.data})+
 gat(2,label(VZ[0][0]+46,VZ[0][1]+8,`g = ${num(G[0],0)}`,{size:23,fill:GRAD,weight:700})+label(VZ[0][0]+46,VZ[1][1]+8,`g = ${num(G[1],0)}`,{size:23,fill:GRAD,weight:700})+
  label(VX[0][0]-44,VX[0][1]+8,`${part(0)} = ${num(V.JTg[0],0)}`,{size:21,anchor:'end',fill:GRAD,weight:700})+
  label(VX[1][0]-44,VX[1][1]+8,`${part(1)} = ${num(V.JTg[1],0)}`,{size:21,anchor:'end',fill:GRAD,weight:700})),
 'A layer with inputs x1, x2 and outputs z1 = x1 + x2, z2 = 2 x1 − x2; the gradient arriving at z is sent back to x along the four arrows');
const mat=M=>R`\begin{pmatrix}${M.map(r=>r.map(v=>tn(v,0)).join('&')).join(R`\\`)}\end{pmatrix}`,colv=v=>mat(v.map(x=>[x]));
const JT=[[J0[0],J1[0]],[J0[1],J1[1]]];
S_('vjp','grad','Backprop through a layer multiplies by Jᵀ',{builds:3,from:deck(8,'Autograd slides',141),
 body:cols(vjpFig,`<div class="p2-stack">${eq(R`\begin{pmatrix}z_1\\z_2\end{pmatrix}=\underbrace{${mat(V.J)}}_{J}\begin{pmatrix}x_1\\x_2\end{pmatrix}`)}
  ${at(2,eq(R`\nabla_{x}L=J^{\top}g=${mat(JT)}${colv(G)}=${colv(V.JTg)}`))}
  ${at(3,`<div class="code-xs">${code(`x = torch.tensor([${V.x.map(v=>v+'.').join(', ')}], requires_grad=True)\nz = torch.stack([x[0] + x[1], 2*x[0] - x[1]])\nz.backward(torch.tensor([${G.map(v=>v+'.').join(', ')}]))\nx.grad          # tensor([${V.JTg.map(v=>v+'.').join(', ')}])`)}</div>`)}</div>`,'0.95fr 1.05fr'),
 lines:[`Lecture 8’s layer. The slopes on the arrows are the entries of the Jacobian J. If the gradient (${G.join(', ')}) arrives at z, what reaches x?`,
  'Each input adds up what every output sends back along its arrows. Written as matrices, that sum is Jᵀ times the gradient.',
  `Autograd computes Jᵀg directly and never builds J. For a layer of ${V.units.toLocaleString('en-US')} units, J would have about ${num(V.entries/1e7,1)} × 10⁷ entries.`],
 ask:['Which arrows carry gradient back to x₁?','Why the transpose?','What does z.backward(g) need g for?'],
 expected:['The two arrows leaving x₁: to z₁ with slope 1 and to z₂ with slope 2. x₁ was used twice, so both contributions add (the branch rule of Lecture 6).',
  'J has a row per output and a column per input. Going backward we sum over outputs for each input, which is a column of J, so a row of Jᵀ.',
  'z is a vector, so backward needs the gradient arriving at z (the seed). For a scalar loss, PyTorch seeds it with 1.'],
 point:'For a layer, backprop is a vector-Jacobian product: the gradient at the output times Jᵀ.',
 misconception:'That autograd forms Jacobians. It only ever computes products Jᵀg, which cost about as much as the forward pass.',
 transition:'Backprop and the optimizer need one more thing: outputs that do not overflow.',source:`Example: Lecture 8 (the autograd deck). ${SCRIPT}`});

// ---------------------------------------------------------------------------------------- 2.6 keep the logits
const SM=D.softmax,[S1,S2,S3]=SM.rows;
chap('2.6','Output: keep the logits');
const fv=(v,n)=>typeof v==='string'?`<span class="mono loss">${v}</span>`:num(v,n);
const smRow=(r,b,note)=>`<tr data-at="${b}"><td>${note}</td><td>(${r.z.map(v=>num(v,0)).join(', ')})</td><td>${r.exp.map(v=>fv(v,2)).join(', ')}</td><td class="q">${r.p.map(v=>fv(v,3)).join(', ')}</td></tr>`;
S_('logits','grad','Softmax overflows; the loss takes logits',{builds:3,from:[9,'#page=32'],
 body:`${eq(R`\qcol{p_i}=\frac{e^{z_i}}{\sum_j e^{z_j}}`)}<div class="p2-sm"><div class="table-scroll"><table><thead><tr><th></th><th>logits z</th><th>e<sup>z</sup></th><th>${kp('p')}</th></tr></thead><tbody>
  ${smRow(S1,1,'small')}${smRow(S2,2,'large, direct')}${smRow(S3,3,`minus ${SM.max}`)}</tbody></table></div></div>
  ${at(3,`<div class="code-sm">${code(`loss = F.cross_entropy(logits, y)   # takes logits, subtracts the max inside`)}</div>`)}`,
 lines:[`Softmax turns logits into probabilities. For (${S1.z.join(', ')}) it is easy. What should (${S2.z.join(', ')}) give?`,
  `Computed directly in float32, e to the ${SM.max} overflows to inf, and inf divided by inf is nan.`,
  `Subtracting the same number from every logit leaves softmax unchanged. Subtracting the largest gives the same three probabilities, and nothing overflows.`],
 ask:[`Before computing: is softmax(${S2.z.join(', ')}) different from softmax(${S1.z.join(', ')})?`,'Where exactly does it break?','So what should a model output, and where should the softmax go?'],
 expected:['No. Only differences between logits matter, and both have gaps of 1.',
  `float32 holds numbers up to about e to the ${SM.fp32_max_exp}; e to the ${SM.max} becomes inf, and inf / inf is nan.`,
  `The raw logits. F.cross_entropy applies a log-softmax that subtracts the max for us; here it gives −ln ${num(S3.p[0],3)} = ${num(SM.ce,3)} for class 1.`],
 point:'Keep the model’s output as logits and let the loss apply the softmax in a stable way.',
 misconception:'That adding softmax as the last layer and then calling cross_entropy is fine. It applies softmax twice and the gradients shrink.',
 transition:'The last knob on the step is its size over time.',source:`Example: Lecture 9 (valid outputs). Computed in float32 by scripts/sci/p2.py.`});

// ---------------------------------------------------------------------------------------- 2.7 a schedule
const SC=D.sched;
chap('2.7','Update: a schedule for the step size');
const WS=700,HSC=380,PS=panel({W:WS,H:HSC,L:84,R:24,T:30,B:52,xMin:0,xMax:SC.T,yMin:0,yMax:SC.eta0*1.2,ticksX:[0,SC.T/4,SC.T/2,SC.T*3/4,SC.T],ticksY:[0,SC.eta0/2,SC.eta0],fmtX:v=>v?v:'step 0'});
const sp=t=>[PS.x(t),PS.y(SC.eta[t])];
const schedStatic=PS.axes+path(SC.eta.map((_,t)=>sp(t)),{stroke:COL.guess,w:2,op:.3})+
 label(PS.x(0)+16,PS.y(SC.eta0)-16,`η₀ = ${SC.eta0}`,{size:20,fill:COL.ink})+label(PS.x(SC.T)-6,PS.y(SC.eta_min)-14,`η<tspan dy="5" font-size="15">min</tspan><tspan dy="-5"> = ${SC.eta_min}</tspan>`,{size:20,anchor:'end',fill:COL.ink});
function schedLive(t){
 const [x,y]=sp(t);
 return path(SC.eta.slice(0,t+1).map((_,i)=>sp(i)),{stroke:COL.guess,w:4.5})+line(x,PS.y(0),x,PS.y(SC.eta0*1.2),{stroke:AX,w:1.5,dash:'4 5'})+dot(x,y,8,{fill:COL.guess})+
  label(WS-24,PS.y(SC.eta0)+6,`at step ${t}`,{size:22,anchor:'end'})+label(WS-24,PS.y(SC.eta0)+46,`η = ${num(SC.eta[t],4)}`,{size:30,anchor:'end',fill:COL.guess,weight:700});
}
const schedAlt=`A cosine schedule: the step size falls from ${SC.eta0} to ${SC.eta_min} over ${SC.T} steps`;
register('p2-sched',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${WS} ${HSC}" role="img" aria-label="${esc(schedAlt)}">${schedStatic}<g class="live">${inner}</g></svg></div>`;
 if(ctx.printing){el.innerHTML=frame(schedLive(SC.mid));return;}
 el.innerHTML=frame('')+`<div class="w-controls">${slider('t','step t',0,SC.T,1,0)}</div>`;
 const live=el.querySelector('.live'),range=el.querySelector('[data-k="t"]'),out=el.querySelector('[data-o="t"]');
 const show=()=>{out.textContent=range.value;live.innerHTML=schedLive(+range.value);};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());show();
});
const [BN0,BN1]=SC.bounce;
S_('schedule','grad','Large steps early, small steps late',{builds:2,from:deck(9,'LR schedule slides',6),
 body:cols(widget('p2-sched',schedAlt),`<div class="p2-stack">${sm(eq(R`\eta_t=\eta_{\min}+\tfrac12(\eta_0-\eta_{\min})\Big(1+\cos\frac{\pi t}{T}\Big)`))}
  ${answer(2,`Near the minimum a mini-batch gradient is mostly noise, and η sets how far the noise throws θ: a noise of ${SC.noise} moves it ${BN0.move} at η = ${BN0.eta}, ${BN1.move} at η = ${BN1.eta}.`)}</div>`,'1.25fr 1fr'),
 lines:[`Lecture 9’s cosine schedule over T = ${SC.T} steps. Drag the step: halfway, η is ${num(SC.eta[SC.mid],4)}. Why would we want the step to shrink at all?`,
  'Large steps cover ground early; small steps settle late. The schedule changes only η, so it works with any optimizer.'],
 ask:['What goes wrong with a constant large η late in training?','When would you add warmup in front of this curve?'],
 expected:['The loss stops falling and bounces: near the minimum the gradient is mostly mini-batch noise, and a large step turns that noise into large moves.',
  'When the first steps are fragile, for example Adam on a large transformer: start η near zero and raise it over the first few hundred steps, then decay.'],
 point:'The optimizer chooses the direction of each step; the schedule chooses its size over time.',
 misconception:'That a decaying η means the model learns less late in training. It means the noisy part of each step shrinks, so the parameters settle.',
 demo:`Drag from 0 to ${SC.T}: the curve is flat at both ends and steepest in the middle. Stop at ${SC.mid} to read ${num(SC.eta[SC.mid],4)}.`,
 transition:'One question checks this part.',source:`Example: Lecture 9 (learning-rate schedules). ${SCRIPT}`});

// ---------------------------------------------------------------------------------------- check and recap
chap();
S_('check-2','grad','Check: the loss went up',{builds:2,nochap:true,
 body:`${cols(`<p class="p2-q">We train Lecture 5’s line fit with SGD, one point per step, η = ${A.eta}. The first two lines it prints are on the right. Is the code broken?</p>`,
   code(`step 0   loss ${num(A.loss0).replace('−','-')}\nstep 1   loss ${num(C2.loss1).replace('−','-')}`,'text'),'1.6fr 1fr')}
  ${answer(2,`Not necessarily. That step used point 2 alone: its own loss fell, ${num(C2.own0)} to ${num(C2.own1)}, while points 3 and 4 got worse. Judge the trend over many steps; if it keeps rising, lower η.`)}`,
 lines:['Two ideas from this part answer it. Talk to your neighbour for a minute.',
  'One noisy step is not a trend.'],
 ask:['Is the code broken?','What would you print to tell noise from a real problem?'],
 expected:[`Not necessarily: a one-point gradient is an unbiased but noisy estimate of the full gradient. The full-batch step from the same start lowers the loss to ${num(CH.all.loss1)}.`,
  'The loss on all the training data (or a running average of the batch losses) every few hundred steps. If that trend rises, lower η or check the code; one batch is too noisy to judge.'],
 point:'A mini-batch step can go uphill; the trend over many steps is what to watch.',
 misconception:'That every SGD step must lower the loss.',
 transition:'Now this part on the recipe card.',source:`Example A of Lecture 5. ${SCRIPT}`});

recap(2,['Backprop gives every gradient in one sweep; a layer multiplies by Jᵀ.',
  'A mini-batch gradient is cheap and right on average, but noisy.',
  'Momentum and Adam shape each step; a schedule shrinks it.'],
 'Part 3, where a low training loss turns out to be the wrong thing to celebrate.',
 {card:{...RECIPES.mlp,step:'backprop on a mini-batch; Adam and a schedule set the step'},title:'MLP classifier',
  ask:'Which row of the card did this part change, and which rows did it leave alone?',
  expected:'Only Update. The data, model, output, loss and test-time rows are the same as in Part 1; the logits slide only moved where the softmax is applied.',
  point:'The Update row is the same for every later model: backprop on a mini-batch, an optimizer and a schedule.',
  transition:'Part 3: generalization.'});
