// Part 3: generalization (Lecture 10). The ten-point polynomial lab of the lecture (seed 21): the degree 9 fit that
// interpolates, the lambda sweep that tames it, four regularisers on their worked examples, the CIFAR-10 ablation,
// the three splits and early stopping. Every number comes from data/p3.json (scripts/sci/p3.py).
import {S_,O,p,ps,small,cols,stack,sm,eq,m,R,tn,at,answer,data,need,word,Word,css,kt,kp,kl,partBreak,recap,chap,RECIPES,
 fig,label,line,arrow,path,dot,panel,chart,gat,widget,register,slider,num,big,COL,MUTED,AX,esc} from '../kit.js';

const D=need(await data('p3'),'poly','sweep','best','early','wd','dropout','smooth','augment','recorded');
const P=D.poly,I=P.interp,SW=D.sweep,BEST=SW[D.best],ES=D.early,EB=ES.rows[ES.best],C=D.recorded.cifar;
const SOURCE='The ten-point lab of Lecture 10 (seed 21), ported from its deck. Numbers: scripts/sci/p3.py.';

css(`.p3-regs{display:grid;grid-template-columns:1fr 1fr;gap:16px 26px}
.p3-reg{border:2px solid var(--line);border-radius:12px;background:#fff;padding:10px 18px 8px;min-width:0}
.p3-reg h3{font-size:23px;font-weight:650;margin:0 0 2px}.p3-reg h3 span{color:var(--muted);font-weight:500;font-size:19px;margin-left:8px}
.p3-reg .display{font-size:24px}.p3-reg .display .katex-display{margin:.15em 0}
.p3-reg p{font-size:21px;line-height:1.35;color:var(--ink)}
.p3-stats .display{font-size:26px}`);

// ---------------------------------------------------------------------------------------- small helpers
const SUP={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};
const ex=v=>Math.round(Math.log10(v));
/** λ as the slides write it: 0, 1, or a power of ten with a superscript (text) or ^{} (TeX). */
const lamTxt=v=>v===0?'0':ex(v)===0?'1':'10'+String(ex(v)).split('').map(c=>SUP[c]).join('');
const lamTex=v=>v===0?'0':ex(v)===0?'1':`10^{${ex(v)}}`;
const ring=(x,y,r=7,c=COL.amber)=>`<circle cx="${(+x).toFixed(1)}" cy="${(+y).toFixed(1)}" r="${r}" fill="#fff" stroke="${c}" stroke-width="3"/>`;
const TRUTH_COL='#8E979F';

/** The polynomial panel: axes, the true curve, the ten training points and nine validation points. A fit is drawn
 inside a clip, because the interpolant leaves the panel near the ends. Returns {P, base, fit(curve), clip}. */
function polyPlot(W,H,id,{legend=true,fitName=''}={}){
 const Q=panel({W,H,L:52,R:12,T:legend?50:20,B:44,xMin:-1,xMax:1,yMin:-2,yMax:2,ticksX:[-1,-.5,0,.5,1],ticksY:[-2,-1,0,1,2],fs:20});
 const x0=Q.x(-1),x1=Q.x(1),y0=Q.y(2),y1=Q.y(-2);
 const clip=`<defs><clipPath id="${id}"><rect x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}"/></clipPath></defs>`;
 const pts=c=>P.grid.map((x,i)=>[Q.x(x),Q.y(c[i])]);
 const truth=path(pts(P.truth),{stroke:TRUTH_COL,w:2.5,dash:'7 6'});
 const points=P.x_val.map((x,i)=>ring(Q.x(x),Q.y(P.y_val[i]))).join('')+P.x_train.map((x,i)=>dot(Q.x(x),Q.y(P.y_train[i]),8,{fill:COL.target})).join('');
 let leg='';
 if(legend){const y=22;
  leg=dot(x0+10,y-7,8,{fill:COL.target})+label(x0+26,y,'training points',{size:20,fill:COL.ink})+
   ring(x0+215,y-7)+label(x0+231,y,'validation points',{size:20,fill:COL.ink})+
   line(x0+440,y-7,x0+475,y-7,{stroke:TRUTH_COL,w:2.5,dash:'7 6'})+label(x0+483,y,'sin(πx)',{size:20,fill:COL.ink});}
 return {Q,clip,base:clip+Q.axes+truth+leg,points,fit:(c,o={})=>`<g clip-path="url(#${id})">${path(pts(c),{stroke:COL.guess,w:4,...o})}</g>`,
  name:fitName};
}

// ---------------------------------------------------------------------------------------- the part
partBreak(3,'Doing well on data you have not seen',
 `Lecture 10 asked what happens when a model fits its training data too well. It changed three rows of the recipe: Data, Loss and At test time.`,
 ['measure the error on points the fit never saw',
  'add a penalty on large weights to the loss',
  'add noise to training: dropout, smoothed targets, crops and flips',
  'let validation choose; read the test set once'],
 {line:'By the end you can say why training error is the wrong judge, and which split chooses the strength of a regulariser.',
  ask:'Your network reaches a training loss of nearly zero. Is it a good model?',
  expected:'We cannot tell from the training loss. It has to be measured on data the network did not train on.',
  transition:'Start with the smallest model that shows the problem: a polynomial through ten points.'});

// ---------------------------------------------------------------------------------------- 3.1 the gap
chap('3.1','At test time: the gap');
{
 const W=660,H=410,F=polyPlot(W,H,'p3-clip-a');
 const body=fig(W,H,F.base+gat(2,F.fit(I.curve))+F.points+
   gat(2,label(F.Q.x(-.62),F.Q.y(-1.72),`degree ${P.degree} fit`,{size:21,fill:COL.guess,weight:650,halo:true})),
  `${Word(P.n_train)} noisy points from sin(πx), ${word(P.n_val)} validation points between them, and the degree ${P.degree} polynomial through the training points`);
 S_('overfit','gen',`${Word(P.n_train)} points, ${word(P.n_weights)} weights`,{builds:3,from:10,
  body:cols(body,`<div class="p3-stats">${stack(
   eq(R`\qcol{\hat y}=w_0+w_1x+\dots+w_9x^9`),
   at(2,eq(R`\text{training error}=\lcol{${tn(I.train)}}`)),
   at(3,eq(R`\text{validation error}=\lcol{${tn(I.val)}}`)),
   at(3,eq(R`\text{size of the weights}\ \|w\|=${tn(I.norm,0)}`)))}</div>`,'1.15fr 1fr'),
  lines:[`${Word(P.n_train)} points from sin(πx) plus noise of size ${P.noise}, and ${word(P.n_val)} validation points at the midpoints. A degree ${P.degree} polynomial has ${word(P.n_weights)} weights. What will its training error be?`,
   `With ${word(P.n_weights)} weights and ${word(P.n_train)} points it passes through every training point, so the training error is ${num(I.train)}.`,
   `Between the points it swings far from sin(πx): the validation error is ${num(I.val)}, about ${num(I.times_noise,0)} times the noise variance ${P.sigma2}.`],
  ask:[`Predict the training error and the validation error of the degree ${P.degree} fit.`,'Is a training error of zero good news?','Where does the large error come from?'],
  expected:[`The training error is zero (${word(P.n_weights)} equations, ${word(P.n_weights)} unknowns), and the validation error is large.`,
   'No. The fit has matched the noise as well as the curve; the training error cannot tell the two apart.',
   `From the swings between the points and near the ends. The weights have to be huge to make them: ‖w‖ = ${num(I.norm,0)}.`],
  point:'A model that matches every training point can still be very wrong between them, and only held-out data shows it.',
  misconception:'That a lower training error always means a better model.',
  transition:'The large weights point to the fix: make large weights cost something.',source:SOURCE});
}

// ---------------------------------------------------------------------------------------- 3.2 the penalty, live
chap('3.2','Loss: a penalty on the weights');
const LW=1152,LH=272,TX=[742,862,990,1110];
const LP=polyPlot(660,LH,'p3-clip-b',{legend:false});
function lamStatic(){
 let g=LP.base+label(TX[0],28,'λ',{size:21,anchor:'middle',fill:COL.ink,weight:650})+label(TX[1],28,'training',{size:21,anchor:'middle',fill:COL.ink,weight:650})+
  label(TX[2],28,'validation',{size:21,anchor:'middle',fill:COL.ink,weight:650})+label(TX[3],28,'‖w‖',{size:21,anchor:'middle',fill:COL.ink,weight:650})+
  line(TX[0]-48,40,TX[3]+38,40,{stroke:AX});
 SW.forEach((s,i)=>{const y=64+i*27;
  g+=label(TX[0],y,lamTxt(s.lam),{size:20,anchor:'middle',fill:COL.ink})+label(TX[1],y,num(s.train,3),{size:20,anchor:'middle',fill:COL.ink})+
   label(TX[2],y,num(s.val,3),{size:20,anchor:'middle',fill:COL.loss})+label(TX[3],y,num(s.norm,1),{size:20,anchor:'middle',fill:COL.ink});});
 return g;
}
function lamLive(i){
 const y=64+i*27;
 return `<rect x="${TX[0]-48}" y="${y-21}" width="${TX[3]+38-TX[0]+48}" height="27" rx="6" fill="#8B2CDE1F"/>`+LP.fit(SW[i].curve)+LP.points;
}
const lamAlt=`The degree ${P.degree} fit at one value of λ, beside the sweep of λ with training error, validation error and the size of the weights`;
register('p3-lambda',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${LW} ${LH}" role="img" aria-label="${esc(lamAlt)}"><g class="live">${inner}</g>${lamStatic()}</svg></div>`;
 if(ctx.printing){el.innerHTML=frame(lamLive(D.best));return;}
 el.innerHTML=frame('')+`<div class="w-controls">${slider('i','penalty strength λ',0,SW.length-1,1,0)}</div>`;
 const live=el.querySelector('.live'),range=el.querySelector('[data-k="i"]'),out=el.querySelector('[data-o="i"]');
 const show=()=>{const i=+range.value;out.textContent='λ = '+lamTxt(SW[i].lam);live.innerHTML=lamLive(i);};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());
 show();
});
S_('reg-play','gen','Make large weights cost something',{builds:2,from:10,
 body:`${sm(eq(R`\lcol{\mathcal J(w)}=\tfrac{1}{2n}\textstyle\sum_i(\qcol{\hat y_i}-\pcol{y_i})^2+\tfrac{\lambda}{2}\|w\|^2`))}
  ${widget('p3-lambda',lamAlt)}
  ${answer(2,`Validation picks λ = ${lamTxt(BEST.lam)}: error ${num(BEST.val,3)}, with ‖w‖ = ${num(BEST.norm,1)}. Training error is always lowest at λ = 0.`)}`,
 lines:[`The loss gets a second term that grows with the weights. Drag λ from 0 upwards. Which column of the table should choose λ?`,
  `Too little λ leaves the swings; too much flattens the curve. Only validation sees both mistakes.`],
 ask:['Which column of the table may choose λ, and why not the others?','What happens at λ = 1?'],
 expected:['Validation. Training error only rises as λ grows, and ‖w‖ only shows that the penalty is working.',
  `The fit is too flat to follow sin(πx): training ${num(SW.at(-1).train,3)} and validation ${num(SW.at(-1).val,3)}. That is underfitting.`],
 point:'A penalty on ‖w‖ trades a little training error for a much smaller validation error, and validation sets its strength.',
 misconception:'That the penalty helps the training error. It always makes the training error worse; it helps on new data.',
 demo:`Start at λ = 0 (the interpolant, ‖w‖ = ${num(I.norm,0)}). Step right one notch at a time and read the validation column; it bottoms out at λ = ${lamTxt(BEST.lam)}, then rises again.`,
 transition:'The penalty is one regulariser. Lecture 10 had three more, each at a different row of the recipe.',source:SOURCE});

// ---------------------------------------------------------------------------------------- 3.3 four regularisers
chap('3.3','Data and Loss: four regularisers');
{
 const w=D.wd,d=D.dropout,s=D.smooth,a=D.augment;
 const vec=v=>`(${v.map(x=>num(x,Number.isInteger(x)?0:2)).join(', ')})`;
 const texVec=v=>`(${v.map(x=>tn(x,Number.isInteger(x)?0:2)).join(',\\,')})`;
 const cardR=(b,name,row,tex,text)=>`<div class="p3-reg"><h3>${name}<span>${row}</span></h3>${at(b,eq(tex)+`<p>${text}</p>`)}</div>`;
 S_('regs','gen','Four ways to prefer a simpler fit',{builds:4,from:10,
  body:`<div class="p3-regs">${[
   cardR(1,'Weight decay','Loss',R`\theta\leftarrow(1-\eta\lambda)\,\theta-\eta\,g`,
    `θ = ${w.theta}, g = ${w.g}, η = ${w.eta}, λ = ${w.lam}: ${m(R`${w.shrink}\times${w.theta}-${tn(w.fit)}=${tn(w.new)}`)}`),
   cardR(2,'Dropout','hidden units',R`\tilde h_i=\frac{m_i}{q}\,h_i`,
    `Keep rate q = ${d.keep}: ${m(R`${texVec(d.h)}\odot${texVec(d.mask)}/${d.keep}=${texVec(d.out)}`)}`),
   cardR(3,'Label smoothing','Loss, its target',R`\pcol{\tilde y}=(1-\epsilon)\,\pcol{y}+\epsilon/K`,
    `K = ${s.K}, ε = ${s.eps}, right class ${s.classes[s.y]}: ${m(R`\pcol{\tilde y}=${texVec(s.target)}`)}`),
   cardR(4,'Crop and flip','Data',R`${a.size}\times${a.size}\ \text{image, padded by}\ ${a.pad}`,
    `${m(R`${a.pad*2+1}\times${a.pad*2+1}`)} crop positions, ${m(R`\times\,${a.flips}`)} flips: ${a.views} views, one label.`)].join('')}</div>`,
  lines:['Weight decay is the penalty of the last slide, seen in the update: shrink θ first, then take the usual step.',
   'Dropout zeroes hidden units at random in training and divides the survivors by the keep rate q, so the average is unchanged.',
   `Label smoothing moves a little of the target to the wrong classes, so the loss stops pushing once ${m(R`\qcol{p_b}`)} reaches ${s.target[s.y]}.`,
   'Augmentation changes the Data row: each epoch sees a fresh crop and flip of every image.'],
  ask:['Where does the factor (1 − ηλ) come from?','Why divide by q during training?','What does the plain one-hot target ask of the logits?','Which transforms would change the label?'],
  expected:[`From the gradient of (λ/2)‖θ‖², which is λθ. One step: ${w.shrink} × ${w.theta} = ${num(w.shrunk)}, then − ${w.eta} × ${w.g} gives ${num(w.new)}.`,
   `So the expected value stays the same: E[h̃₃] = ${d.keep} × ${num(d.out[2],0)} + ${1-d.keep} × 0 = ${num(d.mean3,0)}. At test time we use the full h with no rescaling.`,
   `An infinite gap: p_b → 1 needs the logits to grow without end. With smoothing the push stops at a gap of ln(${s.target[s.y]}/${s.target[0]}) = ${num(s.gap)}.`,
   'A flip that turns a 6 into a 9, or a left turn into a right turn. The transform must keep the label true, or the label must move with it.'],
  point:'Each regulariser adds a preference the ten points alone did not give, at a different row: the update, the hidden units, the target, the data.',
  misconception:'That dropout stays on at test time. model.eval() turns it into the identity.',
  transition:'Which of these helped on real images?',
  source:`Worked numbers as in Lecture 10. The crop is the lecture’s v2.RandomCrop(${a.size}, padding=${a.pad}) with a horizontal flip. Numbers: scripts/sci/p3.py.`});
}

// ---------------------------------------------------------------------------------------- 3.4 the CIFAR-10 ablation
chap('3.4','At test time: what helped');
{
 const W=1152,rows=C.runs,LBL=250,BX=270,BW=720,RH=50,T=6,top=60;
 const x=v=>BX+v/top*BW,H=T+rows.length*RH+44;
 let g=line(BX,T+rows.length*RH,BX+BW,T+rows.length*RH,{stroke:AX});
 for(const v of [0,20,40,60])g+=line(x(v),T+rows.length*RH,x(v),T+rows.length*RH+6,{stroke:AX})+label(x(v),T+rows.length*RH+30,`${v} %`,{size:20,anchor:'middle'});
 rows.forEach((r,i)=>{const y=T+i*RH,isBest=i===C.best,c=isBest?COL.data:'#BCC4CC';
  g+=label(LBL,y+RH/2+8,esc(r.name),{size:23,anchor:'end',fill:COL.ink});
  g+=gat(2,`<rect x="${BX}" y="${y+8}" width="${(x(r.acc)-BX).toFixed(1)}" height="${RH-16}" rx="4" fill="${c}"/>`+
   label(x(r.acc)+12,y+RH/2+8,`${num(r.acc,1)} %`,{size:23,fill:COL.ink,weight:isBest?700:600}));});
 S_('regs-cifar','gen','One change at a time, on CIFAR-10',{builds:2,from:10,
  body:`${fig(W,H,g,`CIFAR-10 test accuracy of five runs that differ in one regulariser each; crop and flip is highest at ${num(rows[C.best].acc,1)} %`)}
   ${small(`Test accuracy. ${big(C.n_train)} training images, the same CNN and ${C.epochs} epochs in every run.`)}
   ${answer(2,`Crop and flip: ${num(rows[C.best].acc,1)} %, ${num(C.gain,1)} points above the baseline. Weight decay (λ = ${C.wd_lam}) changed nothing.`)}`,
  lines:[`Lecture 10 trained one small CNN five times, changing one thing each time. Which change do you expect to help most?`,
   `With only ${big(C.n_train)} images, new views of the data did more than any change to the loss or the network.`],
  ask:['Make a guess before the bars appear: which of the four regularisers wins?','Does this prove weight decay is useless?'],
  expected:['Crop and flip wins by a wide margin.',
   `No. This is one seed and one setting each (λ = ${C.wd_lam}, ε = ${C.eps}); another λ might help, and the small gaps may not survive a second seed.`],
  point:'With scarce data, adding valid views of the data was the strongest regulariser in this run.',
  misconception:'That a regulariser which did not help at one setting never helps. Its strength was not tuned here.',
  transition:'Every number on this slide was a test accuracy. Who was allowed to look at it, and when?',
  source:'Recorded: Lecture 10, “Controlled CIFAR-10 ablations” (lecture7/diagrams/cifar_regularization_experiment.py, one seed), not retrained.'});
}

// ---------------------------------------------------------------------------------------- 3.5 three splits
chap('3.5','Data: three splits');
{
 // Three boxes, one per split, drawn by hand so the names and the points of each split show from the first click;
 // what each split does, and its error at the chosen λ, appear one click at a time.
 const W=1152,bw=340,gx=(W-3*bw)/2,bh=226,H=bh+8;
 const xsTest=Array.from({length:P.n_test},(_,i)=>-1+2*i/(P.n_test-1));
 const S=[[`Training set · ${word(P.n_train)} points`,COL.target,P.x_train,'fits θ','',`training error ${num(BEST.train,3)}`],
  [`Validation set · ${word(P.n_val)} points`,COL.amber,P.x_val,'chooses λ and','when to stop',`validation error ${num(BEST.val,3)}`],
  [`Test set · ${P.n_test} points`,COL.ink,xsTest,'read once,','at the end',`test error ${num(BEST.test,3)}`]];
 let g='';
 S.forEach(([name,c,xs,t1,t2,note],i)=>{const x=i*(bw+gx),y=4,lx=v=>x+28+(v+1)/2*(bw-56),ly=y+84;
  g+=`<rect x="${x+1}" y="${y}" width="${bw-2}" height="${bh}" rx="12" fill="#fff" stroke="${c}" stroke-width="3"/>`+
   label(x+bw/2,y+40,esc(name),{size:23,anchor:'middle',fill:c,weight:700})+line(x+28,ly,x+bw-28,ly,{stroke:AX})+
   xs.map(v=>i===0?dot(lx(v),ly,7,{fill:c}):i===1?ring(lx(v),ly,6,c):`<circle cx="${lx(v).toFixed(1)}" cy="${ly}" r="2.6" fill="${c}"/>`).join('')+
   gat(i+1,label(x+bw/2,y+138,esc(t1),{size:27,anchor:'middle',fill:COL.ink})+(t2?label(x+bw/2,y+170,esc(t2),{size:27,anchor:'middle',fill:COL.ink}):'')+
    label(x+bw/2,y+206,esc(note),{size:21,anchor:'middle',fill:MUTED}));
  if(i)g+=arrow(x-gx+8,y+bh/2,x-8,y+bh/2,{stroke:COL.ink,w:2.5,head:11});});
 S_('splits','gen','Each split has one job',{builds:3,from:10,
  body:`${fig(W,H,g,'Three splits of the ten-point lab: the training set fits the weights, the validation set chooses lambda and when to stop, the test set is read once')}
   ${small(`The ten-point lab, with the fit at λ = ${lamTxt(BEST.lam)}. Each dot is one input x on the interval from ${num(P.grid[0],0)} to ${num(P.grid.at(-1),0)}.`)}`,
  lines:['The training set has one job: gradient steps on it fit the weights θ.',
   `We make every choice (λ, when to stop, which regulariser) on validation. Here it picked λ = ${lamTxt(BEST.lam)}.`,
   'We read the test set once, at the end, for the report. If it changes a choice, it has become a second validation set.'],
  ask:['Why can the training set not choose λ?','Is validation error an honest estimate after we used it to choose?','Why is the test error higher than the validation error here?'],
  expected:['Training error is lowest at λ = 0 by construction; it would always pick the interpolant.',
   'Slightly optimistic: we picked the λ that happened to do best on those nine points. That is why a third split exists.',
   `Validation chose λ to suit its own ${word(P.n_val)} points; the ${P.n_test} test points are new. Test ${num(BEST.test,3)} against validation ${num(BEST.val,3)} is the honest gap.`],
  point:'Training fits θ, validation chooses everything else, and the test set reports once.',
  misconception:'That reporting the best test score over many runs is fine. Choosing by the test score turns the test set into a validation set.',
  transition:'Now use all of it on a curve you will see in every project.',source:SOURCE});
}

// ---------------------------------------------------------------------------------------- check
{
 const W=640,H=392,rows=ES.rows,lo=Math.log10(rows[0].t),hi=Math.log10(rows.at(-1).t),ymax=.5;
 const Q=panel({W,H,L:60,R:16,T:52,B:62,xMin:lo,xMax:hi,yMin:0,yMax:ymax,ticksX:[lo,(lo+hi)/2|0,hi].filter((v,i,a)=>a.indexOf(v)===i),
  ticksY:[0,.1,.2,.3,.4,.5],fmtX:v=>v<3?String(10**v):'10'+String(v).split('').map(c=>SUP[c]).join(''),xLabel:'gradient steps',fs:20});
 const pts=k=>rows.map(r=>[Q.x(Math.log10(r.t)),Q.y(r[k])]);
 const bx=Q.x(Math.log10(EB.t)),by=Q.y(EB.val);
 const g=Q.axes+path(pts('train'),{stroke:COL.target,w:4})+path(pts('val'),{stroke:COL.amber,w:4})+
  label(Q.x(lo)+10,24,'training error',{size:20,fill:COL.target,weight:650})+label(Q.x(lo)+190,24,'validation error',{size:20,fill:COL.amber,weight:650})+
  gat(2,line(bx,Q.y(ymax),bx,Q.y(0),{stroke:COL.ink,w:2,dash:'5 5'})+dot(bx,by,9,{fill:COL.green})+
   label(bx+12,Q.y(.36),`step ${big(EB.t)}`,{size:20,fill:COL.green,weight:650,halo:true})+
   label(bx+12,Q.y(.36)+26,`validation ${num(EB.val,3)}`,{size:20,fill:COL.green,weight:650,halo:true}));
 S_('check-3','gen','Check: the curves part ways',{builds:2,from:10,
  body:cols(fig(W,H,g,`Training and validation error of gradient descent on the ten-point lab over ${big(rows.at(-1).t)} steps`),
   `${p('Training error keeps falling while validation error has started to rise. What do you do?')}
    ${answer(2,`Stop, and restore the step with the lowest validation error: step ${big(EB.t)}, ${num(EB.val,3)}. To do better, add a regulariser and let validation choose its strength.`)}`,'1fr 1fr'),
  lines:[`Gradient descent on the ten-point lab, from w = 0. Training error falls from ${num(rows[0].train,3)} to ${num(rows.at(-1).train,3)}; validation error turns up after step ${big(EB.t)}.`,
   `Restoring step ${big(EB.t)} gives test error ${num(EB.test,3)}, as good as the penalty’s ${num(BEST.test,3)}. Validation made both choices.`],
  ask:['Training error keeps falling while validation error rises. What do you do?','Could we have chosen the step by the test error instead?'],
  expected:[`Early stopping: keep the checkpoint with the lowest validation error (step ${big(EB.t)}), then, if needed, add weight decay, dropout or augmentation, with validation choosing the strength.`,
   'No. Then the test error would no longer be an honest report; it would be a second validation set.'],
  point:'When the curves part, the model has started to fit the noise: stop at the best validation step, then regularise with validation as the judge.',
  misconception:'That we should keep the model from the last epoch. We keep the checkpoint with the lowest validation error.',
  transition:'Here is Part 3 on one card.',source:SOURCE});
}

// ---------------------------------------------------------------------------------------- recap
recap(3,[
 `Training error is the wrong judge: ${num(I.train)} on the ${word(P.n_train)} points, ${num(I.val)} between them.`,
 'A penalty, dropout, smoothed targets and new views add a preference the data did not give.',
 `Validation chose λ = ${lamTxt(BEST.lam)} and when to stop; the test set was read once.`],
 'the model has been an MLP. Language needs one that can look back at earlier tokens.',
 {card:{...RECIPES.mlp,data:`three splits; crops and flips add views`,
   loss:`${m(R`\lcol{-\textstyle\sum_k\pcol{\tilde y_k}\ln p_k}+\tfrac{\lambda}{2}\|\theta\|^2`)}`,
   test:`dropout off; the class with the largest ${m(R`\qcol{p}`)}; test read once`},
  title:'MLP classifier, regularised',
  ask:'Which rows of the card did Part 3 change, and which did it leave alone?',
  expected:'Data (splits and augmentation), Loss (the penalty and smoothed targets) and At test time (dropout off, test read once). The model and the update rule stayed the same, apart from the shrink factor weight decay adds.',
  point:'Generalization changed three rows of the recipe and left the model alone.',
  transition:'Part 4: attention and language.'});
