// Part 1: Foundations (Lectures 1 to 4). Every loss of the course is the negative log of a likelihood, a prior on the
// weights adds a penalty, and layers of ReLUs let the model bend. Every number comes from data/p1.json
// (scripts/sci/p1.py).
import {S_,chap,partBreak,recap,RECIPES,p,ps,small,cols,stack,sm,eq,m,R,at,only,answer,table,code,data,need,word,Word,css,kt,kp,kl,tn,
 fig,label,line,path,dot,arrow,panel,bars,widget,register,slider,gat,gonly,num,COL,MUTED,AX,esc} from '../kit.js';

const D=need(await data('p1'),'coin','ce','gauss','map','xor','hinges','check');
const C=D.coin,CE=D.ce,G=D.gauss,M=D.map,X=D.xor,HG=D.hinges,K=D.check;
const SRC=n=>`${n} Numbers: scripts/sci/p1.py.`;
const idx=(arr,v)=>arr.findIndex(t=>Math.abs(t-v)<1e-9);

css(`.p1-tosses{display:grid;grid-template-columns:repeat(5,62px);gap:10px;margin:2px 0 6px}
.p1-tosses div{height:62px;border-radius:12px;border:3px solid var(--data);display:flex;align-items:center;justify-content:center;font-size:30px;font-weight:700;color:var(--data);background:#fff}
.p1-tosses div.h{background:var(--data);color:#fff}
.scene:not([data-build="1"]) .p1-v1,.scene:not([data-build="2"]) .p1-v2,.scene:not([data-build="3"]) .p1-v3{display:none}
.scene[data-build="1"] .p1-b2{visibility:hidden}
.p1-dist td:first-child{white-space:nowrap}
.p1-log .code{font-size:24px;margin:0}
.p1-left .katex-display>.katex{text-align:left}
.p1-dist table{font-size:23px}.p1-dist td,.p1-dist th{white-space:nowrap;padding:9px 12px}`);

// ---------------------------------------------------------------------------------------- the part opens
partBreak(1,'Where the loss comes from',
 'Every loss in this course was read off a probability model. This part recalls how, and builds the first network.',
 ['Pick a distribution for the output and write the likelihood of the data.','Take the negative log: that is the loss.',
  'A prior on the weights adds a penalty.','Stack linear layers and ReLUs to get a network.'],
 {line:'By the end of this part you can say where cross-entropy and squared error come from.',
  ask:'Where did cross-entropy come from? Was it chosen because it works well?',
  expected:'It was not picked from a list. It is the negative log-likelihood of a categorical distribution, and every other loss of the course is read off a distribution the same way.',
  transition:'Start with the smallest likelihood we wrote: ten tosses of a coin.'});

// ---------------------------------------------------------------------------------------- 1.1 the coin
chap('1.1','Loss: from a likelihood');
const CW=620,CH=390,CX={W:CW,H:CH,L:66,R:22,T:78,B:68,xMin:0,xMax:1},LMAX=1.4,NMAX=16;
const P0=panel({...CX,yMax:1,ticksX:[0,.2,.4,.6,.8,1],xLabel:'θ, the chance of heads'});
const PL=panel({...CX,yMax:LMAX,ticksY:[0,.5,1],yLabel:'L(θ), in thousandths'});
const PN=panel({...CX,yMax:NMAX,ticksY:[0,4,8,12,16],yLabel:'−ln L(θ), in nats'});
const yAxis=Pn=>Pn.axes;                                   // the y panels draw only their own ticks and label
const ci=C.cands.map(c=>idx(C.theta,c)),i0=idx(C.theta,C.start);
const curveL=C.theta.map((t,i)=>[PL.x(t),PL.y(C.lik_milli[i])]);
const curveN=C.theta.map((t,i)=>[t,C.nll[i]]).filter(([,v])=>v<=NMAX).map(([t,v])=>[PN.x(t),PN.y(v)]);
const coinStatic=P0.axes+
 C.cands.map(c=>line(P0.x(c),P0.y(0),P0.x(c),CX.T,{stroke:AX,w:2,dash:'3 6'})).join('')+
 gonly(1,C.cands.map(c=>label(P0.x(c),CX.T-10,`θ = ${c}?`,{size:21,anchor:'middle',fill:COL.ink,weight:600,halo:true})).join(''))+
 gonly(2,yAxis(PL)+path(curveL,{stroke:COL.guess,w:3.5})+ci.map((i,j)=>dot(PL.x(C.theta[i]),PL.y(C.lik_milli[i]),6,{fill:'#fff',stroke:COL.guess,w:3})+
  label(PL.x(C.theta[i])+(j?16:-20),PL.y(C.lik_milli[i])+(j?-12:2),num(C.lik_milli[i]),{size:21,anchor:j?'start':'end',fill:COL.guess,weight:650,halo:true})).join(''))+
 gonly(3,yAxis(PN)+path(curveN,{stroke:COL.loss,w:3.5})+ci.map((i,j)=>dot(PN.x(C.theta[i]),PN.y(C.nll[i]),6,{fill:'#fff',stroke:COL.loss,w:3})+
  label(PN.x(C.theta[i])+(j?-14:0),PN.y(C.nll[i])+(j?-14:34),num(C.nll[i]),{size:21,anchor:j?'end':'middle',fill:COL.loss,weight:650,halo:true})).join(''));
function coinLive(i){
 const t=C.theta[i],th=`θ = ${num(t)}`,rx=CW-CX.R,ry=30;
 return line(P0.x(t),P0.y(0),P0.x(t),CX.T,{stroke:COL.guess,w:2.5})+
  `<g class="p1-v1">${label(rx,ry,th,{size:23,anchor:'end',fill:COL.ink,weight:650})}</g>`+
  `<g class="p1-v2">${dot(PL.x(t),PL.y(Math.min(C.lik_milli[i],LMAX)),9,{fill:COL.guess})}${label(rx,ry,`${th}:  L = ${num(C.lik[i],6)}`,{size:23,anchor:'end',fill:COL.guess,weight:650})}</g>`+
  `<g class="p1-v3">${C.nll[i]<=NMAX?dot(PN.x(t),PN.y(C.nll[i]),9,{fill:COL.loss}):''}${label(rx,ry,`${th}:  −ln L = ${num(C.nll[i])}`,{size:23,anchor:'end',fill:COL.loss,weight:650})}</g>`;
}
const coinAlt=`The likelihood and the negative log-likelihood of the ten tosses as functions of θ, lowest loss at θ = ${C.mle}`;
register('p1-coin',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${CW} ${CH}" role="img" aria-label="${esc(coinAlt)}">${coinStatic}<g class="live">${inner}</g></svg></div>`;
 if(ctx.printing){el.innerHTML=frame(coinLive(ci[0]));return;}
 el.innerHTML=frame('')+`<div class="w-controls">${slider('t','θ',0,C.theta.length-1,1,i0)}</div>`;
 const live=el.querySelector('.live'),range=el.querySelector('[data-k="t"]'),out=el.querySelector('[data-o="t"]');
 const show=()=>{out.textContent=num(C.theta[+range.value]);live.innerHTML=coinLive(+range.value);};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());
 show();
});
S_('nll-coin','found','The loss is a negative log-likelihood',{builds:3,from:1,
 body:cols(`<div class="p1-left">`+stack(`<div class="p1-tosses" role="img" aria-label="${esc(C.seq.join(' '))}">${C.seq.map(s=>`<div class="${s==='H'?'h':''}">${s}</div>`).join('')}</div>`,
   at(2,sm(eq(R`L(\theta)=\theta^{${C.nH}}\,(1-\theta)^{${C.nT}}`))),
   at(3,sm(eq(R`\lcol{-\ln L(\theta)}=-${C.nH}\ln\theta-${C.nT}\ln(1-\theta)`),eq(R`\text{lowest at }\hat\theta=\tfrac{${C.nH}}{${C.n}}=${tn(C.mle,1)}`))))+'</div>',
  widget('p1-coin',coinAlt),'0.82fr 1.18fr'),
 lines:[`Lecture 1’s ${word(C.n)} tosses: ${word(C.nH)} heads, ${word(C.nT)} tails. Drag θ, the coin’s chance of heads. Which θ would you bet on: ${C.cands[0]}, ${C.cands[1]}, or another?`,
  `The likelihood scores each θ by how probable it makes these tosses. At ${C.cands[0]} they are ${num(C.ratio)} times as likely as at ${C.cands[1]}.`,
  `Take the negative log. The product becomes a sum and the best θ becomes the lowest point: ${num(C.nll_min)} nats at θ = ${C.mle}.`],
 ask:['Which θ would you bet on, and why?','Why do we multiply the ten probabilities?','Why take the log at all, if the best θ does not move?'],
 expected:[`${C.mle}, because ${word(C.nH)} of the ${word(C.n)} tosses were heads. Hold that guess; the next two clicks justify it.`,
  'The tosses are independent, so the probability of all ten is the product of the ten probabilities.',
  'The log is increasing, so the best θ stays put. A sum of logs does not underflow the way a product of many small numbers does, and its gradient is a sum over examples, which is what minibatches use.'],
 point:'Training a model is finding the θ under which the data are most probable, written as a loss: the negative log-likelihood.',
 misconception:'That the likelihood is a distribution over θ. It is a score for each θ; its values need not sum to 1 over θ.',
 demo:`Leave θ at ${C.start} at click 1 and let the class bet. At click 2 drag to ${C.cands[0]}; at click 3 drag to either end to show how fast the loss grows.`,
 transition:'A network does the same thing, one example at a time.',source:SRC('The tosses HHTTTHHTTT are Lecture 1’s.')});

// ---------------------------------------------------------------------------------------- 1.2 cross-entropy
chap('1.2','Loss: cross-entropy');
const SB={W:640,H:336,slot:58,gap:34,z0:224,h:150,fs:20};
function sbars(vals,{ox,max,col,title,b,dec=2}){                // one small panel of signed bars, classes 1 to 3
 const {slot,z0,h,fs}=SB,w=slot*.66,pw=slot*vals.length;
 let g=label(ox+pw/2,30,title,{size:21,anchor:'middle',fill:col,weight:700})+line(ox,z0,ox+pw,z0,{stroke:AX,w:2});
 vals.forEach((v,j)=>{const x=ox+j*slot+(slot-w)/2,hh=Math.abs(v)/max*h,y=v>=0?z0-hh:z0;
  g+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${Math.max(hh,1.5).toFixed(1)}" rx="4" fill="${col}"/>`;
  g+=label(x+w/2,v>=0?y-8:y+hh+fs+2,num(v,dec),{size:fs,anchor:'middle',fill:COL.ink,weight:600});
  g+=label(x+w/2,SB.H-8,j+1,{size:21,anchor:'middle',fill:CE.y[j]?COL.target:MUTED,weight:CE.y[j]?700:''});});
 return b?gat(b,g):g;
}
const ox3=[0,1,2].map(k=>10+k*(SB.slot*3+SB.gap));
const ceFig=fig(SB.W,SB.H,sbars(CE.z,{ox:ox3[0],max:Math.max(...CE.z),col:COL.ink,title:'logits z',dec:1})+
 sbars(CE.p,{ox:ox3[1],max:1,col:COL.guess,title:'softmax p',b:3})+sbars(CE.grad,{ox:ox3[2],max:1,col:COL.loss,title:'p − y',b:4}),
 'Three panels of bars for classes 1 to 3: the logits, the softmax probabilities, and the gradient p minus y');
const zs=CE.z.map(v=>tn(v,v%1?1:0)).join(',\\ ');
S_('nll-loss','found','Cross-entropy is the negative log of the right class',{builds:4,from:2,
 body:cols(`<div class="p1-left">`+stack(
   sm(eq(R`\lcol{-\ln p(y\mid x)}=-\pcol{y}\ln\qcol{p}-(1-\pcol{y})\ln(1-\qcol{p})`),
    at(2,eq(R`\pcol{y}=${CE.bin.y},\ \qcol{p}=${tn(CE.bin.p,1)}:\quad \lcol{-\ln ${tn(CE.bin.p,1)}=${tn(CE.bin.loss)}}`))),
   at(3,sm(eq(R`\lcol{-\ln \qcol{p_{1}}}=-z_1+\ln\textstyle\sum_j e^{z_j}=${tn(-CE.z[0],0)}+${tn(CE.lse)}=\lcol{${tn(CE.loss)}}`))),
   at(4,sm(eq(R`\frac{\partial\,\lcol{\text{loss}}}{\partial z}=\qcol{p}-\pcol{y}=(${CE.grad.map(v=>tn(v)).join(',\\ ')})`))))+'</div>',
  ceFig,'1fr 0.95fr'),
 lines:[`Two networks. One says ${kp('p = '+CE.bin.p)} for a label ${kt('y = '+CE.bin.y)}. The other gives the logits shown, and class 1 is correct. What loss does each pay?`,
  `The Bernoulli’s negative log is binary cross-entropy: ${kl(`−ln ${CE.bin.p} = ${num(CE.bin.loss)}`)}. Had it said ${CE.bin.p_bad}, it would pay ${num(CE.bin.loss_bad)}.`,
  `Softmax turns the logits into probabilities. Class 1 gets ${num(CE.p[0])}, so the loss is ${kl(`−ln ${num(CE.p[0])} = ${num(CE.loss)}`)}. That is cross-entropy.`,
  `Its gradient at the logits is p − y. A step against it raises the right class’s logit and lowers the others by what they took.`],
 ask:['What loss does each network pay?','Why is a confident mistake so expensive?','How large can this loss get, and how small?','Why is p − y a good learning signal?'],
 expected:['Each pays −ln of the probability it gave the right answer. Let them compute the first one.',
  `The negative log grows without bound as p goes to 0: p = ${CE.bin.p_bad} already costs ${num(CE.bin.loss_bad)}.`,
  'Zero when the right class gets probability 1; no upper bound as it goes to 0. Ten equal guesses would give ln 10.',
  'It is zero once p matches y, and each logit moves by how wrong its probability is. The same p − y comes back in every classifier of the course.'],
 point:'Binary and multi-class cross-entropy are the negative log-likelihoods of a Bernoulli and a categorical output.',
 misconception:'That cross-entropy needs a separate log-softmax step in code. F.cross_entropy takes the logits and does both, which is also numerically safer (Lecture 9).',
 transition:'Line the cases up: each kind of target has its own distribution.',source:SRC('The logits (2, 1, −0.5) are from Lecture 2’s softmax chart.')});

// ---------------------------------------------------------------------------------------- 1.3 a distribution per target
chap('1.3','Output: a distribution');
const gcell=(b,s)=>at(b,s);
S_('dist-loss','found','Each kind of target picks a distribution, and a loss',{builds:3,from:3,
 body:`<div class="table-sm p1-dist">${table(['the target','distribution','output layer','loss','one example'],[
   ['a label, 0 or 1','Bernoulli','sigmoid','binary cross-entropy',m(R`-\ln ${tn(CE.bin.p,1)}=${tn(CE.bin.loss)}`)],
   [`one of ${m('K')} classes`,'categorical','softmax','cross-entropy',m(R`-\ln ${tn(CE.p[0])}=${tn(CE.loss)}`)],
   ['a real number',gcell(2,`Gaussian, fixed ${m(R`\sigma`)}`),gcell(2,'none (the mean)'),gcell(2,'squared error'),
    gcell(3,m(R`r=${tn(G.r,0)}:\ ${tn(G.sq,0)}+${tn(G.const)}`))]])}</div>
  ${at(3,eq(R`\lcol{-\ln \N(y;\,\qcol{\mu},\sigma^2)}=\frac{(\pcol{y}-\qcol{\mu})^2}{2\sigma^2}+\tfrac12\ln(2\pi\sigma^2)`))}`,
 lines:['The last two slides in one table: each kind of target gets a distribution, and its negative log is the loss. What goes in the last row?',
  'A real number gets a Gaussian with a fixed σ. The network outputs its mean, with nothing to squash it.',
  `Its negative log is the squared error over 2σ², plus a constant that never moves the fit. With r = ${num(G.r,0)} and σ = ${num(G.sigma,0)}: ${num(G.sq,0)} plus ${num(G.const)}.`],
 ask:['For a real-valued target, which distribution, which output layer, which loss?','Why no sigmoid or softmax on the last row?','So what does training with squared error assume about the noise?'],
 expected:['A Gaussian; no squashing; squared error.','A mean can be any real number, so the output layer leaves the score as it is.',
  'Gaussian noise with one fixed σ for every input. Outliers then pull hard on the fit; a Laplace noise model gives absolute error instead (Lecture 3).'],
 point:'Choosing the output distribution chooses the output layer and the loss together.',
 misconception:'That squared error is a neutral default. It is the Gaussian’s negative log-likelihood, and it brings the Gaussian’s light tails with it.',
 transition:'The data gave us the loss. What if we also have a belief about the weights?',source:SRC('Rows follow the retrieval summary of Lectures 2 and 3; the Gaussian example is from the S03 cheat sheet.')});

// ---------------------------------------------------------------------------------------- 1.4 a prior adds a penalty
chap('1.4','Loss: a prior adds a penalty');
const MW=600,MH=400,PM=panel({W:MW,H:MH,L:56,R:18,T:16,B:58,xMin:-1.5,xMax:2.5,yMin:-3,yMax:10,ticksX:[-1,0,1,2],ticksY:[0,4,8],xLabel:'x'});
const ti0=idx(M.taus,M.default);
const lineOf=([b,w])=>[[PM.x(-1.5),PM.y(b+w*-1.5)],[PM.x(2.5),PM.y(b+w*2.5)]];
const leg=(y,col,dash,txt)=>line(84,y-7,124,y-7,{stroke:col,w:4,dash})+label(134,y,txt,{size:22,fill:col,weight:650});
const mapStatic=PM.axes+`<clipPath id="p1-mapclip"><rect x="${PM.x(-1.5)}" y="${PM.y(10)}" width="${PM.x(2.5)-PM.x(-1.5)}" height="${PM.y(-3)-PM.y(10)}"/></clipPath>`+
 `<g clip-path="url(#p1-mapclip)">${path(lineOf(M.ml),{stroke:COL.ink,w:3,dash:'9 6'})}</g>`+
 M.x.map((x,i)=>dot(PM.x(x),PM.y(M.y[i]),9,{fill:COL.data})).join('')+
 leg(52,COL.ink,'9 6',`ML: b = ${num(M.ml[0])}, w = ${num(M.ml[1])}`);
function mapLive(k){
 const f=M.fits[k];
 return `<g class="p1-b2"><g clip-path="url(#p1-mapclip)">${path(lineOf(f),{stroke:COL.guess,w:4.5})}</g>${leg(90,COL.guess,'',`MAP: b = ${num(f[0])}, w = ${num(f[1])}`)}`+
  label(MW-30,PM.y(-2.1),`λ = 1/(2τ²) = ${num(M.lam[k])}`,{size:22,anchor:'end',fill:COL.loss,weight:650,halo:true})+'</g>';
}
const mapAlt=`Four data points with the maximum-likelihood line and the MAP line for a Gaussian prior of width τ`;
register('p1-map',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${MW} ${MH}" role="img" aria-label="${esc(mapAlt)}">${mapStatic}<g class="live">${inner}</g></svg></div>`;
 if(ctx.printing){el.innerHTML=frame(mapLive(ti0));return;}
 el.innerHTML=frame('')+`<div class="w-controls p1-b2">${slider('tau','prior width τ',0,M.taus.length-1,1,ti0)}</div>`;
 const live=el.querySelector('.live'),range=el.querySelector('[data-k="tau"]'),out=el.querySelector('[data-o="tau"]');
 const show=()=>{const k=+range.value;out.textContent=num(M.taus[k]);live.innerHTML=mapLive(k);};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());
 show();
});
const t01=idx(M.taus,0.1),tBig=M.taus.length-1;
S_('map-prior','found','A Gaussian prior is an L2 penalty',{builds:3,from:3,
 body:cols(`<div class="p1-left">`+stack(
   sm(eq(R`\text{ML:}\ \min_\theta\ \tfrac{1}{2\sigma^2}\textstyle\sum_i\big(\pcol{y_i}-(\qcol{b+wx_i})\big)^2`)),
   at(2,sm(eq(R`\text{MAP:}\ \text{add}\ \lcol{-\ln p(\theta)}=\tfrac{1}{2\tau^2}\|\theta\|^2+C`))),
   at(3,`<div class="takeaway">A Gaussian prior on the weights is the L2 penalty, with ${m(R`\lambda=1/(2\tau^2)`)}. In PyTorch it is weight decay.</div>`))+'</div>',
  widget('p1-map',mapAlt),'1fr 1fr'),
 lines:[`${Word(M.x.length)} points from ${M.true[0]} + ${M.true[1]}x, noise σ = ${M.sigma}. Maximum likelihood gives the dashed line. With a prior θ ~ N(0, ${M.default}²), which way does it move?`,
  `Toward zero. The prior adds its negative log to the loss, and at τ = ${M.default} the fit is (${num(M.fits[ti0][0])}, ${num(M.fits[ti0][1])}). Drag τ.`,
  `A narrow prior, τ = ${M.taus[t01]}, pulls to (${num(M.fits[t01][0])}, ${num(M.fits[t01][1])}); a wide one gives maximum likelihood back.`],
 ask:['Which way does the line move, and which of b and w moves more?','What does a small τ say about the weights, and what does it do to the fit?','Where have you met this penalty under another name?'],
 expected:['Both shrink toward 0. Ask for the reason before the click: the prior says weights near 0 are more probable.',
  `It says the weights are almost surely near 0. At τ = ${M.taus[0]} the line is nearly flat, (${num(M.fits[0][0])}, ${num(M.fits[0][1])}): the prior outweighs ${word(M.x.length)} points.`,
  'Ridge regression and weight decay. A Laplace prior gives the L1 penalty instead, which can set weights exactly to zero.'],
 point:'The regularizer is the negative log of a prior: MAP is the negative log-likelihood plus the negative log-prior.',
 misconception:'That regularization is a trick bolted onto the loss. It is the prior written in the same units as the loss.',
 demo:`Click 2: drag τ from ${M.taus[tBig]} down to ${M.taus[0]} and watch the purple line flatten while λ grows.`,
 transition:'The losses are settled. Now the model itself: one weighted sum is not enough.',source:SRC('The four points and σ = 0.5 are Lecture 3’s MAP example.')});

// ---------------------------------------------------------------------------------------- 1.5 layers and ReLUs
chap('1.5','Model: layers and ReLUs');
const XW=1152,XH=350;
const PX=panel({ox:0,W:420,H:XH,L:58,R:24,T:34,B:62,xMin:-.3,xMax:1.3,yMin:-.3,yMax:1.3,ticksX:[0,1],ticksY:[0,1],xLabel:'x₁',yLabel:'x₂'});
const PHs=panel({ox:628,W:524,H:XH,L:58,R:24,T:34,B:62,xMin:-.6,xMax:2.3,yMin:-.4,yMax:1.4,ticksX:[0,1,2],ticksY:[0,1],xLabel:'h₁',yLabel:'h₂'});
const pt=(P,a,b,y,r=12)=>y?dot(P.x(a),P.y(b),r,{fill:COL.target,stroke:'#fff',w:2}):dot(P.x(a),P.y(b),r-1,{fill:'#fff',stroke:COL.ink,w:3});
const zLine=h1=>(-X.b2-X.w2[0]*h1)/X.w2[1],zFoot=(-X.b2-X.w2[1]*-.4)/X.w2[0];   // z = 0 on the h-plane, and where it meets the bottom edge
const hDone=new Set(),hPts=X.h.map((h,i)=>{const k=h.join(',');const first=!hDone.has(k);hDone.add(k);return {h,i,first,n:X.h.filter(q=>q.join(',')===k).length};}).filter(q=>q.first);
const xorFig=fig(XW,XH,
 PX.axes+X.x.map(([a,b],i)=>pt(PX,a,b,X.y[i])+label(PX.x(a)+16,PX.y(b)-16,`(${a}, ${b})`,{size:21,fill:COL.ink})).join('')+
 gat(2,arrow(436,XH/2-30,612,XH/2-30,{stroke:COL.ink,w:3})+label(524,XH/2-46,'two ReLU units',{size:21,anchor:'middle',fill:COL.ink,weight:600})+
  PHs.axes+hPts.map(({h,i,n})=>pt(PHs,h[0],h[1],X.y[i])+label(PHs.x(h[0])+(h[0]===0?-16:0),PHs.y(h[1])+(h[0]===1?38:h[0]===0?34:-22),`(${h.join(', ')})${n>1?', twice':''}`,
   {size:21,anchor:h[0]===0?'end':'middle',fill:COL.ink})).join(''))+
 gat(3,`<path d="M${PHs.x(zFoot)},${PHs.y(-.4)} L${PHs.x(2.3)},${PHs.y(-.4)} L${PHs.x(2.3)},${PHs.y(zLine(2.3))}Z" fill="${COL.target}" opacity=".09"/>`+
  path([[PHs.x(zFoot),PHs.y(-.4)],[PHs.x(2.3),PHs.y(zLine(2.3))]],{stroke:COL.guess,w:3.5})+
  label(PHs.x(2.25),PHs.y(-.1),'z > 0',{size:21,anchor:'end',fill:COL.target,weight:700})+
  hPts.map(({h,i})=>label(PHs.x(h[0])+(h[0]===1?18:h[0]===2?-22:-16),PHs.y(h[1])+(h[0]===0?-20:7),`z = ${num(X.z[i],0)}`,
   {size:21,anchor:h[0]===1?'start':'end',fill:COL.guess,weight:650,halo:true})).join('')),
 'The four XOR inputs on the left; on the right, where two ReLU units send them, and the line z = 0 that separates the classes');
const sg=v=>v<0?`-${Math.abs(v)}`:`+${v}`;
S_('mlp-xor','found','Two ReLU units solve XOR',{builds:3,from:4,
 body:`${xorFig}<div class="state-slot">${only('1',small(`${kt('Filled')}: y = 1. Hollow: y = 0.`))}${only('2-3',cols(
   sm(eq(R`h_1=\operatorname{ReLU}(x_1+x_2),\ \ h_2=\operatorname{ReLU}(x_1+x_2${sg(X.b1[1])})`)),
   at(3,sm(eq(R`\qcol{z}=${X.w2[0]}h_1${sg(X.w2[1])}h_2${sg(X.b2)}`))),'1.55fr 1fr'))}</div>`,
 lines:['Lecture 4’s XOR. The two filled points have y = 1. Can one straight line put them on one side and the hollow points on the other?',
  'No line can. Two ReLU units move the points: (1, 0) and (0, 1) land on the same spot, h = (1, 0).',
  `In h, one line does it: z is positive only for the filled points. This is the network the opening trained.`],
 ask:['Can one straight line separate them?','Why does stacking two linear layers without the ReLU not help?','Which parameters did we set by hand here, and which would training find?'],
 expected:['No. Any line that puts (0, 1) and (1, 0) on one side puts (0, 0) or (1, 1) there too.',
  'Two linear layers multiply into one linear map, which is still one straight boundary. The ReLU is what bends.',
  `All ${word(X.W1.flat().length+X.b1.length+X.w2.length+1)} were set by hand in Lecture 4. Training would find some network that works, not necessarily this one.`],
 point:'A hidden layer of ReLUs moves the data until one straight line is enough.',
 misconception:'That more linear layers add power. Without a nonlinearity they collapse into one.',
 transition:'Two units handle four points. How far does this go?',source:SRC('The XOR network is the one Lecture 4 built by hand.')});

// ---------------------------------------------------------------------------------------- 1.5 hinges
const HW=1152,HH=340,PH=panel({W:800,H:HH,L:60,R:24,T:22,B:52,xMin:-HG.pi,xMax:HG.pi,yMin:-1.3,yMax:1.3,ticksX:[-HG.pi,0,HG.pi],
 fmtX:v=>v<0?'−π':v>0?'π':'0',ticksY:[-1,0,1]});
const M0=String(HG.ms[0]),RX=850;
const hingeStatic=PH.axes+path(HG.x.map((x,i)=>[PH.x(x),PH.y(HG.sin[i])]),{stroke:COL.target,w:4})+
 line(RX,40,RX+40,40,{stroke:COL.target,w:4})+label(RX+52,47,'sin x, the target',{size:22,fill:COL.target,weight:650})+
 line(RX,84,RX+40,84,{stroke:COL.guess,w:4})+label(RX+52,91,'a sum of hinges',{size:22,fill:COL.guess,weight:650})+
 label(RX,170,'hinges',{size:22})+label(RX,268,'largest gap',{size:22});
function hingeLive(k){
 const f=HG.fits[k];
 return path(f.kx.map((x,i)=>[PH.x(x),PH.y(f.ky[i])]),{stroke:COL.guess,w:4})+
  f.kx.slice(1,-1).map((x,i)=>dot(PH.x(x),PH.y(f.ky[i+1]),5.5,{fill:COL.guess})).join('')+
  line(PH.x(f.at),PH.y(f.f_at),PH.x(f.at),PH.y(f.g_at),{stroke:COL.loss,w:5})+
  label(RX,222,f.m,{size:44,fill:COL.guess,weight:700})+label(RX,322,num(f.err,3),{size:44,fill:COL.loss,weight:700});
}
const hingeAlt='sin x on minus pi to pi with a piecewise-linear fit made of ReLU hinges, and the largest gap between them';
register('p1-hinges',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${HW} ${HH}" role="img" aria-label="${esc(hingeAlt)}">${hingeStatic}<g class="live">${inner}</g></svg></div>`;
 if(ctx.printing){el.innerHTML=frame(hingeLive(String(HG.ms[1])));return;}
 el.innerHTML=frame('')+`<div class="w-controls"><span>number of hinges</span>${HG.ms.map(m=>`<button type="button" data-m="${m}" aria-pressed="${String(m)===M0}">${m} hinges</button>`).join('')}</div>`;
 const live=el.querySelector('.live');let k=M0;
 const show=()=>{live.innerHTML=hingeLive(k);};
 for(const b of el.querySelectorAll('button[data-m]'))b.addEventListener('click',()=>{k=b.dataset.m;
  for(const o of el.querySelectorAll('button[data-m]'))o.setAttribute('aria-pressed',o===b);b.blur();show();});
 show();
});
const E=HG.ms.map(k=>HG.fits[k].err);
S_('p1-hinges','found','Enough hinges fit any curve',{builds:2,from:4,
 body:`${sm(eq(R`g(x)=c+\textstyle\sum_{j} v_j\,\operatorname{ReLU}(x-k_j)`))}${widget('p1-hinges',hingeAlt)}`,
 lines:['Each ReLU is a hinge, and a sum of hinges is a bent line. Here it is fitted to sin x. Add hinges: what happens to the largest gap?',
  `It shrinks: ${E.map(v=>num(v,3)).join(', ')} for ${HG.ms.join(', ')} hinges. Enough hinges fit any continuous curve on a bounded range; training still has to find them.`],
 ask:['What happens to the largest gap as hinges are added?','What does this result not promise?'],
 expected:[`It shrinks, roughly with the square of the hinge spacing. Under ${HG.eps} needs ${word(HG.first_under)} hinges (gap ${num(HG.first_under_err,3)}).`,
  'It does not say that gradient descent will find those weights, that a finite data set is enough to pin them down, or that the fit is any good outside the range.'],
 point:'A one-hidden-layer ReLU network can approximate any continuous function on a bounded range, given enough units.',
 misconception:'That this theorem explains why deep networks train well. It is a statement about capacity only.',
 demo:'Click the three buttons in order; point at the red bar, the largest gap, each time.',
 transition:'Time to check the whole part with one number.',source:SRC('Fits of sin x on [−π, π] with equally spaced hinges, as in Lecture 4.')});

// ---------------------------------------------------------------------------------------- check
chap();
const tenBars=bars(Array.from({length:K.K},(_,i)=>({name:String(i+1),v:K.p})),{W:560,H:220,max:.25,fs:19,col:COL.guess,alt:`Ten equal probabilities of ${K.p}`});
S_('check-1','found',`Check: a loss of ${num(K.loss)} at the first step`,{builds:2,
 body:`<div class="p1-log">${code(`step 1   loss ${num(K.loss,4)}`,'text')}</div>
  ${p(`A ${K.K}-class classifier, trained with cross-entropy, reports this loss at its first step. Is something broken?`)}
  ${at(2,cols(`<div class="answer">No. A fresh network gives the ${word(K.K)} classes about equal probability, ${num(K.p)} each, and ${m(R`\lcol{-\ln ${tn(K.p,1)}}=\ln ${K.K}=${tn(K.loss)}`)}. It starts where it should.</div>`,tenBars,'1.15fr 1fr'))}`,
 lines:['Two ideas of this part meet here. Take a minute with your neighbour.',
  `ln ${K.K} is the loss of ${word(K.K)} equal guesses. A first loss far above it would mean the starting logits are too large.`],
 ask:['Is something broken?','What first loss would worry you?'],
 expected:[`No: −ln of the right class’s probability, with ${word(K.K)} equal guesses, is ln ${K.K} = ${num(K.loss)}.`,
  `One much larger than ${num(K.loss)}: the network is confidently wrong before it has learned anything, usually from badly scaled initial weights. One much smaller would hint at a bug or a leak of labels.`],
 point:'Cross-entropy at the start of training should be close to ln K; that is a free sanity check.',
 transition:'The whole part on one card.'});

// ---------------------------------------------------------------------------------------- recap
recap(1,[
 `A loss is ${m(R`\lcol{-\ln}`)} of the probability the model gave the data. Cross-entropy and squared error both come from there.`,
 `A prior on the weights adds ${m(R`\lcol{-\ln p(\theta)}`)}. A Gaussian prior is the L2 penalty.`,
 'ReLUs between linear layers let the model bend: two units solve XOR.'],
 'how the gradient of that loss is computed, for every weight at once.',
 {card:RECIPES.mlp,title:'MLP classifier',
  ask:'Which rows of the card did this part fill in?',
  expected:'The model (an MLP), the output (softmax probabilities) and the loss (−ln of the right class). The update row is what Part 2 is about.',
  point:'Part 1 fixed the model, the output and the loss of the recipe.',
  transition:'Part 2: how the weights move.'});
