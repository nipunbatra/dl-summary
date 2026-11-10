// Part 6: generative modelling (Lectures 23 to 25). KL divergence on the weather, the ELBO on the coin, the
// reparameterisation trick, the tiny Bayesian network, then the autoencoder and the VAE on MNIST test digit #90.
// Every number comes from data/p6.json (scripts/sci/p6.py).
import {S_,partBreak,recap,chap,p,ps,small,cols,stack,sm,at,only,answer,m,eq,R,data,need,word,Word,css,kt,kp,kl,
 label,line,path,dot,arrow,panel,widget,register,slider,num,tn,big,COL,MUTED,AX,PAPER,RECIPES,codeSteps,code,esc,gat,gonly} from '../kit.js';

const D=need(await data('p6'),'weather','classify','coin','reparam','bnn','vae','beta','images');
const W_=D.weather,CL=D.classify,CO=D.coin,RP=D.reparam,BN=D.bnn,V=D.vae,BE=D.beta;
const thou=v=>(v<0?'−':'')+Math.abs(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const SRC_KL='Weather and classifier from Lecture 23; coin and mixture fit from Lecture 24. Numbers: scripts/sci/p6.py.';
const SRC_VAE='MNIST test digit #90 and the trained networks of Lecture 25 (generative/lectures/lecture2/data). Numbers and images: scripts/sci/p6.py.';

// Parts of a figure that wait for the second click, even when a live figure redraws them.
css(`.scene[data-build="1"] .p6-g2{visibility:hidden}
.p6-tiles{display:grid;grid-template-columns:1.25fr .85fr 1.3fr;gap:18px}
.p6-tile{border:1.5px solid var(--line);border-radius:12px;padding:10px 16px;display:flex;flex-direction:column;gap:6px;min-width:0}
.p6-tile .math{font-size:25px}.p6-tile .caption{font-size:19px}
.p6-q{font-size:28px;line-height:1.4}
.p6-strip{max-width:350px}`);

/** A figure with pictures in it: the frame must clip, so it uses the photo-fig wrapper of the kit. */
const pfig=(W,H,inner,alt)=>`<div class="diagram-scroll photo-fig"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(alt)}">${inner}</svg></div>`;
const img=(name,x,y,w,h)=>`<image href="figures/p6-${name}.png" x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="none"/>`;
/** The 2 × 4 grid of decoded samples for one β, with a thin frame round each digit. */
function samples(k,x,y,h){
 const [iw,ih]=D.images['beta-'+k.replace('.','_')],s=h/ih,cell=(ih-12)/2;          // the PNG: two rows of four 28-pixel digits, 12 px apart
 let g=img('beta-'+k.replace('.','_'),x,y,iw*s,h);
 for(let i=0;i<D.images.samples;i++){const cx=x+(i%4)*(cell+12)*s,cy=y+Math.floor(i/4)*(cell+12)*s;
  g+=`<rect x="${cx.toFixed(1)}" y="${cy.toFixed(1)}" width="${(cell*s).toFixed(1)}" height="${(cell*s).toFixed(1)}" fill="none" stroke="${AX}" stroke-width="1.5"/>`;}
 return g;
}
const sampleW=h=>{const [iw,ih]=D.images['beta-1'];return iw*h/ih;};

// ---------------------------------------------------------------------------------------- the part opens
partBreak(6,'Learning a distribution',
 'Lectures 23 to 25 learned whole distributions, over a network’s weights and over images. The loss became a KL, and the model gained a sampled code.',
 ['measure how far one distribution is from another: the KL divergence',
  'when the posterior is out of reach, fit a simpler q by raising the ELBO',
  `sample through ${m(R`\theta=\mu+\sigma\epsilon`)} so that gradients flow`,
  'put an encoder and a decoder around a code z: the VAE'],
 {line:'By the end you can say why the VAE’s loss is a reconstruction error plus a KL.',
  ask:'Every loss so far scored one prediction against one target. What would a loss between two whole distributions look like?',
  expected:'Something that is zero when the two distributions agree and grows as they differ. That is the KL divergence, and Part 1’s cross-entropy is already one.',
  transition:'6.1: the KL divergence, on Lecture 23’s weather.'});

// ---------------------------------------------------------------------------------------- 6.1 KL divergence
chap('6.1','Loss: KL divergence');
const KLP=W_.path,KW=600,KH=330,KY=v=>250-v/.6*186;
const kx=(i,j)=>40+i*180+(j?84:24);                                     // outcome i, bar j (0: p, 1: q)
const klStatic=line(20,250,580,250,{stroke:AX})+
 `<rect x="24" y="14" width="20" height="20" rx="4" fill="${COL.target}"/>`+label(52,31,'p, our town',{size:21,fill:COL.ink})+
 `<rect x="250" y="14" width="20" height="20" rx="4" fill="${COL.guess}"/>`+label(278,31,'q, the code’s town',{size:21,fill:COL.ink})+
 W_.names.map((n,i)=>label(kx(i,0)+66,280,n,{size:21,anchor:'middle'})+
  `<rect x="${kx(i,0)}" y="${KY(W_.p[i]).toFixed(1)}" width="56" height="${(250-KY(W_.p[i])).toFixed(1)}" rx="4" fill="${COL.target}"/>`+
  label(kx(i,0)+28,KY(W_.p[i])-8,num(W_.p[i]),{size:20,anchor:'middle',fill:COL.ink})).join('');
function klLive(k){
 const s=KLP[k];
 return s.q.map((v,i)=>`<rect x="${kx(i,1)}" y="${KY(v).toFixed(1)}" width="56" height="${(250-KY(v)).toFixed(1)}" rx="4" fill="${COL.guess}"/>`+
  label(kx(i,1)+28,KY(v)-8,num(v),{size:20,anchor:'middle',fill:COL.guess,weight:650})).join('')+
  `<g class="p6-g2">${label(20,318,`H(p) = ${num(W_.H_p)}`,{size:21,fill:COL.target,weight:650})+label(190,318,`H(p, q) = ${num(s.H_pq)}`,{size:21,fill:COL.guess,weight:650})+
   label(380,318,`KL = ${num(s.kl)} bits`,{size:21,fill:COL.loss,weight:650})}</g>`;
}
const klAlt='Bar charts of our town’s weather p and the other town’s q, with the entropy, the cross-entropy and the KL in bits';
register('p6-kl',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig">${`<div class="diagram-scroll"><svg class="plot wsvg" viewBox="0 0 ${KW} ${KH}" role="img" aria-label="${klAlt}">${klStatic}<g class="live">${inner}</g></svg></div>`}</div>`;
 if(ctx.printing){el.innerHTML=frame(klLive(0));return;}
 el.innerHTML=frame('')+`<div class="w-controls">${slider('t','move q towards p',0,KLP.length-1,1,0)}</div>`;
 const live=el.querySelector('.live'),range=el.querySelector('[data-k="t"]'),out=el.querySelector('[data-o="t"]');
 const show=()=>{const k=+range.value;out.textContent=`q = (${KLP[k].q.map(v=>num(v)).join(', ')})`;live.innerHTML=klLive(k);};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());
 show();
});
const PRED=num(CL.p_true);
S_('kl','genm','The KL divergence: the extra cost of the wrong code',{builds:3,from:[23,'#kl-extra-bits/3'],
 body:cols(widget('p6-kl',klAlt),sm(
  eq(R`H(\pcol p,\qcol q)=\sum_x\pcol{p(x)}\,\log_2\frac{1}{\qcol{q(x)}}`),
  small('the bits a day, when the days come from p and the code was built for q'),
  at(2,eq(R`\begin{aligned}\lcol{\KL(\pcol p\,\|\,\qcol q)}&=H(\pcol p,\qcol q)-H(\pcol p)\\&=${tn(W_.H_pq)}-${tn(W_.H_p)}=\lcol{${tn(W_.kl)}}\text{ bits}\end{aligned}`)),
  at(3,eq(R`\lcol{\KL(\pcol{\text{label}}\,\|\,\qcol{\text{prediction}})}=-\ln\qcol{${tn(CL.p_true)}}=\lcol{${tn(CL.loss_nats)}}\text{ nats}`))),'1fr 1.08fr'),
 lines:['Lecture 23’s weather. Our days follow p; we borrow the code another town built for its weather, q. How many extra bits a day does that cost?',
  'H(p) is what the best code for our days costs. The KL is the extra we pay. Drag q towards p: at q = p the extra is zero.',
  `Part 1’s loss is the same quantity. The label is p, one-hot on ${CL.classes[CL.label]}; the prediction gives ${CL.classes[CL.label]} ${PRED}. The label’s entropy is 0, so cross-entropy is a KL.`],
 ask:['Before any formula: which outcome does the borrowed code handle worst for us?',
  'Can the KL ever be negative? When is it zero?',
  `Why is it KL(label ‖ prediction), and not the other way round?`],
 expected:[`Sunny. It is our most common day (${num(W_.p[0])}), but the other town’s code gives it a long codeword because sunny is rare there (${num(W_.q[0])}).`,
  'Never negative: no code beats the one built for the true distribution. It is zero only when q = p, which the slider shows.',
  'The label is where the data come from, so it sits on the left. The reverse would need ln(q/0) for every wrong class, which is infinite.'],
 point:'KL(p ‖ q) = H(p, q) − H(p) is the extra cost of using q when the data come from p, and the classification loss is one such KL.',
 misconception:'That KL is a distance. It is not symmetric: KL(p ‖ q) and KL(q ‖ p) differ, and the order matters in variational inference.',
 demo:`Drag the slider from 0 to the end: the KL falls from ${num(KLP[0].kl)} to ${num(KLP.at(-1).kl)} bits as q reaches p. Leave it at 0 before the click to 6.2.`,
 transition:'Variational inference uses the same KL with the order flipped, to a posterior we cannot compute.',source:SRC_KL});

// ---------------------------------------------------------------------------------------- 6.2 the ELBO
chap('6.2','Loss: the ELBO');
const WK=CO.walk,EW=620,EH=384,EV=CO.log_ev;
const EP=panel({W:EW,H:250,L:20,R:20,T:46,B:50,xMin:0,xMax:1,yMin:0,yMax:Math.max(...CO.post_pdf,...WK.flatMap(w=>w.pdf))*1.05,ticksX:[0,.5,1],xLabel:'θ, the chance of heads'});
const lo=Math.min(...WK.map(w=>w.elbo))-.12,hi=EV+.18,EX=v=>40+(v-lo)/(hi-lo)*540,EY=335;
const curveOf=pdf=>CO.r.map((x,i)=>[EP.x(x),EP.y(pdf[i])]);
const qName=w=>`Beta(${num(w.a,w.a%1?1:0)}, ${num(w.b,w.b%1?1:0)})`;
const elboStatic=EP.axes+path(curveOf(CO.post_pdf),{stroke:COL.target,w:4})+
 label(36,26,`posterior Beta(${CO.post[0]}, ${CO.post[1]})`,{size:21,fill:COL.target,weight:650})+
 `<g class="p6-g2">${line(40,EY,580,EY,{stroke:AX})+line(EX(EV),EY-46,EX(EV),EY+12,{stroke:COL.data,w:3})+
  label(Math.min(EX(EV),500),EY-54,`ln p(𝒟) = ${num(EV)}`,{size:21,fill:COL.data,weight:650,anchor:'middle'})}</g>`;
function elboLive(k){
 const w=WK[k],xe=EX(w.elbo);
 return path(curveOf(w.pdf),{stroke:COL.guess,w:4})+label(584,26,`q = ${qName(w)}`,{size:21,fill:COL.guess,weight:650,anchor:'end'})+
  `<g class="p6-g2">${line(xe,EY,EX(EV),EY,{stroke:COL.loss,w:6})+dot(xe,EY,9,{fill:COL.guess})+
   label(EX(EV)-12,EY-14,`gap = KL = ${num(w.gap)}`,{size:21,fill:COL.loss,anchor:'end',weight:650})+
   label(Math.max(120,Math.min(xe,500)),EY+38,`ELBO = ${num(w.elbo)}`,{size:21,fill:COL.guess,anchor:'middle',weight:650})}</g>`;
}
const elboAlt='A Beta q against the coin’s posterior, and the log evidence split into the ELBO and the KL gap';
register('p6-elbo',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><div class="diagram-scroll"><svg class="plot wsvg" viewBox="0 0 ${EW} ${EH}" role="img" aria-label="${elboAlt}">${elboStatic}<g class="live">${inner}</g></svg></div></div>`;
 if(ctx.printing){el.innerHTML=frame(elboLive(CO.walk_default));return;}
 el.innerHTML=frame('')+`<div class="w-controls">${slider('s','move q',0,WK.length-1,1,CO.walk_default)}</div>`;
 const live=el.querySelector('.live'),range=el.querySelector('[data-k="s"]'),out=el.querySelector('[data-o="s"]');
 const show=()=>{const k=+range.value;out.textContent=qName(WK[k]);live.innerHTML=elboLive(k);};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());
 show();
});
const Q0=CO.start,QP=CO.posterior;
S_('elbo','genm','The ELBO: a lower bound we can raise',{builds:4,from:[24,'#elbo-goal/3'],
 body:cols(widget('p6-elbo',elboAlt),sm(
  eq(R`\dcol{\ln p(\mathcal D)}=\ln\!\int p(\mathcal D\mid\theta)\,p(\theta)\,d\theta=\dcol{${tn(EV)}}`),
  at(2,eq(R`\begin{aligned}\dcol{\ln p(\mathcal D)}&=\qcol{\ELBO(q)}+\lcol{\KL\big(q\,\|\,p(\theta\mid\mathcal D)\big)}\\\dcol{${tn(EV)}}&=\qcol{${tn(Q0.elbo)}}+\lcol{${tn(Q0.gap)}}\end{aligned}`)),
  at(3,eq(R`\begin{aligned}\qcol{\ELBO(q)}&=\E_q\big[\ln p(\mathcal D\mid\theta)\big]-\KL\big(q\,\|\,p(\theta)\big)\\&=${tn(Q0.fit)}-${tn(Q0.kl_prior)}\end{aligned}`)),
  answer(4,`At ${qName(QP)} the gap is ${num(QP.gap)}: the ELBO reaches ln p(𝒟), so this q is the posterior.`)),'1.02fr 1fr'),
 lines:[`Lecture 24’s coin: a Beta(${CO.prior.join(', ')}) prior, then ${CO.heads} heads in ${CO.tosses} tosses. For a coin we can compute the evidence and check VI against it.`,
  `Pick any q, here ${qName(Q0)}. The evidence splits into the ELBO, which we can compute, and the KL to the posterior, which we cannot.`,
  'The ELBO needs only q, the likelihood and the prior: q fits the data and stays near the prior.',
  'Drag q. The total never changes, so whatever the ELBO gains, the KL loses. Where does the gap close?'],
 ask:['For a network, why can we not compute this number?','Why is the ELBO always below ln p(𝒟)?','Which term pulls q towards the data, and which holds it back?','What makes the gap exactly zero?'],
 expected:[`The integral runs over every θ. Lecture 23 put a grid of points on each of the ${word(BN.n_weights)} weights of its tiny network and found it would take hours; a real network has millions of weights.`,
  `The KL is never negative (6.1), so the ELBO can only sit at or below the fixed total. For ${qName(Q0)} it is ${num(Q0.gap)} below.`,
  `The first term rewards q for putting its mass where the data are likely; the KL to the prior charges it for moving away from Beta(${CO.prior.join(', ')}).`,
  `q equal to the posterior, ${qName(QP)}: there the KL is 0. The fit is ${num(QP.fit)} and the KL to the prior ${num(QP.kl_prior)}, which add up to ln p(𝒟).`],
 point:'ln p(𝒟) = ELBO + KL(q ‖ posterior) is a fixed total, so raising the ELBO, which we can compute, lowers the KL, which we cannot.',
 misconception:'That the ELBO is maximised by fitting the data alone. A spike at the best θ fits better but pays a larger KL to the prior; on the coin its ELBO is lower than the posterior’s.',
 demo:`At click 4, drag slowly from the left: the purple curve slides onto the teal one, and the red gap shrinks to 0 at ${qName(QP)}, then opens again past it.`,
 transition:'To raise the ELBO by gradient steps we need its gradient, but the ELBO is an average over samples of q.',source:SRC_KL});

// ---------------------------------------------------------------------------------------- 6.3 the reparameterisation trick
chap('6.3','Update: gradients through a sample');
const node=(x,y,t,{col=COL.ink,dash='',r=34,fill='#fff',size=24}={})=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${col}" stroke-width="3"${dash?` stroke-dasharray="${dash}"`:''}/>`+label(x,y+8,t,{size,anchor:'middle',fill:col,weight:650});
const box=(x,y,w,h,t,col)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="#fff" stroke="${col}" stroke-width="3"/>`+label(x+w/2,y+h/2+8,t,{size:23,anchor:'middle',fill:col,weight:650});
const back=(x1,y1,x2,y2)=>arrow(x1,y1,x2,y2,{stroke:COL.loss,w:2.5,head:11,dash:'7 6'});
const RG1=label(0,24,'θ drawn by q.sample()',{size:21,fill:MUTED,weight:650})+
 node(50,80,'μ')+node(50,160,'σ')+arrow(86,86,220,114,{w:2.5,head:11})+arrow(86,154,220,126,{w:2.5,head:11})+
 box(230,90,170,50,'θ ~ 𝒩(μ, σ²)',COL.amber)+arrow(400,115,500,115,{w:2.5,head:11})+node(540,115,'loss',{col:COL.loss,r:38,size:21})+
 gat(2,back(500,96,404,96)+label(318,186,'the gradient stops at the draw',{size:20,fill:COL.loss,anchor:'middle'})+
  line(226,96,206,84,{stroke:COL.loss,w:2.5,dash:'7 6'})+line(186,66,204,84,{stroke:COL.loss,w:4})+line(204,66,186,84,{stroke:COL.loss,w:4}));
const RG2=label(0,236,'θ = μ + σε',{size:21,fill:MUTED,weight:650})+
 node(50,284,'μ')+node(50,358,'σ')+node(160,410,'ε',{col:COL.amber,r:28})+
 arrow(86,290,220,318,{w:2.5,head:11})+arrow(86,354,220,330,{w:2.5,head:11})+arrow(176,386,232,346,{w:2.5,head:11,stroke:COL.amber})+
 box(230,298,170,50,'μ + σε',COL.guess)+arrow(400,323,500,323,{w:2.5,head:11})+node(540,323,'loss',{col:COL.loss,r:38,size:21})+
 back(500,304,404,304)+back(228,306,90,276)+back(228,340,90,368);
const RGW=600,RGH=446;
S_('reparam','genm','Gradients through a sample: θ = μ + σε',{builds:3,from:[24,'#reparam/3'],
 body:cols(`<div class="diagram-scroll"><svg class="plot wsvg" viewBox="0 0 ${RGW} ${RGH}" role="img" aria-label="Two computation graphs: with q.sample() the backward pass stops at the draw; with theta = mu + sigma epsilon it reaches mu and sigma">${RG1+gat(2,RG2)}</svg></div>`,sm(
  `<div class="code-sm">${code('theta = q.sample()')}</div>`,
  small(`In Lecture 24 this fit left q at 𝒩(${num(RP.mix.sample.mu)}, ${num(RP.mix.sample.sigma)}²) for ${RP.mix.steps} steps, with the KL still ${num(RP.mix.sample.kl,1)}.`),
  at(2,eq(R`\ncol{\epsilon}\sim\N(0,1),\qquad\theta=\mu+\sigma\ncol{\epsilon}`)),
  at(3,eq(R`\begin{aligned}\theta&=${tn(RP.mu,0)}+${tn(RP.sigma,0)}\times(\ncol{${tn(RP.eps)}})=${tn(RP.theta,1)}\\\frac{\partial\theta}{\partial\mu}&=${tn(RP.dmu,0)},\qquad\frac{\partial\theta}{\partial\sigma}=\ncol{\epsilon}=\ncol{${tn(RP.dsigma)}}\end{aligned}`)),
  at(3,`<div class="code-sm">${code('theta = q.rsample()   # one word changed')}</div>`)),'1fr 1.05fr'),
 lines:['Lecture 24 fitted q by drawing θ from it and averaging the loss. The code ran, and q did not move. Where does the gradient stop?',
  'Move the randomness into ε, a fixed input with no parameters. Then θ = μ + σε is ordinary arithmetic, and backprop goes through it.',
  `Here is one draw. With rsample() the same fit ends at 𝒩(${num(RP.mix.rsample.mu)}, ${num(RP.mix.rsample.sigma)}²), KL ${num(RP.mix.rsample.kl,3)}.`],
 ask:['Which arrow in the top graph has no derivative?','Is θ still random?','What is the gradient of θ with respect to σ, in words?'],
 expected:['The one into the draw. “Sample from 𝒩(μ, σ²)” is not a function of μ and σ that autograd can differentiate, so nothing flows back to them.',
  'Yes, with exactly the same distribution. Only the place where the randomness enters has moved: into ε, which needs no gradient.',
  `The noise itself: with ε = ${num(RP.eps)}, θ moves by ${num(RP.dsigma)} for each unit of σ. Each sample gives its own gradient, and their average estimates the true one.`],
 point:'Writing θ = μ + σε moves the randomness into an input, so the gradient of the loss reaches μ and σ; in PyTorch that is rsample().',
 misconception:'That q.sample() raises an error when used in a loss. It runs without complaint; the parameters simply get no gradient.',
 transition:'Next we fit the tiny network of Lecture 23 with this trick.',source:SRC_KL});

// ---------------------------------------------------------------------------------------- 6.4 the Bayesian network
chap('6.4','At test time: uncertainty');
const BP=panel({W:640,H:400,L:58,R:16,T:54,B:52,xMin:-2.5,xMax:2.5,yMin:-2,yMax:2,ticksX:[-2,-1,0,1,2],ticksY:[-2,-1,0,1,2],xLabel:'x'});
const band=(mean,sd,col,op)=>{const up=BN.grid.map((x,i)=>[BP.x(x),BP.y(mean[i]+2*sd[i])]),dn=BN.grid.map((x,i)=>[BP.x(x),BP.y(mean[i]-2*sd[i])]).reverse();
 return `<path d="${[...up,...dn].map((q,i)=>`${i?'L':'M'}${q[0].toFixed(1)},${q[1].toFixed(1)}`).join(' ')}Z" fill="${col}" fill-opacity="${op}" stroke="none"/>`;};
const meanLine=(mean,col,dash='')=>path(BN.grid.map((x,i)=>[BP.x(x),BP.y(mean[i])]),{stroke:col,w:3,dash});
const bnnFig=BP.axes+gat(2,band(BN.hmc_mean,BN.hmc_sd,COL.target,.22)+meanLine(BN.hmc_mean,COL.target,'8 6'))+band(BN.vi_mean,BN.vi_sd,COL.guess,.3)+meanLine(BN.vi_mean,COL.guess)+
 BN.x.map((x,i)=>dot(BP.x(x),BP.y(BN.y[i]),6)).join('')+
 label(66,26,'mean-field VI: mean ± 2 sd',{size:21,fill:COL.guess,weight:650})+gat(2,label(630,26,'MCMC reference',{size:21,fill:COL.target,weight:650,anchor:'end'}))+
 label(BP.x(0),BP.y(-1.62),'no data here',{size:20,anchor:'middle',fill:MUTED});
S_('bnn','genm','A Bayesian network: predictions with error bars',{builds:2,from:[24,'#bnn-vs-ref/2'],
 body:cols(`<div class="diagram-scroll"><svg class="plot wsvg" viewBox="0 0 640 400" role="img" aria-label="The tiny network's predictive band from variational inference, and the MCMC reference band, over 17 data points with a gap">${bnnFig}</svg></div>`,sm(
  eq(R`\qcol{\ELBO}=\underbrace{${tn(BN.fit)}}_{\text{fit the data}}-\underbrace{${tn(BN.kl)}}_{\text{stay near the prior}}=\qcol{${tn(BN.elbo)}}`),
  small(`up from ${thou(BN.elbo_start)} at the first step, after ${big(BN.steps)} steps of ${BN.draws} sampled networks each`),
  answer(2,`In the gap the band’s sd grows from ${num(BN.near_data.vi)} to ${num(BN.at_gap.vi)}. The reference is wider still: ${num(BN.at_gap.hmc)} in the gap and ${num(BN.at_edge.hmc)} at x = ${num(BN.at_edge.x,1)}, against ${num(BN.at_edge.vi)}.`)),'1.05fr 1fr'),
 lines:[`Lecture 23’s tiny network: ${word(BN.n_weights)} weights, ${BN.n_points} points with a gap. VI fitted q, a mean and a width per weight. Where should it be least sure?`,
  'At test time we draw networks from q and average them. VI widens its band in the gap, but less than the reference does: reverse KL makes q too sure.'],
 ask:['Where should a network that knows what it does not know be least sure?','Why is the VI band narrower than the reference?'],
 expected:['In the gap between the two groups of points, and beyond the ends of the data, where many curves fit equally well.',
  'VI minimises KL(q ‖ posterior), which charges q for mass where the posterior has little, so q hugs one mode and stays narrow. A diagonal q cannot tilt to follow correlated weights either.'],
 point:'A Bayesian network trained by VI returns a distribution over predictions, and its spread is the uncertainty; mean-field VI underestimates it.',
 misconception:'That the band is a confidence interval from repeated data. It is the spread of predictions from networks drawn from q, plus the noise.',
 transition:'Lecture 25 meets the same problem with images, where the unknown is a code instead of the weights.',source:'The tiny network of Lectures 23 and 24; bands from generative/lectures/lecture1/data/science.json. Numbers: scripts/sci/p6.py.'});

// ---------------------------------------------------------------------------------------- 6.5 the autoencoder and the VAE
chap('6.5','Model: encoder, code, decoder');
const A=V.arch,IM=120,trap=(x,y,w,h,wide,t,sub)=>{const n=wide==='in'?[[x,y],[x+w,y+h*.25],[x+w,y+h*.75],[x,y+h]]:[[x,y+h*.25],[x+w,y],[x+w,y+h],[x,y+h*.75]];
 return path([...n,n[0]],{stroke:COL.ink,w:2.5,fill:'#fff'})+label(x+w/2,y+h/2-4,t,{size:22,anchor:'middle',fill:COL.ink,weight:650})+label(x+w/2,y+h/2+24,sub,{size:19,anchor:'middle'});};
const zbox=(x,y,w,h,lines,col)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="#fff" stroke="${col}" stroke-width="3"/>`+
 lines.map((t,i)=>label(x+w/2,y+h/2+8+(i-(lines.length-1)/2)*28,t,{size:21,anchor:'middle',fill:col,weight:650})).join('');
const pipe=(y,recon,enc,code,extra)=>{const a=(x1,x2)=>arrow(x1,y+IM/2,x2,y+IM/2,{w:2.5,head:11});
 return img('x3',0,y,IM,IM)+`<rect x="0" y="${y}" width="${IM}" height="${IM}" fill="none" stroke="${COL.target}" stroke-width="3"/>`+a(126,164)+
  trap(170,y,160,IM,'in','encoder',enc)+a(334,374)+code+a(606,646)+trap(652,y,160,IM,'out','decoder',`${A[2]} → ${A[1]} → ${A[0]}`)+a(816,854)+
  img(recon,860,y,IM,IM)+`<rect x="860" y="${y}" width="${IM}" height="${IM}" fill="none" stroke="${COL.guess}" stroke-width="3"/>`+extra;};
const AEY=36,VAY=226;
const aeFig=label(0,24,'autoencoder',{size:21,fill:MUTED,weight:650})+
 pipe(AEY,'ae3',`${A[0]} → ${A[1]} → ${A[2]}`,zbox(380,AEY+25,220,70,[`z = (${V.ae_z.map(v=>num(v)).join(', ')})`],COL.ink),
  gat(2,label(1000,AEY+44,'squared error',{size:21,fill:COL.loss,weight:650})+label(1000,AEY+78,num(V.ae_sse),{size:26,fill:COL.loss,weight:700})))+
 gat(3,label(0,VAY-12,'VAE',{size:21,fill:MUTED,weight:650})+
  pipe(VAY,'vae3',`${A[0]} → ${A[1]} → ${2*A[2]}`,zbox(380,VAY+8,220,104,[`μ = (${V.mu.map(v=>num(v,3)).join(', ')})`,`σ = (${V.sd.map(v=>num(v,3)).join(', ')})`],COL.guess),
   label(1000,VAY+44,'decoded',{size:21,fill:COL.guess,weight:650})+label(1000,VAY+76,'from z = μ',{size:21,fill:COL.guess,weight:650})));
S_('ae-vae','genm','An autoencoder, and the VAE that grows out of it',{builds:3,from:[25,'#ae-loss/3'],
 body:pfig(1152,VAY+IM+4,aeFig,`Our 3 through the autoencoder, ${A.join(' to ')} and back, and through the VAE, whose encoder returns a mean and a width`),
 lines:[`Lecture 25’s autoencoder on MNIST test digit #${V.index}, our ${V.label}: ${A[0]} pixels squeezed into ${word(A[2])} numbers and rebuilt. What does its loss compare?`,
  `The loss is the squared error between the image and its rebuild, summed over all ${A[0]} pixels. Random codes still decode to junk: nothing says where codes should go.`,
  'The VAE’s encoder returns a small cloud: a mean and a width for each code number. In training, the decoder gets a draw z = μ + σε.'],
 ask:['What are the input and the target of an autoencoder?','Why can this autoencoder not generate new digits?','Why give each image a cloud of codes rather than one point?'],
 expected:['The same image: x goes in and x is the target. There are no labels.',
  'Nothing in its loss says where the codes should be. Its codes for real digits sit far from where 𝒩(0, I) draws land, and the decoder was never trained there.',
  'So that the decoder learns to draw something sensible everywhere near a code, not only at one point. With a penalty that holds the clouds near 𝒩(0, I), random codes then decode to digits.'],
 point:'The VAE keeps the autoencoder’s shape but makes the code a sampled draw from a cloud that the encoder predicts.',
 misconception:'That an autoencoder with a small reconstruction error is a good generator. The best rebuilder in Lecture 25, with the longest code, gives static from random codes.',
 transition:'The three fixes give a loss, and it is the ELBO of 6.2.',source:SRC_VAE});

// ---------------------------------------------------------------------------------------- 6.6 the VAE's loss, and β
chap('6.6','Loss: reconstruction plus KL');
const BV=BE.values,SQ=170,BX=v=>20+(v+4)/8*SQ,BYc=v=>40+SQ-(v+4)/8*SQ,SH=SQ;
const outside=k=>BE.codes[k].filter(([a,b])=>Math.abs(a)>4||Math.abs(b)>4).length;
const betaStatic=label(20,24,`codes of ${BE.codes[BE.default].length} test digits`,{size:20,fill:MUTED,weight:650})+
 `<rect x="20" y="40" width="${SQ}" height="${SQ}" fill="#fff" stroke="${AX}" stroke-width="1.5"/>`+line(BX(0),40,BX(0),40+SQ,{stroke:AX,w:1})+line(20,BYc(0),20+SQ,BYc(0),{stroke:AX,w:1})+
 `<circle cx="${BX(0)}" cy="${BYc(0)}" r="${BX(2)-BX(0)}" fill="none" stroke="${COL.target}" stroke-width="2.5" stroke-dasharray="6 5"/>`+
 label(290,24,'decoded from z ~ 𝒩(0, I)',{size:20,fill:MUTED,weight:650});
function betaLive(k){
 const C=BE.codes[k],out=outside(k);
 return C.slice(1).filter(([a,b])=>Math.abs(a)<=4&&Math.abs(b)<=4).map(([a,b])=>`<circle cx="${BX(a).toFixed(1)}" cy="${BYc(b).toFixed(1)}" r="3" fill="${MUTED}" fill-opacity=".45"/>`).join('')+
  (Math.abs(C[0][0])<=4&&Math.abs(C[0][1])<=4?dot(BX(C[0][0]),BYc(C[0][1]),7,{fill:COL.guess}):'')+
  samples(k,290,40,SH)+
  label(290+sampleW(SH)+40,68,`β = ${+k}`,{size:26,fill:COL.ink,weight:700})+
  label(290+sampleW(SH)+40,104,`KL per image: ${num(BE.kl[k])}`,{size:21,fill:COL.loss,weight:650})+
  label(290+sampleW(SH)+40,136,`squared error: ${num(BE.sse[k])}`,{size:21,fill:COL.ink})+
  label(290+sampleW(SH)+40,168,out?`${out} codes lie off this square`:'every code is on the square',{size:19,fill:MUTED})+
  label(290+sampleW(SH)+40,198,'dashed circle: 2 sd of 𝒩(0, I)',{size:19,fill:COL.target});
}
const BWW=1152,BWH=216,betaAlt='The codes of 400 test digits and eight decoded samples from a VAE trained with the chosen KL weight beta';
register('p6-beta',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig">${pfig(BWW,BWH,betaStatic+`<g class="live">${inner}</g>`,betaAlt)}</div>`;
 if(ctx.printing){el.innerHTML=frame(betaLive(BE.default));return;}
 el.innerHTML=frame('')+`<div class="w-controls"><span>weight on the KL</span>${BV.map(k=>`<button type="button" data-beta="${k}" aria-pressed="${k===BE.default}">β = ${+k}</button>`).join('')}</div>`;
 const live=el.querySelector('.live');let k=BE.default;
 const show=()=>{live.innerHTML=betaLive(k);};
 for(const b of el.querySelectorAll('button[data-beta]'))b.addEventListener('click',()=>{k=b.dataset.beta;
  for(const o of el.querySelectorAll('button[data-beta]'))o.setAttribute('aria-pressed',o===b);b.blur();show();});
 show();
});
const S2=V.sigma_pix;
S_('vae-loss','genm','The VAE’s loss: reconstruction plus KL',{builds:5,from:[25,'#vi-kl/5'],
 body:stack(`<div class="code-sm">${codeSteps([[1,'x_hat, mu, logvar = model(x)'],[2,`recon = ((x_hat - x)**2).sum(1) / (2 * ${S2}**2)`],
   [3,'kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(1)'],[4,'loss = (recon + kl).mean()']])}</div>`,
  `<div class="state-slot">${only('1-4',stack(
   eq(R`\lcol{-\ELBO(x)}=\E_q\Big[\tfrac{1}{2\cdot${S2}^2}\big\lVert\pcol{x}-\qcol{f_\theta(z)}\big\rVert^2\Big]+\KL\big(q_\phi(z\mid x)\,\|\,\N(0,I)\big)+\text{const}`),
   `<div class="p6-tiles">${at(2,`<div class="p6-tile">${small('reconstruction, our 3')}${m(R`\E_q[\ln p(\pcol x\mid z)]=${tn(V.fit)}`)}${small(`= ${num(V.const)} − ${num(V.weight,1)} × ${num(V.exp_sse)}`)}</div>`)}${
    at(3,`<div class="p6-tile">${small('KL term, our 3')}${m(R`\KL=${tn(V.kl)}`)}</div>`)}${
    at(4,`<div class="p6-tile">${small('the budget, our 3')}${m(R`\qcol{\ELBO}=${tn(V.fit)}-${tn(V.kl)}=\qcol{${tn(V.elbo)}}`)}${m(R`\dcol{\ln p(x)}=\dcol{${tn(V.log_px)}},\ \text{gap }\lcol{${tn(V.gap)}}`)}</div>`)}</div>`))}${
   only(5,widget('p6-beta',betaAlt))}</div>`),
 lines:['Lecture 25’s loss is four lines of code. The model returns the rebuilt image and the encoder’s cloud, its mean and log-variance.',
  `Gaussian pixels with σ = ${S2} turn the log-likelihood into squared error times ${num(V.weight,1)}, plus a constant. Here is that term for our 3.`,
  'The KL to 𝒩(0, I) has a closed form for two Gaussians. It pulls every image’s cloud towards the prior.',
  'Together they are minus 6.2’s ELBO, for one image. It sits just below ln p(x); the gap is the KL to the true posterior.',
  'Weight the KL by β and train again. Click through the seven trained VAEs: when do the codes fill the circle, and when do samples look like digits?'],
 ask:['Which variable in the first line is the cloud of 6.5?','Why does a squared error appear in a log-likelihood?','What does the KL term do to the codes of all the images together?','We minimise the loss. What happens to the gap, the KL to the posterior?','At which β do the samples look best, and what do the codes do at β = 0?'],
 expected:['mu and logvar: the encoder’s q(z | x). The draw z = mu + exp(logvar / 2) · eps happens inside model(x).',
  `The log of a Gaussian density is minus the squared error over 2σ², plus a constant: the same reason squared error is the Gaussian’s negative log-likelihood in Part 1. Here 1 / (2 · ${S2}²) = ${num(V.weight,1)}.`,
  'Each cloud is pulled towards 𝒩(0, I), so together they fill the region that random draws land in. That is what lets the decoder generate.',
  `ln p(x) does not depend on q, so pushing the ELBO up pushes the gap down. For our 3 it is ${num(V.gap)} nats: the encoder’s q is close to the posterior.`,
  `Around β = ${+BV[2]} to ${+BV[3]}. At β = ${+BV[0]} there is no pull: the KL per image is ${num(BE.kl['0'],1)}, ${outside(BV[0])} of the ${BE.codes[BV[0]].length} codes leave the square and samples from 𝒩(0, I) are poor.`],
 point:'The VAE is trained on minus the ELBO for each image: a reconstruction term from Gaussian pixels plus the KL of the encoder’s q to 𝒩(0, I).',
 misconception:'That the KL term is a regulariser bolted on by hand. It falls out of the ELBO of 6.2, with the code z in the place of the weights θ.',
 demo:'At click 5, start at β = 1, then click 0 and 256: watch the codes spread off the square, then collapse onto the centre. Keep 256 for the check that follows.',
 transition:'We end with a question on the two ends of that sweep.',source:SRC_VAE});

// ---------------------------------------------------------------------------------------- check
chap();
const stripFig=(k,alt)=>`<div class="p6-strip">${pfig(Math.ceil(sampleW(132))+2,134,samples(k,1,1,132),alt)}</div>`;
const B1=BE.default,B2=BV.at(-1);
S_('check-6','genm','Check: the two ends of the β sweep',{builds:2,from:[25,'#summary/2'],
 body:stack(`<p class="p6-q">With β = ${+B2} the KL term is ${num(BE.kl[B2])} nats per image. What do the generated digits look like, and why?</p>`,
  cols(stack(small(`β = ${+B1}: KL ${num(BE.kl[B1])} per image`),stripFig(B1,`Eight digits decoded from random codes, beta = ${+B1}`)),
   at(2,stack(small(`β = ${+B2}: KL ${num(BE.kl[B2])} per image`),stripFig(B2,`Eight decoded samples at beta = ${+B2}, all the same blur`))),'1fr 1fr'),
  answer(2,`All show one blur, the average digit. A KL of 0 means q(z | x) = 𝒩(0, I) for every image: the code says nothing about x, so the decoder draws the mean.`)),
 lines:['Answer in pairs before the click. You need two ideas from this part: when a KL is zero, and what the decoder can learn from a code.',
  `Rebuilds fail too: squared error rises from ${num(BE.sse[B1])} at β = ${+B1} to ${num(BE.sse[B2])} at β = ${+B2}.`],
 ask:[`With β = ${+B2} the KL is ${num(BE.kl[B2])}. What do the samples look like, and why?`,'Is this a failure of training?'],
 expected:[`All ${word(D.images.samples)} are one blurry digit. KL(q ‖ prior) = 0 only if q equals 𝒩(0, I) for every image, so z carries no information about x; the best the decoder can do is draw the average image.`,
  'No: it is the best answer to the loss we wrote. A KL weight that large makes any information in the code cost more than it saves in reconstruction. This is called posterior collapse.'],
 point:'The KL term trades information in the code against closeness to the prior; with too much weight on it, the code is ignored.',
 transition:'Part 6 in one slide.',source:SRC_VAE});

// ---------------------------------------------------------------------------------------- recap
recap(6,[
 `${m(R`\KL(p\,\|\,q)`)}: the extra cost of q when the data come from p.`,
 `No posterior? Raise the ELBO: ${m(R`\ln p(\mathcal D)=\ELBO+\KL(q\,\|\,\text{posterior})`)}.`,
 `${m(R`\theta=\mu+\sigma\epsilon`)} passes gradients through a sample; the VAE trains on minus the ELBO.`],
 'one more change of output, from one label to a list of objects.',
 {card:RECIPES.vae,title:'Variational autoencoder',
  ask:'Which rows of the card are new compared with the classifier of the opening?',
  expected:'Data has no labels; the model has an encoder and a decoder with a sampled code; the output is a distribution over codes and images; the loss is minus the ELBO; at test time we sample a code and decode it. Only the update is the same gradient step.',
  point:'Generative modelling changed every row of the recipe except the update.',
  transition:'Part 7: object detection.'});
