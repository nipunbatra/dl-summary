// The wrap-up: threads that run through all seven parts, the ten recipes side by side, the recipe as a flowchart for
// a new problem, and the course on one slide. The appendix lists every recipe card and every lecture with its links.
// Numbers come from data/p8.json (scripts/sci/p8.py), which collects them from the parts' own data files.
import {S_,O,cols,card,answer,m,R,data,need,course,MODELS,RECIPES,PARTS,SLOTS,word,Word,css,kp,kl,
 fig,label,path,dot,panel,chart,widget,register,gat,num,COL,MUTED,AX,esc} from '../kit.js';

const D=need(await data('p8'),'points','worst','curve','y_max','vit','vlm'),PT=D.points,V=D.vit,L=D.vlm;
const lecs=a=>a.length===1?String(a[0]):a.length===2&&a[1]===a[0]+1?`${a[0]} and ${a[1]}`:`${a[0]} to ${a.at(-1)}`;
const Lec=a=>`Lecture${a.length>1?'s':''} ${lecs(a)}`;

css(`.p8-ce td,.p8-ce th{white-space:nowrap}.p8-ce td:nth-child(3){white-space:normal}
.p8-ce a,.p8-rows a,.p8-lect a{color:var(--ink);text-decoration:none;font-weight:650}.p8-ce a:hover,.p8-rows a:hover{text-decoration:underline}
.p8-ce td:nth-child(n+4){text-align:right;font-variant-numeric:tabular-nums}
.scene[data-build="3"] .p8-ce tr.p8-worst td{background:#BE123C12}
.p8-rows td,.p8-rows th{text-align:center}.p8-rows td:first-child,.p8-rows th:first-child{text-align:left;white-space:nowrap}
.p8-rows td.p8-on::before{content:"";display:inline-block;width:54px;height:20px;border-radius:6px;background:var(--q);opacity:.8;vertical-align:middle}
.p8-rows tr.p8-count td{color:var(--muted);font-weight:650;border-bottom:0}
.scene[data-build="2"] .p8-rows .p8-step{background:#8B2CDE14}
.p8-master{display:grid;grid-template-columns:330px 1fr;gap:26px;align-items:start}
.p8-list.w-controls{flex-direction:column;align-items:stretch;gap:5px;flex-wrap:nowrap}
.p8-list.w-controls button{text-align:left;border-radius:10px;padding:5px 14px}
.p8-master .answer{margin-top:12px;font-size:22px}
.p8-lect td:first-child{white-space:nowrap;text-align:right}.p8-lect td:nth-child(2){width:44%}
.p8-lect a{color:var(--p);font-weight:600;margin-right:12px;white-space:nowrap}`);

// ---------------------------------------------------------------------------------------- one loss, five times
const CW=470,CH=390;
const CP=panel({W:CW,H:CH,L:62,R:18,T:40,B:62,xMin:0,xMax:1,yMin:0,yMax:D.y_max,ticksX:[0,.5,1],ticksY:[0,1,2,3],
 xLabel:'p, given to what was observed',yLabel:'loss = −ln p'});
const OFF=[[14,-12,'start'],[16,6,'start'],[-12,26,'end'],[16,6,'start'],[12,-14,'start']];   // where each dot's number sits, so no two touch
if(OFF.length!==PT.length)throw new Error('p8: one label offset per point');
const ceFig=fig(CW,CH,CP.axes+path(D.curve.map(([q,v])=>[CP.x(q),CP.y(Math.min(v,D.y_max))]),{stroke:COL.loss,w:4})+
 gat(2,PT.map((q,i)=>dot(CP.x(q.p),CP.y(q.loss),8,{fill:COL.ink,stroke:'#fff',w:2})+
  label(CP.x(q.p)+OFF[i][0],CP.y(q.loss)+OFF[i][1],i+1,{size:21,fill:COL.ink,weight:700,anchor:OFF[i][2],halo:true})).join('')),
 'The curve of minus log p against p, with the five losses of the table as numbered points on it');
const ceTable=`<div class="table-sm tight p8-ce"><div class="table-scroll"><table><thead><tr><th></th><th>Lecture</th><th>What was observed</th><th>p</th><th data-at="2">−ln p</th></tr></thead><tbody>${
 PT.map((q,i)=>`<tr${i===D.worst?' class="p8-worst"':''}><td>${i+1}</td><td>${lecs(q.lec)}</td><td><a href="#${q.slide}">${esc(q.what)}</a></td><td>${kp(num(q.p))}</td><td data-at="2">${kl(num(q.loss))}</td></tr>`).join('')}</tbody></table></div></div>`;
const WORST=PT[D.worst];
S_('threads-ce','wrap',`The same loss, ${word(PT.length)} times`,{builds:3,
 body:cols(ceFig,`${ceTable}${answer(3,`Row ${D.worst+1}, the character model of ${Lec(WORST.lec)}. Its ${kp('p = '+num(WORST.p))} is the smallest, so its loss ${kl(num(WORST.loss))} is the largest.`)}`,'1fr 1.42fr'),
 lines:[`${Word(PT.length)} models from ${word(PT.length)} parts of the course. Each gave a probability ${kp('p')} to what was observed. Which one was penalised most?`,
  `Each loss is ${kl('−ln p')}, so all ${word(PT.length)} sit on one curve.`,
  'The loss reads one number, the probability given to what was observed.'],
 ask:['Without computing anything: how do you rank the five?','Where on this curve is a model that is certain and right? Certain and wrong?',
  'Squared error and the ELBO are not on this slide. Do they belong on the curve?'],
 expected:['By p alone. The smallest p gets the largest penalty, whatever the model is.',
  'Certain and right is p = 1, where the loss is 0. Certain and wrong is p near 0, where the curve rises without limit.',
  'They are the same idea with a density in place of a probability: squared error is −ln of a Gaussian density (Lectures 1 and 2), and the ELBO is a bound on ln p(𝒟) (Lecture 24). A density can exceed 1, so those losses can be negative and do not sit on this curve.'],
 point:'Across the course the loss was −ln of the probability given to what was observed.',
 misconception:'That CLIP, a language model and a classifier use different losses. They differ in what the probability is over.',
 demo:'Each row of the table opens the slide its number came from.',
 transition:'The loss is one thread. The attention block is another.',
 source:'Every number is read from this deck’s Parts 1, 4, 5 and 6 (scripts/sci/p8.py).'});

// ---------------------------------------------------------------------------------------- one block, four models
const AW=1152,RH=92,AT=44,AF=20,BW=52,BH=42,BG=6,TX=290,MX=805,OX=915;
const TOK={w:COL.data,p:COL.amber,c:COL.guess};
/** A strip of tokens: [kind, text, read]. kind: w a word piece, p an image patch, c [CLS]; read marks where the output is taken. */
const toks=(x,y,ts)=>ts.map(([k,t,read],i)=>{const bx=x+i*(BW+BG);
 return `<rect x="${bx}" y="${y-BH/2}" width="${BW}" height="${BH}" rx="7" fill="${TOK[k]}" fill-opacity=".16" stroke="${read?COL.ink:TOK[k]}" stroke-width="${read?4:2}"/>`+
  (t?label(bx+BW/2,y+6,esc(t),{size:17,anchor:'middle',fill:COL.ink}):'');}).join('');
/** A 4 by 4 picture of the mask: a filled cell means the row's token may attend to the column's. */
const mask=(x,y,causal)=>Array.from({length:16},(_,k)=>{const i=k>>2,j=k&3,on=!causal||j<=i;
 return `<rect x="${x+j*15}" y="${y-30+i*15}" width="13" height="13" rx="2" fill="${on?COL.ink:'#fff'}" fill-opacity="${on?.72:1}" stroke="${on?'none':AX}"/>`;}).join('');
const ry=i=>AT+RH/2+i*RH;
const arow=(i,name,lec,strip,masks,read1,read2)=>label(0,ry(i)-4,name,{size:AF,fill:COL.ink,weight:650})+label(0,ry(i)+22,Lec(lec),{size:17})+strip+masks+
 label(OX,ry(i)+(read2?-4:7),read1,{size:AF,fill:COL.ink})+(read2?label(OX,ry(i)+22,read2,{size:AF,fill:COL.ink}):'');
const P3=[['p',''],['p',''],['p','']];
const attnFig=fig(AW,AT+4*RH+4,
 label(TX,22,'tokens',{size:18,weight:700})+label(MX,22,'mask',{size:18,weight:700})+label(OX,22,'the output is read at',{size:18,weight:700})+
 arow(0,'Language model',[12,13,14],toks(TX,ry(0),['the','sun','rose','in','the','east'].map(t=>['w',t,true])),mask(MX,ry(0),true),'every position:','the next token')+
 gat(2,arow(1,'Vision Transformer',[16],toks(TX,ry(1),[['c','CLS',true],...P3,['p',''],['p','']])+label(TX+6*(BW+BG)+6,ry(1)+7,`… ${V.patches} patches`,{size:17,fill:COL.ink}),mask(MX,ry(1),false),'[CLS]:','the class'))+
 gat(3,arow(2,'CLIP',[17],toks(TX,ry(2),[['c','CLS',true],['p',''],['p','']])+toks(TX+3*(BW+BG)+26,ry(2),[['w','a'],['w','black'],['w','dog'],['w','end',true]]),
  mask(MX-35,ry(2),false)+mask(MX+33,ry(2),true),'two vectors, compared','by a dot product'))+
 gat(4,arow(3,'Vision-language model',[18,19],toks(TX,ry(3),[...P3,['p',''],['w','what'],['w','is'],['w','this'],['w','?',true]]),mask(MX,ry(3),true),'the last position:','the next answer token')),
 'Four models built from the same attention block: what their tokens are, which mask they use and where the output is read');
S_('threads-attn','wrap','The same block, in four models',{builds:4,
 body:`${attnFig}<p class="caption" style="margin-top:4px">${kTok('data','a word piece')} ${kTok('noise','an image patch')} ${kTok('q','[CLS]')} · dark outline: the output is read here · filled mask cell: may attend</p>`,
 lines:['Part 4 built one block: self-attention under a mask. A language model hides the future and predicts the next token at every position.',
  `Lecture 16 changed what a token is. A photo became ${V.patches} patches and a [CLS] token; nothing is hidden, and the class is read at [CLS].`,
  'CLIP runs two encoders, one for the photo and one for the caption, and compares the two vectors they return.',
  `A vision-language model puts the ${L.rows} patch rows in front of the question, in one sequence. The answer comes out at the last position, one token at a time.`],
 ask:['Why must a language model hide the future during training?','Why does the Vision Transformer need no mask?',
  'Which of CLIP’s two encoders uses a causal mask?','What does the projector do before the patch rows enter the language model?'],
 expected:['The target at each position is the next token. If the block could see it, the model would copy it and learn nothing.',
  'It predicts one class for the whole photo, so every patch may look at every other patch. There is no future to leak.',
  'The text encoder, which reads the caption left to right and returns the vector at its last token. The image encoder is a Vision Transformer read at [CLS].',
  `It changes their width from ${L.d_v} to ${L.d_lm} so they fit the language model. Their number stays ${L.rows}.`],
 point:'One block served four models. They differ in what a token is, which mask is used and where the output is read.',
 misconception:'That vision models need a different architecture from language models. After Lecture 16 the block is the same.',
 transition:'Two threads, then. Now every part against the six rows of the card.',
 source:'Parts 4 and 5 of this deck; Lectures 12 to 19.'});
function kTok(v,s){return `<span style="display:inline-block;width:15px;height:15px;border-radius:4px;border:2px solid var(--${v});background:color-mix(in srgb,var(--${v}) 16%,transparent);vertical-align:-2px"></span> ${s}`;}

// ---------------------------------------------------------------------------------------- the parts against the six rows
const nChanged=k=>PARTS.filter(t=>t[3].includes(k)).length;
const rowsTable=`<div class="table-sm tight p8-rows"><div class="table-scroll"><table><thead><tr><th>Part</th><th>Lectures</th>${SLOTS.map(([k,n])=>`<th${k==='step'?' class="p8-step"':''}>${n}</th>`).join('')}</tr></thead><tbody>${
 PARTS.map((t,i)=>`<tr><td><a href="#recap-${i+1}">${i+1} · ${t[1]}</a></td><td>${t[2]}</td>${SLOTS.map(([k,n])=>
  `<td class="${t[3].includes(k)?'p8-on':''}${k==='step'?' p8-step':''}"${t[3].includes(k)?` aria-label="changed ${n}"`:''}></td>`).join('')}</tr>`).join('')}
 <tr class="p8-count"><td>Parts that changed it</td><td></td>${SLOTS.map(([k])=>`<td${k==='step'?' class="p8-step"':''}>${nChanged(k)}</td>`).join('')}</tr></tbody></table></div></div>`;
S_('parts-rows','wrap',`${Word(PARTS.length)} parts, six rows`,{builds:2,
 body:`${rowsTable}${answer(2,`Part 2 settled it. Backprop and an optimizer work for any differentiable loss, so the later parts changed the loss and left the step alone.`)}`,
 lines:['A bar marks a row of the card that the part changed. Read down the Update column: why is it almost empty?',
  'This is why one training loop could serve every model of the course.'],
 ask:['Which rows changed most often?','Part 6 sent the gradient through a sample, θ = μ + σε. Is that a new update rule?'],
 expected:[`Data, Output, Loss and At test time: ${word(nChanged('data'))} parts each. A new task mostly means new data, a new output and a new use of it.`,
  'No. The reparameterisation changes how the loss is written so that backprop can reach μ and σ. The step is still a gradient step, which is why Part 6 has no bar in that column.'],
 point:'New tasks changed the data, the output and the loss. The update was built once, in Part 2.',
 misconception:'That each architecture needs its own optimizer. Adam trained the classifier, the language model and the VAE alike.',
 demo:'Each part’s name opens its recap slide.',
 transition:'The same view, one model at a time.'});

// ---------------------------------------------------------------------------------------- the ten recipes
const stepSame=r=>r.step===RECIPES.mlp.step||r.step===RECIPES.charlm.step;
const changed=id=>id==='mlp'?[]:SLOTS.map(s=>s[0]).filter(k=>k==='step'?!stepSame(RECIPES[id]):RECIPES[id][k]!==RECIPES.mlp[k]);
const cardOf=(M,sm=false,hi=changed(M.id))=>card(RECIPES[M.id],{hi,sm,title:`${M.name} · ${Lec(M.lec)}`});
const lower=s=>/^(Bayesian|CLIP|MLP|Vision T)/.test(s)?s:s.replace(/^./,c=>c.toLowerCase());
const FEW=MODELS.slice(1).reduce((a,M)=>changed(M.id).length<changed(a.id).length?M:a);
if(FEW.id!=='vit')throw new Error('p8: the master slide says the Vision Transformer has the fewest changed rows');
const newStep=MODELS.filter(M=>changed(M.id).includes('step'));
const masterAlt=`The recipe cards of the ${word(MODELS.length)} models, one at a time; rows that differ from the MLP classifier’s are tinted`;
let chosen=MODELS[0].id;
register('p8-recipes',(el,ctx)=>{
 const cards=[...el.closest('.p8-master').querySelectorAll('[data-card]')];
 const show=()=>{for(const c of cards)c.hidden=c.dataset.card!==chosen;
  for(const b of el.querySelectorAll('button[data-model]'))b.setAttribute('aria-pressed',b.dataset.model===chosen);};
 if(ctx.printing){el.innerHTML='';show();return;}
 el.innerHTML=`<div class="w-controls p8-list">${MODELS.map(M=>`<button type="button" data-model="${M.id}" aria-pressed="false">${M.name}</button>`).join('')}</div>`;
 for(const b of el.querySelectorAll('button[data-model]'))b.addEventListener('click',()=>{chosen=b.dataset.model;b.blur();show();});
 show();
});
S_('master','wrap',`${Word(MODELS.length)} models, one card each`,{builds:2,
 body:`<div class="p8-master">${widget('p8-recipes',masterAlt)}<div>${MODELS.map((M,i)=>`<div data-card="${M.id}"${i?' hidden':''}>${cardOf(M)}</div>`).join('')}
  ${answer(2,`${Word(newStep.length)}: the ${newStep.map(M=>lower(M.name)).join(' and the ')}. Both draw a sample inside the model, and the gradient has to pass through it.`)}</div></div>`,
 lines:[`Pick a model. Its card shows the six rows, and the tinted rows are the ones that differ from the MLP classifier’s. How many models tint the Update row?`,
  'Even there it is a gradient step. Everything else that changed is data, model, output, loss or use.'],
 ask:['Which model has the fewest tinted rows, and why?','Which models need no labels at all?'],
 expected:[`The ${lower(FEW.name)}, with ${word(changed(FEW.id).length)}. It is a classifier like the MLP, so its loss and its decision are the MLP’s.`,
  'The language models, CLIP and the VAE. Their targets come from the data itself: the next token, the paired caption, the image.'],
 point:'Every model of the course is the same card with some rows rewritten.',
 misconception:'That a tinted row means a new idea each time. The tinted Loss rows of the language models, CLIP and the vision-language model are all −ln p of something observed.',
 demo:'Click down the list in order. Stop at CLIP (no class labels) and at the VAE (no labels at all).',
 transition:'The card also works forwards, for a problem the course never covered.'});

// ---------------------------------------------------------------------------------------- the recipe as a flowchart, for a new problem
const boxes=(ts,col)=>SLOTS.map(([,n],i)=>({k:n,t:ts[i],col}));
S_('pipeline','wrap','The recipe, for a problem of your own',{builds:3,
 body:chart([
  {name:'Six questions',steps:boxes(['What is\nobserved?','What turns x\ninto scores?','Which\ndistribution\nover y?','−ln p of\nwhat was\nobserved','A gradient\nstep','How is the\noutput used?'])},
  {name:'What the course put in each box',at:2,steps:boxes(['labels, tokens,\ncaptions, boxes','MLP, attention\nblocks, encoders','softmax,\nGaussian,\na list of boxes','cross-entropy,\nsquared error,\n−ELBO','SGD,\nmomentum,\nAdam','pick, sample,\ncorrect, filter'])},
  {name:'Tomorrow’s PM2.5, with an error bar',at:3,steps:boxes(['today’s weather,\ntomorrow’s\nPM2.5','an MLP','a Gaussian:\nmean and\nspread','−ln of its\ndensity at the\nmeasured value','Adam, with\na schedule','mean ± two\nspreads'],COL.guess)}],
  {W:1152,bh:86,gx:18,gy:14,fs:18,alt:'The recipe as a flowchart of six boxes, filled three ways: as questions, with what the course used, and for a PM2.5 forecast'}),
 lines:['The card as a flowchart. Each box is a question to answer before you write any code.',
  'These are the answers the course gave. Now a new problem: forecast tomorrow’s PM2.5 and say how sure you are. What goes in each box?',
  'One good answer. The Gaussian output carries the error bar, and its loss is squared error divided by the predicted spread, plus a term for the spread.'],
 ask:['Which box would you answer first?','For the forecast, what is observed and what must the model output?','How does the network keep the spread positive?'],
 expected:['The first and the third: what is observed, and which distribution the model puts over it. The loss follows from those two.',
  'Observed: tomorrow’s measured PM2.5, a number. Output: a distribution over that number, so a mean and a spread.',
  'A softplus or an exponential on the raw output, the “valid outputs by design” of Lecture 9.'],
 point:'A new problem is a new set of answers to the same six questions.',
 misconception:'That uncertainty needs a Bayesian network. A Gaussian output head already gives an error bar for the noise in the data.',
 transition:'The course, once more, on one slide.',
 source:'Lectures 1 to 3 for the Gaussian likelihood; Lecture 9 for the positive spread.'});

// ---------------------------------------------------------------------------------------- the course on one slide
S_('summary','wrap','The course on one slide',{builds:1,
 body:`<div class="sum3"><div class="st"><h3>One loss</h3><ul>
   <li>The model outputs a probability distribution.</li>
   <li>The loss is ${m(R`\lcol{-\ln p}`)} of what was observed.</li></ul></div>
  <div class="sd"><h3>One update</h3><ul>
   <li>Backprop gives the gradient for every parameter.</li>
   <li>SGD, momentum or Adam takes the step.</li></ul></div>
  <div class="sm"><h3>${Word(MODELS.length)} models</h3><ul>
   <li>Attention reads characters, word pieces and image patches.</li>
   <li>Each model fills the same six rows of the card.</li></ul></div></div>`,
 lines:[`${Word(course.count)} lectures as one method. For your own problem, which row of the card comes first?`],
 ask:['Take a problem from your own work. What is observed, and which distribution over it would your model output?'],
 expected:['Any answer that names what is observed and a distribution over it: a class and a softmax, a number and a Gaussian, a sentence and a next-token distribution. The loss then follows as −ln p of what was observed.'],
 point:`Choose what is observed and a distribution over it. The loss and the update follow, as they did for all ${word(MODELS.length)} models.`,
 misconception:'That a new problem needs a new learning algorithm. It needs a new answer to the six rows.',
 transition:'The appendix has every recipe card and every lecture with its links.'});

// ---------------------------------------------------------------------------------------- appendix: every card, every lecture
for(const M of MODELS)O(`p8-card-${M.id}`,`Recipe card: ${lower(M.name)}`,{builds:1,
 body:`<div style="max-width:900px;margin:0 auto">${cardOf(M)}</div>`,
 ask:['Which rows differ from the MLP classifier’s card?'],
 expected:[M.id==='mlp'?'None. This is the card the others are compared with.':`The tinted ones: ${changed(M.id).map(k=>SLOTS.find(t=>t[0]===k)[1]).join(', ')}.`]});
const SCHED='https://nipunbatra.github.io/dl-2026/schedule.html';
const KEEP=['notes','slides','cheat','pdf','recording','notebook'];   // the videos of single ideas and the labs stay on the course page
const CUTS=[[1,5],[6,10],[11,15],[16,20],[21,course.count]];
CUTS.forEach(([a,b],i)=>{
 const ls=course.lectures.filter(l=>l.n[0]>=a&&l.n[0]<=b);
 O(`p8-lectures-${i+1}`,`Lectures ${a} to ${b}, with their material`,{builds:1,
  body:`<div class="table-sm xtight p8-lect"><div class="table-scroll"><table><thead><tr><th></th><th>Lecture</th><th>Material</th></tr></thead><tbody>${
   ls.map(l=>`<tr><td>${l.n.join(', ')}</td><td><a style="color:var(--ink)" href="${SCHED}#lecture-${l.n[0]}" target="_blank" rel="noopener">${esc(l.title.replace(' — ',': '))}</a></td><td>${
    l.links.filter(x=>KEEP.includes(x.kind)).map(x=>`<a href="${esc(x.href)}" target="_blank" rel="noopener">${esc(x.label)}</a>`).join(' ')}</td></tr>`).join('')}</tbody></table></div></div>`,
  ask:['Which of these would you reread first?'],
  expected:['Any. Each title opens the lecture’s row on the course page, where the short videos and the labs are also listed.']});});
