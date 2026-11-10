// Part 4: attention and language (Lectures 11 to 15). One task, predicting the next token, and a better way for a
// token to gather its context. Every number comes from data/p4.json (scripts/sci/p4.py).
import {S_,partBreak,recap,chap,p,ps,small,cols,stack,sm,at,only,answer,table,codeSteps,code,m,eq,R,tn,data,need,course,
 RECIPES,word,Word,css,kt,kp,fig,label,line,rect,gat,widget,register,slider,num,big,COL,MUTED,AX,esc} from '../kit.js';

const D=need(await data('p4'),'names','aab','temp','bank','model','leak','scale','mask','recorded','families');
const N=D.names,AB=D.aab,TP=D.temp,BK=D.bank,MD=D.model,LK=D.leak,SC=D.scale,MK=D.mask,TS=D.recorded.tinystories,FM=D.families;
const SRC11='The character model is the one Lecture 11 trained on Indian names (part1.html). Numbers: scripts/sci/p4.py.';
const SRC12='The bank model is the hand-built one of Lectures 12 and 13 (attention.html). Numbers: scripts/sci/p4.py.';
/** A link to a lecture's slides on another page than the first one the schedule lists (Lecture 14 has two). */
const fromPage=(n,page,hash)=>{const l=course.lectures.find(x=>x.n.includes(n)),a=l.links.find(x=>x.kind==='slides'&&x.href.includes(page));
 if(!a)throw new Error(`No ${page} slides for Lecture ${n}`);return {href:a.href.replace(/#.*$/,'')+hash,label:`From Lecture ${n}`};};

css(`.p4-sents{display:grid;gap:22px;font-size:31px;line-height:1.35;margin-top:30px}
.p4-sents>div{display:flex;gap:18px;align-items:baseline}
.p4-sents b{color:var(--muted);font-size:22px;min-width:110px}
.p4-bank{color:var(--q);font-weight:700;text-decoration:underline;text-decoration-thickness:3px;text-underline-offset:5px}
.p4-blank{color:var(--muted)}
.p4-row{margin-top:34px;display:flex;align-items:baseline;gap:22px}.p4-row .formula-scroll{flex:none}
.p4-fam h3{font-size:25px;margin:0 0 4px}
.p4-fam .caption{margin:4px 0 0}
.p4-fam p.small{font-size:21px;line-height:1.35;margin:8px 0 0}
.p4-fam svg{max-height:195px}
.p4-upd table{font-size:22px}.p4-upd td,.p4-upd th{padding:5px 12px;text-align:right}
.p4-upd td:first-child,.p4-upd th:first-child{text-align:left}
.p4-upd .hi{color:var(--d);font-weight:700}`);

// ---------------------------------------------------------------------------------------- a bar chart we can draw ghosts on
/** Vertical bars inside an SVG group. items: [{name, v, col, nameCol, nameWeight}]; ghost: values drawn as dashed
 outlines behind the bars (the same chart at another setting); from: the click at which the bars appear. */
function vbars(items,{ox=0,oy=0,W,H,max,fs=21,gap=.34,ghost=null,dec=2,from=0}){
 const T=fs*1.5,B=fs*1.7,n=items.length,bw=W/n,base=oy+H-B,sc=v=>Math.max(0,v)/max*(H-T-B);
 let g=line(ox,base,ox+W,base,{stroke:AX});
 items.forEach((d,i)=>{const x=ox+i*bw+bw*gap/2,w=bw*(1-gap),h=Math.max(sc(d.v),1.5),top=base-h;
  if(ghost)g+=rect([x,base-sc(ghost[i]),x+w,base],{stroke:MUTED,w:2,dash:'5 4'});
  const b=`<rect x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="4" fill="${d.col||COL.guess}"/>`+
   label(x+w/2,top-8,num(d.v,d.dec??dec),{size:fs,anchor:'middle',fill:COL.ink,weight:600,halo:!!ghost});
  g+=from?gat(from,b):b;
  g+=label(x+w/2,base+fs*1.25,esc(d.name),{size:fs,anchor:'middle',fill:d.nameCol||MUTED,weight:d.nameWeight||''});});
 return g;
}
const spaced=s=>[...s].join(' ');                                        // "aab" is written "a a b", as the lecture does

// ---------------------------------------------------------------------------------------- the break
partBreak(4,'Predicting the next token',
 'This part kept one task, predicting the next token, and changed how the model gathers the context it predicts from.',
 ['Text becomes tokens; each position predicts the next one.','Self-attention: each token mixes the values of the tokens before it.',
  'Mask the future, add heads, stack blocks.','Generate one token at a time.'],
 {line:'By the end you can say how a token reads the others, and why training hides the future.',
  ask:'In Lecture 11, what did the name model see, and what did it output?',
  expected:`The last ${word(N.w)} characters, and a probability for each of the ${AB.n_vocab} characters that could come next.`,
  transition:'Start with the data and the output of that first model.'});

// ---------------------------------------------------------------------------------------- 4.1 next token
chap('4.1','Data and Output: the next token');
const PW=460,ROWH=40,ROW0=76,cx=112,tx=300;
const pairsFig=fig(PW,ROW0+(N.pairs.length-1)*ROWH+18,
 label(cx,26,'context',{size:22,anchor:'middle',weight:650,fill:COL.ink})+label(tx,26,'target',{size:22,anchor:'middle',weight:650,fill:COL.target})+
 line(10,42,PW-10,42,{stroke:AX})+
 N.pairs.map(([c,t],i)=>{const y=ROW0+i*ROWH,on=i===N.focus;
  return (on?rect([8,y-29,PW-8,y+11],{stroke:'none',fill:'#8B2CDE17',rx:8}):'')+
   `<text x="${cx}" y="${y}" font-size="26" text-anchor="middle" fill="${COL.ink}" font-family="Menlo,Consolas,monospace"${on?' font-weight="700"':''}>${esc(spaced(c))}</text>`+
   `<text x="${tx}" y="${y}" font-size="26" text-anchor="middle" fill="${COL.target}" font-family="Menlo,Consolas,monospace" font-weight="700">${esc(t)}</text>`+
   line(196,y-9,250,y-9,{stroke:MUTED,w:2})+`<path d="M250,${y-9} l-10,-6 v12z" fill="${MUTED}"/>`;}).join(''),
 `The name ${N.name} cut into ${word(N.pairs.length)} pairs: ${word(N.w)} characters in, the next one as the target`);
const BW=680,BH=300,SHOWN=TP.order.length,REST=AB.n_vocab-SHOWN;
const restBar=v=>({name:'rest',v,col:'#C9CED4'});
const aabItems=[...AB.top.map(([c,v])=>({name:c,v,dec:c===AB.target?3:2,col:c===AB.target?COL.target:COL.guess,nameCol:c===AB.target?COL.target:MUTED,nameWeight:c===AB.target?700:''})),restBar(AB.rest)];
const aabFig=fig(BW,BH,label(0,24,`p(next character | ${spaced(AB.context)}), the ${word(SHOWN)} largest`,{size:22,fill:COL.ink,weight:600})+
 vbars(aabItems,{oy:34,W:BW,H:BH-34,max:Math.ceil(AB.top[0][1]*10)/10,from:2}),
 `The model's probabilities for the character after ${AB.context}: ${AB.top.slice(0,3).map(([c,v])=>`${c} ${num(v)}`).join(', ')}`);

// the live figure: the same model starts a new name, and a temperature reshapes its next-character probabilities
const pAt=T=>{const z=TP.logits.map(v=>v/T),mx=Math.max(...z),e=z.map(v=>Math.exp(v-mx)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);};
const tempItems=T=>{const q=pAt(T),top=TP.order.map((i,j)=>({name:TP.vocab[i],v:q[i],col:j?COL.guess:COL.guess}));
 return [...top,restBar(1-top.reduce((a,d)=>a+d.v,0))];};
const T1=tempItems(1).map(d=>d.v);
const tempAlt=`Next-character probabilities after ${TP.context} at a chosen temperature, with T = 1 drawn as dashed outlines`;
const tempLive=T=>label(0,24,`p(next | ${spaced(TP.context)}) at T = ${num(T)}: ${TP.top} gets ${num(pAt(T)[TP.vocab.indexOf(TP.top)])}`,{size:22,fill:COL.ink,weight:600})+
 vbars(tempItems(T),{oy:34,W:BW,H:BH-34,max:TP.y_max,ghost:T1});
register('p4-temp',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${BW} ${BH}" role="img" aria-label="${esc(tempAlt)}"><g class="live">${inner}</g></svg></div>`;
 if(ctx.printing){el.innerHTML=frame(tempLive(TP.T_print));return;}
 el.innerHTML=frame('')+`<div class="w-controls">${slider('T','temperature T',TP.T_min,TP.T_max,TP.T_step,1)}<span class="w-hint">dashed: T = 1</span></div>`;
 const live=el.querySelector('.live'),range=el.querySelector('[data-k="T"]'),out=el.querySelector('[data-o="T"]');
 const show=()=>{out.textContent=num(+range.value);live.innerHTML=tempLive(+range.value);};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());
 show();
});

const pT=k=>num(TP.p_top[k]),MK0=Object.keys(TP.p_top);                 // the keys as the script wrote them: '0.5', '1.0', '1.5'
S_('next-token','attn',`One name, ${word(N.pairs.length)} guesses, a loss for each`,{builds:4,from:[11,'#s07'],
 body:cols(stack(pairsFig,at(3,eq(R`\begin{aligned}\lcol{\text{loss}}&=-\ln\qcol{p(\pcol{\text{${AB.target}}}\mid\text{${AB.context}})}\\&=-\ln\qcol{${tn(AB.p_target,3)}}=\lcol{${tn(AB.loss)}}\end{aligned}`))),
  only([1,2,3],aabFig)+only([4],widget('p4-temp',tempAlt)),'0.8fr 1.2fr'),
 lines:[`The name ${N.name}, read ${word(N.w)} characters at a time. Each window gives one training pair. Given ${kt(spaced(AB.context))}, what should the model output?`,
  `A probability for each of the ${AB.n_vocab} characters. The target ${kt(AB.target)} gets ${kp(num(AB.p_target,3))}; the boundary ${AB.top[0][0]}, which ends the name, gets ${num(AB.top[0][1])}.`,
  `The loss is the negative log of the target's probability, in nats. Each of the ${word(N.pairs.length)} pairs adds one such term.`,
  `At test time there is no target, so we sample. The model starts a new name with ${spaced(TP.context)}. Drag the temperature T.`],
 ask:[`How many training pairs does ${N.name} give, and why one more than its length?`,'Why is i not the top bar? Is the model wrong?',
  `What would the loss be if the model gave i a probability of 1?`,'What does a small T do to the bars, and what does a large one do?'],
 expected:[`${Word(N.pairs.length)}: one per character, plus one whose target is the boundary that ends the name.`,
  `No. Many names end after ${spaced(AB.context)}, so the boundary is the more common next character. Before training, i got only ${num(AB.p_target_untrained,3)}.`,
  `Zero. The loss only shrinks to zero when the model is sure of the right answer.`,
  `Small T sharpens: ${TP.top} goes from ${pT(MK0[1])} at T = 1 to ${pT(MK0[0])} at T = ${+MK0[0]}, and T near 0 is the greedy choice. Large T flattens: ${pT(MK0[2])} at T = ${+MK0[2]}, and the rare characters (the rest bar) gain.`],
 point:'The data supplies its own targets: every position of every name is a training pair, and the loss is −ln of the target\'s probability.',
 misconception:'That temperature is used in training. It only changes how we sample at test time; the loss is computed at T = 1.',
 demo:`Leave T at 1, then drag to ${+MK0[0]} and to ${+MK0[2]}. The dashed outlines stay at T = 1, so the class can see the bars sharpen and flatten. "rest" is the other ${REST} characters together.`,
 transition:'The window was three characters. How does a model read a whole sentence?',source:SRC11});

// ---------------------------------------------------------------------------------------- 4.2 bank
chap('4.2','Model: attention');
const BKEYS=Object.keys(BK),BMAX=Math.ceil(Math.max(...BKEYS.flatMap(k=>BK[k].weights))*10)/10;
const sentHTML=k=>{const b=BK[k];return b.words.map((w,i)=>i===b.receiver?`<span class="p4-bank">${esc(w)}</span>`:esc(w)).join(' ')+' <span class="p4-blank">___</span>';};
const KW=1152,KH=270,slot=KW/BK.A.words.length;
const bankAlt='How much the word bank reads each earlier word of its sentence, as bars over the words';
function bankLive(k){
 const b=BK[k],base=KH-58,top=46,sc=v=>v/BMAX*(base-top);
 let g=label(0,24,`Sentence ${k}: the weights bank puts on each word it can read (they sum to 1)`,{size:22,fill:COL.ink,weight:600})+line(0,base,KW,base,{stroke:AX});
 b.words.forEach((w,i)=>{const x=i*slot+slot/2,read=i<=b.receiver,me=i===b.receiver;
  if(read){const h=Math.max(sc(b.weights[i]),1.5),bw=slot*.5;
   g+=`<rect x="${(x-bw/2).toFixed(1)}" y="${(base-h).toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="4" fill="${COL.guess}"/>`+
    label(x,base-h-8,num(b.weights[i]),{size:21,anchor:'middle',fill:COL.ink,weight:600});}
  g+=label(x,base+30,esc(w),{size:23,anchor:'middle',fill:me?COL.guess:read?COL.ink:'#AAB3BC',weight:me?700:''});});
 const x0=(b.receiver+1)*slot;
 g+=label((x0+KW)/2,base-20,'later words: not read',{size:20,anchor:'middle',fill:MUTED})+line(x0+14,base+44,KW-14,base+44,{stroke:'#AAB3BC',w:2});
 return g;
}
register('p4-bank',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${KW} ${KH}" role="img" aria-label="${esc(bankAlt)}"><g class="live">${inner}</g></svg></div>`;
 if(ctx.printing){el.innerHTML=frame(bankLive(BKEYS[0]));return;}
 el.innerHTML=frame('')+`<div class="w-controls"><span>sentence</span>${BKEYS.map((k,i)=>`<button type="button" data-s="${k}" aria-pressed="${!i}">${k}: … ${esc(BK[k].first[0])} … bank</button>`).join('')}</div>`;
 const live=el.querySelector('.live');let k=BKEYS[0];
 const show=()=>{live.innerHTML=bankLive(k);};
 for(const b of el.querySelectorAll('button[data-s]'))b.addEventListener('click',()=>{k=b.dataset.s;
  for(const o of el.querySelectorAll('button[data-s]'))o.setAttribute('aria-pressed',o===b);b.blur();show();});
 show();
});
const [A_,B_]=BKEYS.map(k=>BK[k]);
S_('attn-bank','attn','One word, two sentences',{builds:3,from:[12,'#s10-frame-sources'],
 body:only([1],`<div class="p4-sents">${BKEYS.map(k=>`<div><b>Sentence ${k}</b><span>${sentHTML(k)}</span></div>`).join('')}</div>`+
   `<div class="p4-upd p4-row">${table(['lookup row',...MD.axes],[[m(R`E_{\text{tok}}[\text{bank}]`),...MD.bank_row.map(v=>num(v))]])}</div>`)+
  only([2,3],widget('p4-bank',bankAlt))+
  answer(3,`Bank's message mixes the values of the words it reads: ${m(R`m_${A_.receiver+1}=\sum_{j\le ${A_.receiver+1}}\alpha_{${A_.receiver+1}j}\,v_j`)}. One lookup row gives two mixtures.`),
 lines:[`Both sentences give bank the same lookup row. How can the model tell a river bank from a bank that takes cheques?`,
  `Bank's query is matched with every key up to itself. In A, ${A_.first[0]} gets ${num(A_.first[1])} and ${A_.second[0]} ${num(A_.second[1])}. Switch to B.`,
  `Bank's new row moves toward water in A and toward money in B. Lecture 12 chose the matrices by hand to show this.`],
 ask:['What does an embedding table return for bank in the two sentences?',`In B, which two words does bank read most?`,'Where do the query, key and value of a word come from?'],
 expected:['The same row both times: a lookup cannot see the sentence.',`${B_.first[0]} (${num(B_.first[1])}) and ${B_.second[0]} (${num(B_.second[1])}).`,
  `From the word's own row times three learned matrices, W_Q, W_K and W_V, the same for every position (Lecture 13).`],
 point:'Attention gives each token a weighted average of the tokens before it, with weights computed from the content of the sentence.',
 misconception:'That attention weights are learned parameters. The matrices W_Q, W_K and W_V are learned; the weights are computed afresh for every sentence.',
 demo:'Click B, then A again. The words change under the bars, and the tallest bar moves to the word that tells the two banks apart.',
 transition:'Next we write the whole layer, for every token at once, in the lecture\'s code.',source:SRC12});

// ---------------------------------------------------------------------------------------- 4.3 the equation
chap('4.3','Model: one attention layer');
const R7=A_.receiver+1,[raw0]=SC.w_raw,[mant,expo]=raw0.toExponential(2).split('e');
const nM=MK.scores.length,Mtex=Array.from({length:nM},(_,i)=>Array.from({length:nM},(_,j)=>j>i?R`-\infty`:'0').join('&')).join(R`\\`);
const mrow=MK.scores.map((_,j)=>MK.masked.includes(j)?R`-\infty`:'0').join(R`,\,`);
const AX4=MD.axes,VEC=v=>v.map(x=>num(x)),ROWS=[[m(R`e_${R7}`),...VEC(A_.e)],[m(R`\Delta e_${R7}=m_${R7}W_O`),...VEC(A_.delta)],[m(R`e_${R7}'`),...VEC(A_.e_new)]];
const iWater=0;                                                           // the first axis of the hand-built rows is "water"
const pad=s=>s.padEnd(23);
const panels=[
 stack(eq(R`S=QK^{\top},\qquad S_{ij}=q_i\cdot k_j`),eq(R`Q=EW_Q,\quad K=EW_K,\quad V=EW_V`),small('One row of S per token, one column per token it might read.')),
 stack(eq(R`\operatorname{softmax}([-8,\;8])=[${mant}\times10^{${+expo}},\;${tn(SC.w_raw[1])}]`),
  eq(R`\operatorname{softmax}\big([-8,\;8]/\sqrt{${SC.d_k}}\big)=[${tn(SC.w_scaled[0])},\;${tn(SC.w_scaled[1])}]`),
  small(`With d<sub>k</sub> = ${SC.d_k}, raw scores spread about ±${SC.sqrt}. Divided, they spread about ±1.`)),
 stack(eq(R`M=\begin{bmatrix}${Mtex}\end{bmatrix}`),eq(R`\begin{aligned}&[${MK.scores.join(R`,\,`)}]+[${mrow}]\\&\xrightarrow{\ \operatorname{softmax}\ }[${MK.weights.map(v=>v?tn(v):'0').join(R`,\;`)}]\end{aligned}`)),
 stack(eq(R`m_i=\sum_{j\le i}A_{ij}\,v_j`),sm(eq(R`m_${R7}=${tn(A_.first[1])}\,v_{\text{${A_.first[0]}}}+${tn(A_.second[1])}\,v_{\text{${A_.second[0]}}}+\dots=[${A_.m.map(v=>tn(v)).join(R`,\;`)}]`)),
  small(`Row ${R7} of A holds the bars of the last slide.`)),
 `<div class="p4-upd">${table(['',...AX4],ROWS.map(r=>r.map((c,j)=>j===iWater+1?`<span class="hi">${c}</span>`:c)))}</div><div style="margin-top:18px">${eq(R`E'=E+\operatorname{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}}+M\Big)\,V\,W_O`)}</div>`];
S_('attn-eq','attn','One attention layer, a piece per click',{builds:5,from:[14,'#s16-frame3-routing'],
 body:cols(`<div class="code-sm">${codeSteps([[1,pad('S = Q @ K.T')+'# scores'],[2,pad('S = S / d_k ** 0.5')+'# scale'],[3,pad('S = S + mask')+'# hide the future'],
  [4,pad('A = S.softmax(dim=-1)')+'# weights'],[4,pad('H = A @ V')+'# mix the values'],[5,pad('E_new = E + H @ W_O')+'# project, add']])}</div>`,
  panels.map((x,i)=>only([i+1],x)).join(''),'1fr 1.05fr'),
 lines:['Stack the token rows into E. One product scores every query against every key.',
  `Divide by √d<sub>k</sub>. Unscaled, the weaker key gets ${mant} × 10<sup>${+expo}</sup>; scaled, it gets ${num(SC.w_scaled[0])}. The order is kept.`,
  `The mask adds −∞ wherever j > i, so those weights come out exactly 0 and the rest still sum to 1.`,
  `Softmax turns each row into weights that sum to 1. They mix the values into a message m<sub>i</sub> of ${word(MD.d_v)} numbers.`,
  `W<sub>O</sub> turns the message into an update of width ${MD.d_model}, and the residual adds it. Bank's ${AX4[iWater]} coordinate goes from ${num(A_.e[iWater])} to ${num(A_.e_new[iWater])}.`],
 ask:['What is the shape of S for a sentence of T tokens?','Does the scaling change which key wins?','Why add −∞ before the softmax, not zero the weights after it?',
  'What does each row of A add up to?','Why add the update to e instead of replacing e?'],
 expected:['T by T: one score for every pair of positions.',`No. It only stops the softmax from saturating: [${SC.w_scaled.map(v=>num(v)).join(', ')}] instead of a near one-hot.`,
  'e to the −∞ is exactly 0 and the softmax still normalises. Zeroing afterwards leaves rows that do not sum to 1.',
  'One. Each message is a weighted average of the values.',
  `The residual keeps what the token already was and adds what it learned from the others. In A, bank keeps its row and gains ${num(A_.delta[iWater])} on the ${AX4[iWater]} axis.`],
 point:'One attention layer is two lines: weights from queries and keys, scaled and masked, then a mix of values added back to the rows.',
 misconception:'That the √d_k scaling fixes overflow. Overflow is handled by subtracting the maximum inside the softmax; the scaling keeps the softmax from saturating.',
 transition:'Does this layer predict better than the window of Lecture 11? Lecture 14 trained both.',
 source:`The lecture writes the same steps as two lines: A = (Q @ K.T / d_k**0.5 + mask).softmax(dim=-1) and E_new = E + (A @ V) @ W_O. Numbers: scripts/sci/p4.py.`});

// ---------------------------------------------------------------------------------------- 4.4 perplexity
chap('4.4','At test time: perplexity');
const MODEL_COL=[COL.amber,COL.data,COL.guess];                            // one colour per model, fixed
const PPW=1152,PPH=318,NX=300,PX=540,X0=600,X1=1060,PMAX=Math.ceil(Math.max(...TS.models.map(t=>t.ppl))/20)*20,RH=66,RY=70;
const px=v=>X0+v/PMAX*(X1-X0);
const pplFig=fig(PPW,PPH,
 label(NX,30,'model',{size:22,anchor:'end',weight:650,fill:COL.ink})+label(PX,30,'parameters',{size:22,anchor:'end',weight:650,fill:COL.ink})+
 label(X0,30,'perplexity on held-out stories (lower is better)',{size:22,weight:650,fill:COL.ink})+line(20,44,PPW-20,44,{stroke:AX})+
 TS.models.map((t,i)=>{const y=RY+i*RH;
  return label(NX,y+30,esc(t.name),{size:24,anchor:'end',fill:COL.ink})+label(PX,y+30,big(t.params),{size:24,anchor:'end',fill:MUTED})+
   gat(2,rect([X0,y+6,px(t.ppl),y+44],{stroke:'none',fill:MODEL_COL[i],rx:4})+label(px(t.ppl)+12,y+33,num(t.ppl),{size:24,fill:COL.ink,weight:650}));}).join('')+
 line(X0,RY+TS.models.length*RH,X1,RY+TS.models.length*RH,{stroke:AX})+
 Array.from({length:PMAX/20+1},(_,j)=>j*20).map(v=>line(px(v),RY+TS.models.length*RH,px(v),RY+TS.models.length*RH+7,{stroke:AX})+
  label(px(v),RY+TS.models.length*RH+32,v,{size:21,anchor:'middle'})).join(''),
 `Held-out perplexity on TinyStories: ${TS.models.map(t=>`${t.name} ${num(t.ppl)}`).join(', ')}`);
const [MLP,H1,H4]=TS.models;
S_('attn-ppl','attn','Perplexity on stories the models never saw',{builds:2,from:fromPage(14,'part3','#s05-cap-scores'),
 body:stack(eq(R`\text{perplexity}=\exp\big(\text{mean of }\lcol{-\ln p(\text{true next token})}\big)`),pplFig),
 lines:[`If every true next token got probability ${num(TS.quarter.p)}, the perplexity would be ${num(TS.quarter.ppl,0)}: as unsure as a choice among ${word(TS.quarter.ppl)}. Which model scores lowest?`,
  `Four heads: ${num(H4.ppl)}, with ${big(H4.params)} parameters against the MLP's ${big(MLP.params)}. Attention reads further with fewer weights.`],
 ask:['The MLP has the most parameters. Should it win?',`Why do one head and four heads have the same parameter count?`],
 expected:[`No. Its window concatenates the last tokens, so its first layer grows with the window, and it still sees less context. Attention shares its weights across positions.`,
  `Four heads of width ${TS.head_width} split the same ${TS.d_model} columns of W_Q, W_K and W_V that one head of width ${TS.d_model} uses. Each head gets its own softmax, so they can read different words: perplexity falls by ${num(TS.gain_four_vs_one,1)}%.`],
 point:'Perplexity is the exponential of the held-out loss; attention beat the fixed window with fewer parameters.',
 misconception:'That more parameters always mean a better language model. What the parameters can see matters more here.',
 transition:'The same block, wired three ways, gives the three families of Lecture 15.',
 source:`TinyStories, ${TS.vocab} tokens, context ${TS.context}, mean of ${word(TS.seeds)} seeds; recorded from Lecture 14 (part3.html, capstone-evidence.json), not retrained.`});

// ---------------------------------------------------------------------------------------- 4.5 three families
chap('4.5','Model: three families');
/** Who may attend to whom: a filled cell means row i may read column j. */
function maskGrid(rows,nCols,allowed,alt){
 const cell=34,fs=21,left=Math.max(...rows.map(r=>[...r].length))*fs*.6+16,W=left+nCols*cell+2,H=rows.length*cell+2;
 let g='';
 rows.forEach((r,i)=>{g+=label(left-10,i*cell+cell/2+fs*.36+1,esc(r),{size:fs,anchor:'end',fill:COL.ink});
  for(let j=0;j<nCols;j++)g+=`<rect x="${left+j*cell+2}" y="${i*cell+2}" width="${cell-4}" height="${cell-4}" rx="4" fill="${allowed(i,j)?COL.guess:'#fff'}" fill-opacity="${allowed(i,j)?.8:1}" stroke="${allowed(i,j)?'none':AX}"/>`;});
 return fig(W,H,g,alt);
}
const SRCW=FM.source,TGTW=FM.target,nS=SRCW.length;
const FAM=[
 {t:'Decoder-only',g:maskGrid(SRCW,nS,(i,j)=>j<=i,'Causal mask: each word reads itself and the words before it'),c:'each word reads the ones before it',
  pr:'the next word',use:'completing text (GPT)'},
 {t:'Encoder',g:maskGrid(SRCW,nS,()=>true,'Full mask: every word reads every word'),c:'every word reads every word',
  pr:`a label per word (${SRCW[nS-1]}: place)`,use:'tagging names (BERT)'},
 {t:'Encoder-decoder',g:maskGrid(TGTW,nS,()=>true,'Cross-attention: each Hindi position reads every English word'),c:'each Hindi word reads every English word',
  pr:'the next Hindi word',use:'translation'}];
S_('families','attn','One block, three families',{builds:3,from:[15,'#model-families'],
 body:`<div class="three p4-fam">${FAM.map(f=>`<div><h3>${f.t}</h3>${f.g}${small(f.c)}${at(2,ps(`<b>Predicts:</b> ${f.pr}`))}${at(3,ps(`<b>Use:</b> ${f.use}`))}</div>`).join('')}</div>`,
 lines:[`One sentence, “${SRCW.join(' ')}”. A filled cell lets that row read that column. Which family can write text one word at a time?`,
  'The training target follows from the mask: next words where the future is hidden, labels where every word is seen.',
  'The mask and the readout choose the family; the attention block is the same in all three.'],
 ask:['Which of the three can generate text, and which cannot?','Why can an encoder not be trained to predict the next word?','Which would you pick for translating English to Hindi, and why not decoder-only?'],
 expected:['Decoder-only and encoder-decoder generate; the encoder sees the whole input, so it labels instead.',
  'Every position already reads the next word, so predicting it would be copying (this is the mask question of the next slide).',
  'Encoder-decoder: the English is read in full by the encoder, and each Hindi step looks it up again by cross-attention. A decoder-only model can also translate if the English is put first in the prompt.'],
 point:'Decoder-only, encoder and encoder-decoder use the same attention block. They differ in who may attend to whom and in what the loss reads.',
 transition:'One question on the mask before we close the part.',
 source:'The sentence and its Hindi translation are Lecture 15’s running example (data/p4.json).'});

// ---------------------------------------------------------------------------------------- check
chap();
const LW=1152,LH=160,leakItems=LK.words.map((w,i)=>({name:w,v:LK.weights[i],col:i>LK.row?COL.loss:COL.guess,
 nameCol:i===LK.row?COL.guess:i===LK.row+1?COL.target:MUTED,nameWeight:i===LK.row||i===LK.row+1?700:''}));
const leakFig=fig(LW,LH+34,label(0,24,`Row ${LK.row+1}, “${LK.words[LK.row]}” (target: ${LK.target}), no mask. Red: words it should not see.`,{size:21,fill:COL.ink,weight:600})+
 vbars(leakItems,{oy:34,W:LW,H:LH,max:Math.ceil(Math.max(...LK.weights)*10)/10,gap:.45}),
 `Attention weights of row ${LK.row+1} of sentence A with the mask off; ${num(LK.future)} of the weight falls on later words`);
S_('check-4','attn','Check: the forgotten mask',{builds:2,
 body:cols(p('You forget the mask. The training loss falls to almost zero. Why is that bad news?'),
  code('S = Q @ K.T / d_k ** 0.5\n# S = S + mask    (forgotten)\nA = S.softmax(dim=-1)'),'1fr 1fr')+
  at(2,leakFig)+answer(2,`Every position can read the token it is asked to predict, so it learns to copy. At test time that token does not exist yet.`),
 lines:['Two ideas meet here: what the target of each position is, and what the mask hides.',
  `Row ${LK.row+1} puts ${num(LK.future)} of its weight on later words: ${num(LK.on_target)} on ${LK.target}, its own answer.`],
 ask:['What is the target at position i, and which positions can it read without the mask?','What would the loss on held-out text look like at test time?'],
 expected:['The target is token i + 1; without the mask, position i reads it directly.',
  'Generation feeds one token at a time, so the later tokens are not there. The model was never trained to predict without them, and its text falls apart even though the training loss looked perfect.'],
 point:'The causal mask makes training match test time: each position sees only what it will have when it generates.',
 misconception:'That a near-zero training loss means a good model. Here it means the task leaked into the input.',
 transition:'Here is the whole part on one card.',source:SRC12});

// ---------------------------------------------------------------------------------------- recap
recap(4,['Data: text as tokens; each target is the next token.',
 'Model: each token mixes the values of the tokens before it.',
 'At test time: sample, append, repeat.'],
 'the same block, fed image patches.',
 {card:RECIPES.gpt,title:'Transformer language model',
  ask:'Which rows of the card are the same as in the MLP classifier of the opening?',
  expected:'Loss and Update: still −ln of the probability of the right answer, still a gradient step. Data, Model, Output and At test time changed.',
  point:'A language model is the classifier of the opening, run at every position, with attention to gather the context.',
  transition:'Part 5: what if the tokens are not words?'});
