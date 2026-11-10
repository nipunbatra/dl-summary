// Part 5: vision and language (Lectures 16 to 22). The ViT cuts Lecture 16's photo into patch tokens; CLIP trains two
// encoders to agree on a batch of photos and captions; a VLM puts the image rows in front of a language model; a
// concept bottleneck model routes the class through concepts a person can correct.
// Every number comes from data/p5.json (scripts/sci/p5.py).
import {S_,O,partBreak,recap,chap,p,ps,cols,stack,sm,eq,m,R,at,only,answer,codeSteps,table,data,need,RECIPES,MODELS,
 word,Word,css,kt,kp,kl,tn,num,big,esc,fig,photo,label,line,path,dot,rect,pill,arrow,panel,chart,grid,gat,gonly,
 widget,register,slider,COL,MUTED,AX} from '../kit.js';

const D=need(await data('p5'),'vit','clip','vlm','cbm','cub'),V=D.vit,C=D.clip,L=D.vlm,B=D.cbm,U=D.cub;
const PHOTO='newfoundland_31 from Oxford-IIIT Pet (Parkhi, Vedaldi, Zisserman and Jawahar), CC BY-SA 4.0';
const SRC=s=>`${s} Numbers: scripts/sci/p5.py.`;
const four=['vit','clip','vlm','cbm'].map(id=>MODELS.find(M=>M.id===id));
const TRAIN={vit:'photos with a class label',clip:'photos with their captions',vlm:'a photo, a question and its answer',cbm:'photos with concept labels and a class'};
const pct=(v,n=2)=>`${num(100*v,n)} %`;
const P16=V.sizes[String(V.patch)],[PW,PH]=V.photo;

css(`.p5-side{display:flex;flex-direction:column;gap:14px}
.p5-side .code{margin:0}
.p5-top{display:grid;grid-template-columns:auto 1fr auto;gap:4px 16px;font-size:22px;line-height:1.3;align-items:baseline;width:max-content}
.p5-top b{font-weight:650;color:var(--q)}.p5-top span{font-variant-numeric:tabular-nums;text-align:right}
.p5-loss table{font-size:22px}.p5-loss th,.p5-loss td{padding:6px 12px;text-align:center}.p5-loss td:first-child,.p5-loss th:first-child{text-align:left}
.p5-tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.p5-tiles .tile{font-size:25px;padding:14px 16px}.p5-tiles .tile b{font-size:18px;margin-top:4px}
.p5-tiles .p5-on{font-size:21px;margin-top:12px;padding-top:10px;border-top:1.5px solid var(--line);color:var(--ink)}
.p5-sq .photo-fig svg{width:340px;max-width:100%;height:auto}`);

// ---------------------------------------------------------------------------------------- the part opens
partBreak(5,'Images as tokens, labels as language',
 'Part 4’s transformer read text. These lectures gave it images, then tied the images to words and to concepts.',
 ['cut the image into patches and reuse the transformer','train an image encoder and a text encoder to agree',
  'hand the image tokens to a language model','make the model reason through concepts a person can check'],
 {line:`Aim: for each of the ${word(four.length)} models, say what it is trained on, what it outputs and what it does at test time.`,
  ask:'Part 4’s model read text tokens. What would you have to change to make it read a photo?',
  expected:'Only the input: some way to turn the photo into a sequence of vectors. The attention blocks do not care where the rows came from.',
  transition:'Start with the photo every one of these lectures used.'});

// ---------------------------------------------------------------------------------------- 5.1 the ViT
chap('5.1','Data and Model: patches as tokens');
const CROP=V.crop,FOLLOW=V.follow_px;
/** The photo cropped to the model's 224 × 224 input, with the patch grid of size P drawn on it. */
function patchGrid(P){
 const z=V.sizes[P],n=z.side,st=CROP[2]/n,[x0,y0,s]=CROP,w=n>20?1:1.6;
 let g='';
 for(let i=1;i<n;i++)g+=line(x0+i*st,y0,x0+i*st,y0+s,{stroke:'#ffffffd0',w})+line(x0,y0+i*st,x0+s,y0+i*st,{stroke:'#ffffffd0',w});
 const r0=Math.floor(FOLLOW[1]/+P),c0=Math.floor(FOLLOW[0]/+P);
 return g+rect([x0+c0*st,y0+r0*st,x0+(c0+1)*st,y0+(r0+1)*st],{stroke:COL.guess,w:3.5});
}
const patchAlt=`Lecture 16's photo, cut into a grid of square patches; one patch is outlined`;
const readout=P=>{const z=V.sizes[P];return `${z.side} × ${z.side} = ${big(z.tokens)} tokens · with [CLS], ${big(z.with_cls)}² = ${big(z.scores)} attention scores per head`;};
register('p5-patch',(el,ctx)=>{
 const P0=String(V.patch),frame=inner=>`<div class="w-fig p5-sq">${photo('p5-newfoundland',PW,PH,`<g class="live">${inner}</g>`,patchAlt,{view:[CROP[0],CROP[1],CROP[2],CROP[2]]})}</div>`;
 if(ctx.printing){el.innerHTML=frame(patchGrid(P0))+`<div class="w-controls"><span class="w-readout">patch ${P0}: ${readout(P0)}</span></div>`;return;}
 el.innerHTML=frame('')+`<div class="w-controls"><span>patch size</span>${V.patches.map(q=>`<button type="button" data-p="${q}" aria-pressed="${String(q)===P0}">${q} × ${q}</button>`).join('')}</div><div class="w-controls"><span class="w-readout"></span></div>`;
 const live=el.querySelector('.live'),out=el.querySelector('.w-readout');let P=P0;
 const show=()=>{live.innerHTML=patchGrid(P);out.textContent=readout(P);};
 for(const b of el.querySelectorAll('button[data-p]'))b.addEventListener('click',()=>{P=b.dataset.p;
  for(const o of el.querySelectorAll('button[data-p]'))o.setAttribute('aria-pressed',o===b);b.blur();show();});
 show();
});
const sh=(...a)=>`# B, ${a.map(big).join(', ')}`;
S_('vit','vision','A photo becomes a sequence of patch tokens',{builds:4,from:[16,'#story-patch-grid'],
 body:cols(widget('p5-patch',patchAlt),`<div class="p5-side"><div class="code-xs">${codeSteps([
  [0,`p = F.unfold(img, ${V.patch}, stride=${V.patch}).mT  ${sh(P16.tokens,P16.numbers)}`],
  [2,`x = embed(p)                         ${sh(P16.tokens,V.d)}`],
  [3,`x = prepend_cls(x) + pos             ${sh(P16.with_cls,V.d)}`],
  [3,`for block in blocks:                 # ${V.blocks} blocks`],
  [3,`    x = block(x)                     ${sh(P16.with_cls,V.d)}`],
  [4,`logits = head(norm(x)[:, 0])         ${sh(V.classes)}`]])}</div>
  ${at(4,`<div class="p5-top">${V.top3.map(([k,v],i)=>`${i?`<span></span>`:`<b>top class</b>`}<span${i?'':' class="kp"'}>${k}</span><span${i?'':' class="kp"'}>${pct(v)}</span>`).join('')}</div>`)}</div>`,'0.72fr 1.28fr'),
 lines:[`Lecture 16’s photo at ${V.img} × ${V.img}, cut into square patches; each patch is one token. Switch the patch size: what happens to the tokens and the attention cost?`,
  `At patch size ${V.patch}, each patch is ${V.patch} × ${V.patch} × ${V.channels} = ${big(P16.numbers)} numbers. One shared linear map turns every patch into ${V.d} features.`,
  `A learned [CLS] row and a position vector per row give ${P16.with_cls} × ${V.d}. Then ${word(V.blocks)} attention blocks with no mask, the encoder of Part 4.`,
  `The final [CLS] row goes through a linear head with ${big(V.classes)} outputs, then a softmax. The right breed comes first, far ahead of the next two.`],
 ask:['Why not one token per pixel?','Does the linear map know where its patch came from?','Why no causal mask here?','Where is the cross-entropy?'],
 expected:[`${big(V.pixels)} tokens, and attention compares every pair: the cost grows with the square of the count. Patches of ${V.patch} give ${P16.tokens}.`,
  'No. It is shared by every patch, so the same crop anywhere gives the same vector. That is why a position vector is added.',
  'The whole photo is there at once: no patch is a “future” one to hide. This is the encoder branch of Part 4, not the decoder.',
  `On the ${big(V.classes)} logits, against the photo’s class: the same −ln p of the right class as in Part 1. Only the input changed.`],
 point:'A ViT is Part 4’s encoder with a new Data row: patches instead of word tokens, and a [CLS] row read out for the class.',
 misconception:'That the ViT needs a new kind of layer for images. Apart from the patch projection, every layer is the text encoder’s.',
 demo:`Press ${V.patches[0]}, ${V.patches[1]} and ${V.patches[2]}: the tokens go ${V.patches.map(q=>V.sizes[q].tokens).join(', ')}, and the scores per head grow by the square. The outlined patch is the one Lecture 16 followed (P63). Lecture 22 rebuilt this ViT live in PyTorch with AI tools, on MNIST; its recording is on the schedule.`,
 transition:`Its ${big(V.classes)} classes are a fixed list. CLIP lets words supply the classes.`,
 source:SRC(`Photo: ${PHOTO}, as the model sees it. ViT-Tiny (vit_tiny_patch16_224); the prediction is Lecture 16’s saved run.`)});

// ---------------------------------------------------------------------------------------- 5.2 CLIP's loss
chap('5.2','Loss: match photos to captions');
const CM=C.C,N=C.n,[TMIN,TMAX,TSTEP]=C.tau_range;
const softmax=a=>{const mx=Math.max(...a),e=a.map(v=>Math.exp(v-mx)),s=e.reduce((x,y)=>x+y,0);return e.map(v=>v/s);};
/** CLIP's batch at temperature t: the row softmax P, the column softmax Q, and the loss averaged over both directions. */
function clipAt(t){
 const S=CM.map(r=>r.map(c=>c/t)),P=S.map(softmax),Q=S[0].map((_,j)=>softmax(S.map(r=>r[j])));   // Q[j]: caption j over the photos
 const li=-P.reduce((a,r,i)=>a+Math.log(r[i]),0)/N,lt=-Q.reduce((a,c,j)=>a+Math.log(c[j]),0)/N;
 return {P,li,lt,loss:(li+lt)/2};
}
if(Math.abs(clipAt(C.tau).loss-C.loss)>1e-5)throw new Error('p5: the CLIP loss in JS does not match p5.json');
const TW=1152,TH=372,GY=58,CW=100,CH=60,FS=22;
const photos=C.names.map(n=>`${n} photo`),capts=C.names.map(n=>`“${n}”`);
const gC=grid(CM,{rows:photos,cols:capts,cw:CW,ch:CH,fs:FS,color:COL.data,max:1,dec:2,bare:true,oy:GY,mark:(i,j)=>i===j});
const PX=gC.W+150,LX=PX+N*CW+70,rowY=i=>GY+gC.H-N*CH+i*CH+CH/2+FS*.36;
const tauStatic=label(0,24,'cosine of each photo with each caption',{size:FS,fill:COL.ink,weight:650})+gC.g+
 gat(2,arrow(gC.W+16,GY+gC.H/2+18,PX-14,GY+gC.H/2+18,{stroke:COL.ink,w:2.5})+label((gC.W+PX)/2,GY+gC.H/2-14,'÷ τ, then',{size:20,anchor:'middle',fill:COL.ink})+
  label((gC.W+PX)/2,GY+gC.H/2+50,'softmax',{size:20,anchor:'middle',fill:COL.ink})+label((gC.W+PX)/2,GY+gC.H/2+74,'along a row',{size:20,anchor:'middle',fill:COL.ink})+
  label(PX,24,'P: each photo picks a caption',{size:FS,fill:COL.guess,weight:650})+`<g class="live-p"></g>`)+
 gat(3,label(LX,24,'−ln P, paired cell',{size:FS,fill:COL.loss,weight:650})+`<g class="live-l"></g>`+
  label(0,TH-8,`a model that cannot tell them apart gives each caption 1/${N}: ln ${N} = ${num(C.ln_n)}`,{size:21,fill:COL.ink}))+
 gat(4,pill(TW,TH,`after ${C.recorded.steps} steps at τ = ${C.tau}: ${num(C.recorded.after_steps)}`,{fill:COL.loss,size:21,anchor:'end',maxX:TW}));
function tauLive(t){
 const r=clipAt(t),gP=grid(r.P,{cols:capts,cw:CW,ch:CH,fs:FS,color:COL.guess,max:1,dec:2,bare:true,ox:PX,oy:GY,L:0,mark:(i,j)=>i===j});
 const pen=r.P.map((row,i)=>label(LX+40,rowY(i),num(-Math.log(row[i])),{size:FS,anchor:'middle',fill:COL.loss,weight:650})).join('');
 return [gP.g,pen+label(0,GY+gC.H+38,`loss ${num(r.loss)}`,{size:25,fill:COL.loss,weight:700})+
  label(118,GY+gC.H+38,`the mean of the rows (${num(r.li)}) and of the columns (${num(r.lt)})`,{size:20,fill:COL.ink})];
}
const tauAlt=`The ${N} by ${N} cosine matrix of Lecture 17's batch, its row softmax at a temperature, and the loss`;
register('p5-tau',(el,ctx)=>{
 const frame=`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${TW} ${TH}" role="img" aria-label="${esc(tauAlt)}">${tauStatic}</svg></div>`;
 el.innerHTML=frame+(ctx.printing?'':`<div class="w-controls" data-at="2">${slider('tau','temperature τ',TMIN,TMAX,TSTEP,C.tau)}</div>`);
 const lp=el.querySelector('.live-p'),ll=el.querySelector('.live-l');
 const show=t=>{const [a,b]=tauLive(t);lp.innerHTML=a;ll.innerHTML=b;};
 if(ctx.printing){show(C.tau);return;}
 const range=el.querySelector('[data-k="tau"]'),out=el.querySelector('[data-o="tau"]');
 const upd=()=>{out.textContent=num(+range.value);show(+range.value);};
 range.addEventListener('input',upd);range.addEventListener('change',()=>range.blur());upd();
});
const tKeys=Object.keys(C.sweep);
S_('clip','vision','CLIP: a softmax along each row of the batch',{builds:4,from:[17,'#cosine-matrix'],
 body:widget('p5-tau',tauAlt),
 lines:[`Lecture 17’s batch: ${word(N)} photos and their ${word(N)} captions, each encoded to a unit vector. Which cells of this matrix should training push up?`,
  `Divide by the temperature τ = ${C.tau}, then a softmax along each row: each photo picks a caption. Drag τ and watch the rows sharpen or flatten.`,
  `The loss is −ln of the paired cell, averaged over the rows and again down the columns, where each caption picks a photo.`,
  `Training moves both encoders and the temperature so that the diagonal wins. On this one batch the loss keeps falling.`],
 ask:['Which cells are the observed pairs?','What does a small τ do to a row?','Where have you seen −ln of the right entry before?','Where are the class labels?'],
 expected:['The diagonal: photo i was posted with caption i. Every other cell in its row is a competitor.',
  `It sharpens it: at τ = ${tKeys[0]} the loss is ${num(C.sweep[tKeys[0]])}, at τ = ${tKeys.at(-1)} it is ${num(C.sweep[tKeys.at(-1)])}. The order of the captions never changes.`,
  `Everywhere since Part 1: it is the cross-entropy of a ${N}-way classifier whose classes are the captions in the batch.`,
  'There are none. The pairing of each photo with its own caption is the only supervision.'],
 point:'CLIP’s loss is ordinary cross-entropy, where the classes are the captions in the batch, done once per row and once per column.',
 misconception:'That the softmax is taken over all captions in the dataset. It runs over the captions in the batch only, which is why CLIP needs large batches.',
 demo:`Start at τ = ${C.tau} (loss ${num(C.loss)}). Drag to ${tKeys[0]}: the rows are nearly one-hot and the loss falls to ${num(C.sweep[tKeys[0]])}. Drag to ${tKeys.at(-1)}: the rows flatten towards 1/${N} and the loss rises towards ln ${N}.`,
 transition:'After training, the same two encoders classify photos with no new training at all.',
 source:SRC(`The authored ${C.dim}-D batch of Lecture 17 (cat, dog, car). The loss after ${C.recorded.steps} steps is the lecture’s run.`)});

// ---------------------------------------------------------------------------------------- 5.3 CLIP at test time
chap('5.3','At test time: classes written as captions');
const MW=600,MH=300,BX=0,BW=470,BMAX=Math.max(...C.menu.map(x=>x[1]),...C.wrong_menu.map(x=>x[1]));
function menu(items,b){
 const best=items.reduce((a,x)=>x[1]>a[1]?x:a);
 return items.map(([t,v],i)=>{const y=66+i*90,hit=t===best[0];
  return label(BX,y,`“${esc(t)}”`,{size:24,fill:COL.ink})+
   gat(b,`<rect x="${BX}" y="${y+14}" width="${(v/BMAX*BW).toFixed(1)}" height="26" rx="4" fill="${hit?COL.guess:AX}"/>`+
    label(BX+v/BMAX*BW+10,y+35,num(v,4),{size:22,fill:hit?COL.guess:COL.ink,weight:hit?700:''}));}).join('');
}
const menuFig=fig(MW,MH,label(0,24,'cosine of the photo with each caption',{size:21})+gonly('1-2',menu(C.menu,2))+gonly(3,menu(C.wrong_menu,3)),
 'The captions written for the classes, and their cosine with the photo');
const [W1,W2]=[...C.wrong_menu].sort((a,b)=>b[1]-a[1]).map(([t,v])=>[`“${t}”`,v]);
S_('clip-use','vision','At test time, each class is written as a caption',{builds:3,from:[17,'#mot-wrong-menu-result'],
 body:cols(photo('p5-newfoundland',PW,PH,'','Lecture 16’s Newfoundland photo'),menuFig,'1fr 1fr'),
 lines:['The ViT could only answer with its stored classes. With CLIP we write each class as a caption, encode it, and compare. Which caption wins?',
  `The photo goes to the caption with the largest cosine. All ${word(C.menu.length)} scores come from one image vector; the captions are encoded once.`,
  `Now leave “dog” out of the list. CLIP still picks one of the captions we wrote, ${W1[0]}, ahead of ${W2[0]} by ${num(W1[1]-W2[1],4)}.`],
 ask:['How would you add a new class to the ViT? And to CLIP?','Why “a photo of a …” rather than the bare word?','Is the winner a confident answer?'],
 expected:['The ViT needs a new output row and training on labelled photos. CLIP needs only a new caption: no training.',
  'CLIP was trained on captions, so a caption-like sentence lands closer to where the photos are. The bare word works, a little worse.',
  'No. The softmax always spreads 1 over the captions we wrote, even when none is right. The cosines here barely differ.'],
 point:'At test time CLIP needs no class rows: each class is a caption, and the prediction is the closest caption.',
 misconception:'That CLIP generates the label. It only ranks the captions we supply; it writes no text.',
 transition:'To answer in words, the image has to reach a model that writes text.',
 source:SRC(`Photo: ${PHOTO}. Cosines: frozen CLIP ViT-B/32, Lecture 17’s saved measurements.`)});

// ---------------------------------------------------------------------------------------- 5.4 the VLM
chap('5.4','Model: an image in front of a language model');
const vlmChart=chart([{steps:[
 {k:'Image',t:`${V.img} × ${V.img}\nphoto`},
 {k:'Vision encoder',t:`${L.rows} patch rows\n× ${L.d_v}`,n:'[CLS] dropped'},
 {k:'Projector',t:`${L.d_v} → ${L.d_lm}\nfor every row`},
 {k:'Language model',t:'image rows,\nthen the prompt',col:COL.data},
 {k:'Output',t:`p over ${big(L.vocab)}\ntokens`,col:COL.guess}]}],{W:1152,bh:104,gx:40,fs:21,alt:'A vision-language model: image, vision encoder, projector, language model, next token'});
const LW_=540,LH_=170,LX0=170,lx=v=>LX0+v/L.full*(LW_-LX0-200);
const lenBar=(y,img,name)=>{const h=36;
 return `<rect x="${LX0}" y="${y}" width="${(lx(img)-LX0).toFixed(1)}" height="${h}" fill="${COL.data}" rx="3"/>`+
  `<rect x="${lx(img).toFixed(1)}" y="${y}" width="${(lx(img+L.prompt)-lx(img)).toFixed(1)}" height="${h}" fill="${MUTED}" rx="3"/>`+
  label(LX0-12,y+h*.7,name,{size:20,anchor:'end',fill:COL.ink})+label(lx(img+L.prompt)+10,y+h*.7,`${img} + ${L.prompt} = ${img+L.prompt}`,{size:21,fill:COL.ink,weight:700});};
const lenFig=fig(LW_,LH_,label(0,22,'positions the language model reads',{size:20,fill:COL.ink,weight:650})+
 lenBar(40,L.rows,'all patch rows')+lenBar(92,L.slots,`${word(L.slots)} summaries`)+
 `<rect x="${LX0}" y="${LH_-22}" width="18" height="18" rx="3" fill="${COL.data}"/>`+label(LX0+26,LH_-6,'image',{size:19,fill:COL.data,weight:650})+
 `<rect x="${LX0+110}" y="${LH_-22}" width="18" height="18" rx="3" fill="${MUTED}"/>`+label(LX0+136,LH_-6,'prompt',{size:19,fill:MUTED,weight:650}),
 'Positions the language model reads: all patch rows and the prompt, against eight summaries and the prompt');
const lossTable=`<div class="p5-loss">${table([kt('target'),...L.answer.map(t=>kt(t==='end'?'end':`“${esc(t.trim())}”`))],
 [[kp('p'),...L.probs.map(v=>kp(num(v)))],[kl('−ln p'),...L.pen.map(v=>kl(num(v)))]])}</div>`;
S_('vlm','vision','A VLM puts the image in front of a language model',{builds:3,from:[18,'#summary-connections'],
 body:`${vlmChart}${cols(at(2,lenFig),at(3,`${lossTable}<p class="small" style="margin-top:8px">mean over the answer: ${kl(num(L.mean,3))}</p>`),'1fr 1fr')}`,
 lines:[`Lectures 18 and 19 kept the language model and gave it the image. For one photo and a ${L.prompt}-token prompt, how many positions does it read?`,
  `Every patch row becomes a position in front of the prompt. Lecture 19’s ${word(L.slots)} learned summary vectors read all ${L.rows} rows and shorten the prefix.`,
  `Training on the reference answer “A black dog.”: the loss is −ln p of each answer token, given the image and the tokens before it.`],
 ask:['What does the projector change, and what does it leave alone?','Are the summaries a choice of patches?','Why is the prompt not graded?'],
 expected:[`It changes the width, ${L.d_v} to ${L.d_lm}, so the rows fit the language model. It keeps their number: ${L.rows} stay ${L.rows}, and none becomes a word.`,
  'No. Each is a learned query that mixes all the rows by attention, so every summary depends on the whole image.',
  'It is given, not predicted. Only the answer tokens get labels; the prompt’s are set to −100 and skipped.'],
 point:`A VLM is Part 4’s language model with image rows in front; it is trained with the same next-token loss, on the answer.`,
 misconception:'That the model reads the image in words. The image rows are vectors at earlier positions; they never become token IDs.',
 transition:'All these models are hard to question. The last one is built so that a person can check it.',
 source:SRC(`DeiT-Tiny rows and DistilGPT2 in Lectures 18 and 19’s notebook; the answer probabilities are the lecture’s chosen numbers.`)});

// ---------------------------------------------------------------------------------------- 5.5 the concept bottleneck
chap('5.5','Output: concepts you can correct');
const KW=1152,KH=250,KF=21,ky=i=>66+i*64;
const CX=430,CB=200,QX=1000,QB=100;
const cbmStatic=`<rect x="0" y="${ky(0)-30}" width="150" height="${ky(2)-ky(0)+60}" rx="12" fill="#fff" stroke="${AX}" stroke-width="2"/>`+
 label(75,ky(1)-4,'photo',{size:KF,anchor:'middle',fill:COL.ink})+label(75,ky(1)+24,'x',{size:KF,anchor:'middle',fill:COL.ink})+
 arrow(162,ky(1)-6,232,ky(1)-6,{stroke:COL.ink,w:2.5})+label(197,ky(1)-18,'g',{size:KF,anchor:'middle',fill:COL.ink})+
 label(CX,22,'concepts ĉ',{size:KF,anchor:'end',fill:COL.ink,weight:650})+B.concepts.map((c,i)=>label(CX,ky(i)+7,esc(c),{size:KF,anchor:'end',fill:COL.ink})).join('')+
 arrow(700,ky(1)-6,822,ky(1)-6,{stroke:COL.ink,w:2.5})+label(761,ky(1)-18,'s = W ĉ',{size:KF,anchor:'middle',fill:COL.ink})+label(761,ky(1)+26,'softmax',{size:KF,anchor:'middle',fill:COL.ink})+
 label(QX-14,22,'class p',{size:KF,anchor:'end',fill:COL.ink,weight:650})+B.classes.map((c,i)=>label(QX-14,ky(i)+7,esc(c),{size:KF,anchor:'end',fill:COL.ink})).join('');
function cbmAt(fix){
 const c=fix?B.c_fix:B.c,s=B.W.map(r=>r.reduce((a,w,j)=>a+w*c[j],0)),q=softmax(s),top=q.indexOf(Math.max(...q));
 return {c,q,top};
}
for(const f of [0,1]){const q=cbmAt(f).q,ref=f?B.p_fix:B.p;if(q.some((v,i)=>Math.abs(v-ref[i])>1e-4))throw new Error('p5: the CBM head in JS does not match p5.json');}
function cbmLive(fix){
 const {c,q,top}=cbmAt(fix);
 return c.map((v,i)=>{const ed=fix&&i===B.edit;return `<rect x="${CX+14}" y="${ky(i)-14}" width="${(v*CB).toFixed(1)}" height="28" rx="4" fill="${ed?COL.amber:COL.guess}"/>`+
   label(CX+24+v*CB,ky(i)+8,num(v),{size:KF,fill:ed?COL.amber:COL.ink,weight:ed?700:''});}).join('')+
  (fix?label(CX+14,ky(B.edit)+34,'set by a person',{size:17,fill:COL.amber}):'')+
  q.map((v,i)=>`<rect x="${QX}" y="${ky(i)-14}" width="${Math.max(1.5,v*QB).toFixed(1)}" height="28" rx="4" fill="${COL.guess}"${i===top?'':' opacity=".4"'}/>`+
   label(QX+10+v*QB,ky(i)+8,num(v,2),{size:KF,fill:COL.ink,weight:i===top?700:''})).join('');
}
const cbmAlt='A concept bottleneck model: a photo, three predicted concepts, and the class probabilities read from the concepts alone';
register('p5-cbm',(el,ctx)=>{
 const frame=`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${KW} ${KH}" role="img" aria-label="${esc(cbmAlt)}">${cbmStatic}<g class="live"></g></svg></div>`;
 const live=()=>el.querySelector('.live');
 if(ctx.printing){el.innerHTML=frame;live().innerHTML=cbmLive(false);return;}
 const c0=B.concepts[B.edit];
 el.innerHTML=frame+`<div class="w-controls"><span>${esc(c0)}</span><button type="button" data-fix="0" aria-pressed="true">as predicted, ${num(B.c[B.edit])}</button><button type="button" data-fix="1" aria-pressed="false">corrected to ${num(B.c_fix[B.edit],0)}</button></div>`;
 let fix=false;const show=()=>{live().innerHTML=cbmLive(fix);};
 for(const b of el.querySelectorAll('button[data-fix]'))b.addEventListener('click',()=>{fix=b.dataset.fix==='1';
  for(const o of el.querySelectorAll('button[data-fix]'))o.setAttribute('aria-pressed',o===b);b.blur();show();});
 show();
});
const dc=B.c_fix[B.edit]-B.c[B.edit],col0=B.W.map(r=>r[B.edit]),vec=a=>`(${a.map(v=>Number.isInteger(v)?String(v):tn(v,1)).join(',')})`;
const ct=B.concept_term;
S_('cbm','vision','A concept bottleneck: the class is read from concepts alone',{builds:3,from:[20,'#s04.11'],
 body:`${widget('p5-cbm',cbmAlt)}${cols(at(2,sm(eq(R`\begin{aligned}\Delta s&=W\,\Delta\hat c=${tn(dc,1)}\times${vec(col0)}=${vec(B.delta)}\\ p_{\text{zebra}}&=\qcol{${tn(B.p[0],2)}}\ \to\ \qcol{${tn(B.p_fix[0],2)}}\end{aligned}`))),
  at(3,sm(eq(R`\begin{aligned}\lcol{\mathcal L}&=\lambda_c\,\lcol{\mathcal L_c}+\lambda_y\,\lcol{\mathcal L_y}\\ -\ln\qcol{${tn(ct.pred,1)}}&=\lcol{${tn(ct.loss,2)}}\ \ \text{(one concept, label }\pcol{${ct.label}}\text{)}\end{aligned}`))),'1.25fr 1fr')}`,
 lines:[`Lecture 20’s worksheet: ${word(B.concepts.length)} concepts, then one linear head W. The model says horse. A person sees stripes: correct it to ${num(B.c_fix[B.edit],0)}. What happens?`,
  `Only the stripes column of W moves the scores. Nothing is retrained: the same head, given the corrected concept, now says zebra.`,
  'Training needs a label for every concept as well as the class: a concept loss on ĉ plus the usual class cross-entropy.'],
 ask:['Which numbers does the class head see?','Would the same edit change a multi-task model’s answer?','What does each training scheme do with these two losses?'],
 expected:[`Only the ${word(B.concepts.length)} concept values. That is the bottleneck: the photo reaches the class only through them.`,
  'No. There the concept head is a side branch, and the class head never reads it. In a bottleneck the concepts are the only path.',
  'Independent fits g on the concepts and f on the true concepts; sequential fits f on g’s outputs; joint minimises both losses together.'],
 point:'A concept bottleneck model changes the Output row: it outputs concepts first, and a person can correct them at test time.',
 misconception:'That correcting a concept always helps. If the head has a wrong-signed weight, a correct edit can make the answer worse (Lecture 20, the billing example).',
 demo:'Press the two buttons: the stripes bar turns amber and the class bars swap from horse to zebra.',
 transition:'Does this hold on real photos, with real concept predictors?',
 source:SRC('The zebra, horse and giraffe worksheet of Lecture 20; weights from the cbm repository.')});

// CUB: accuracy before and after a person corrects concept groups
const CUBK=['independent','sequential','joint'],CUBC={independent:COL.data,sequential:COL.green,joint:COL.amber};
const GW=1152,GH=372,G=U.groups,GP=panel({W:GW,H:GH,L:70,R:300,T:36,B:58,xMin:0,xMax:G,yMin:.45,yMax:1,ticksX:[0,1,2,3,4].map(k=>k*G/4),
 ticksY:[.5,.6,.7,.8,.9,1],fmtY:v=>String(Math.round(100*v)),xLabel:`concept groups corrected by a person (of ${G})`,yLabel:'test accuracy (%)',fs:20});
const yStd=GP.y(U.acc.standard),endX=GP.x(G)+14;
const startLab={joint:-12,sequential:24,independent:8},endLab={independent:-10,sequential:20,joint:8};
const cubFig=fig(GW,GH,GP.axes+
 line(GP.x(0),yStd,GP.x(G),yStd,{stroke:MUTED,w:2.5,dash:'8 6'})+label(endX,yStd+7,`standard, no concepts: ${num(100*U.acc.standard,1)}`,{size:20,fill:MUTED})+
 gat(2,CUBK.map(k=>path(U.curves[k].map((v,i)=>[GP.x(i),GP.y(v)]),{stroke:CUBC[k],w:4})).join('')+
  CUBK.map(k=>label(endX,GP.y(U.curves[k][G])+endLab[k],`${k}: ${num(100*U.curves[k][G],1)}`,{size:21,fill:CUBC[k],weight:650})).join(''))+
 CUBK.map(k=>dot(GP.x(0),GP.y(U.acc[k]),8,{fill:CUBC[k]})).join('')+
 gonly(1,CUBK.map(k=>label(GP.x(0)+16,GP.y(U.acc[k])+startLab[k],`${k}: ${num(100*U.acc[k],1)}`,{size:21,fill:CUBC[k],weight:650,halo:true})).join('')),
 'Test accuracy on CUB of three concept bottleneck models as a person corrects more concept groups, against a standard model');
S_('p5-cub','vision','On real birds: accuracy, then accuracy after corrections',{builds:2,from:[20,'#s23.6'],
 body:cubFig,
 lines:[`Lecture 20’s run on CUB: ${U.species} bird species, ${U.concepts} concepts in ${U.groups} groups. Which model would you deploy if a bird expert will correct its concepts?`,
  'Correct one group of concepts after another. The least accurate model gains the most: its head was trained on true concepts, so it trusts corrected ones.'],
 ask:['Joint and standard are equally accurate. What does joint give that standard cannot?','Why does joint gain the least?'],
 expected:['A place to intervene: its concepts can be read and corrected. The standard model has no concepts to correct.',
  'Its head learned to read the predictor’s soft, drifted scores. True 0 and 1 values are not what it was trained on.'],
 point:'Accuracy alone does not rank concept bottleneck models; how much a correction helps depends on how the head was trained.',
 misconception:'That the most accurate model is always the best choice. With an expert in the loop, the independent model wins.',
 transition:'One question that needs every model of this part.',
 source:SRC(`CUB, frozen CLIP ViT-B/32 features, one run (Lecture 20, from the cbm repository).`)});

// ---------------------------------------------------------------------------------------- check and recap
S_('check-5','vision','Check: which of these need class labels?',{builds:2,nochap:true,
 body:`<div class="tiles p5-tiles">${four.map(M=>`<div class="tile">${M.name}<b>outputs ${M.out}</b>${at(2,`<div class="p5-on">trained on ${TRAIN[M.id]}</div>`)}</div>`).join('')}</div>
  ${answer(2,`The ViT and the concept bottleneck model, which needs concept labels too.`)}`,
 lines:[`Which of these ${word(four.length)} models need a class label on every training photo? Think about what the Data row of each held.`,
  'The Data row is where these models differ most; every loss is still −ln p.'],
 ask:['Which need class labels to train?','Which of them can add a new class without any training?'],
 expected:['ViT and the concept bottleneck model. CLIP learns from captions, the VLM from answers in words.',
  'CLIP: write the new class as a caption. A VLM can also be asked about it in words. The ViT and the bottleneck need new rows in their heads.'],
 point:'Replacing class labels by text is what lets CLIP and the VLM train on photos from the web.',
 transition:'Part 5 in one slide.'});

recap(5,[`Patches make ${P16.tokens} tokens for Part 4’s encoder.`,
 'CLIP’s classes are the captions in the batch.',
 'A VLM reads image rows; a bottleneck’s concepts can be corrected.'],
 'so far every model predicted a label or a token; next, models of the data itself.',
 {card:{...RECIPES.clip,model:'an image encoder and a text encoder'},title:'CLIP',sm:true,ask:'Which row of this card is different from the ViT’s card?',
  expected:'Data (captions instead of class labels), Output and Loss (the batch’s captions are the classes), and At test time (captions written for the classes).',
  point:'Part 5 changed every row except Update.',transition:'Part 6: models of the data itself.'});
