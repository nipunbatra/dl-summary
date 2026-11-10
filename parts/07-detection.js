// Part 7: object detection (Lecture 26). The output becomes a list of boxes; IoU compares two boxes; every grid cell has
// a target and the loss compares cell by cell; at test time a score threshold and NMS keep one box per object, and
// precision, recall and AP measure the list. Every number comes from data/p7.json (scripts/sci/p7.py).
import {S_,O,partBreak,recap,chap,p,ps,small,stack,cols,at,only,m,eq,sm,R,tn,answer,kt,kp,kl,table,code,data,need,css,word,Word,RECIPES,
 photo,rect,pill,label,line,dot,path,panel,fig,gat,widget,register,slider,num,big,COL,MUTED,AX,PAPER,esc} from '../kit.js';

const D=need(await data('p7'),'photo','classifier','iou','grid','det','check','pr');
const PH=D.photo,IU=D.iou,G=D.grid,DT=D.det,CK=D.check,PR=D.pr,FIN=DT.final;
const PHOTO='Photo: Lecture 26’s courtyard. The photographs are course-owned generated images (see SOURCES.md); every box and score drawn on them is computed.';
const YOLO='Boxes and scores: YOLOv8n, recorded from Lecture 26 (data/science.json).';
const SRC=(...a)=>[...a,'Numbers: scripts/sci/p7.py.'].join(' ');
const bt=v=>big(v).replaceAll(',','{,}');                                // a large count inside TeX

css(`.p7-iou .photo-fig svg,.p7-nms .photo-fig svg{max-height:330px}
.p7-read{display:flex;gap:6px 26px;flex-wrap:wrap;font-size:21px;line-height:1.3;margin:4px 0 0;font-variant-numeric:tabular-nums;color:var(--ink)}
.p7-read b{color:var(--q)}
.p7-kv{font-size:25px}.p7-kv dd{font-weight:650}
.p7-steps{font-size:24px}
.p7-rec th,.p7-rec td{text-align:center;padding:4px 7px}
.p7-code .code{font-size:16px;line-height:1.5}
.p7-q{font-size:25px;line-height:1.42}`);

/** Call f(build) whenever the slide holding el moves to another click, so a live figure can show what the click says. */
const onBuild=(el,f)=>{const sc=el.closest('.scene');if(sc)new MutationObserver(()=>f(+sc.dataset.build)).observe(sc,{attributes:true,attributeFilter:['data-build']});};
const courtyard=(inner,alt,o={})=>photo('p7-courtyard',PH.W,PH.H,inner,alt,o);
const short=n=>n==='potted plant'?'plant':n,colOf=n=>n==='potted plant'?COL.amber:COL.guess;
/** A detection drawn the way detectors draw it: its box and a tab with the class and the score. */
const detBox=(k,{w=6,size=30}={})=>rect(k.box,{stroke:colOf(k.name),w})+pill(k.box[0],k.box[1],`${short(k.name)} ${num(k.score)}`,{fill:colOf(k.name),size,maxX:PH.W});

// ---------------------------------------------------------------------------------------- the part opens
partBreak(7,'From one label to a list of objects',
 'Every model so far gave one answer per input. A detector returns a list of objects, as long as the photo needs.',
 ['The output is a list of boxes, each with a class and a score.','IoU says how well two boxes agree.',
  'Every grid cell predicts, and the loss compares cell by cell.','At test time, keep the confident boxes and drop the duplicates.'],
 {line:'By the end you can say how a network with a fixed-size output returns a list whose length changes from photo to photo.',
  ask:'A photo holds a dog and a cat. Which rows of the recipe must change to get both of them, with where they are?',
  expected:'The Output (a list of boxes), and so the Data (people draw boxes) and the Loss. At test time there is new work: choosing which boxes to keep.',
  transition:'Start from what a classifier says about one photo.'});

// ---------------------------------------------------------------------------------------- 7.1 a list of boxes
chap('7.1','Output: a list of boxes');
const nBox=FIN[0].box.length;
S_('det-list','det','From one label to a list of boxes',{builds:3,
 body:cols(courtyard(gat(3,FIN.map(k=>detBox(k)).join('')),'The courtyard photo: a dog, a cat and two potted plants; from click 3, the detector’s four boxes'),
  stack(at(2,p(`A classifier gives one label for the whole photo: ${kp(`${esc(D.classifier.label)}, ${num(D.classifier.p)}`)}.`)),
   at(3,`<div class="table-sm xtight">${table(['class','score','box (x₁, y₁, x₂, y₂)'],FIN.map(k=>[esc(k.name),num(k.score),`(${k.box.map(v=>num(v,0)).join(', ')})`]))}</div>`)),'1.15fr 1fr'),
 lines:['Lecture 26’s courtyard photo. Before any boxes: if we gave it to a classifier from Part 5, what would come back?',
  'One label for the whole photo, with its probability. The cat, the plants, and where anything sits, are all lost.',
  `A detector returns a list: one row per object, each a class, a score and a box. Another photo gives a list of another length.`],
 ask:['What would an image classifier return for this photo?','What does that answer leave out?','How many numbers describe one detection, and what changes from photo to photo?'],
 expected:['One class with a probability: a softmax over its classes, as in Part 5.','The second animal, the plants, how many objects there are, and where each one is.',
  `A class, a score and ${word(nBox)} numbers for the box. What changes is the number of rows: ${word(FIN.length)} here.`],
 point:'The Output row changes: a list of boxes, each with a class and a score, whose length depends on the photo.',
 misconception:'That a detector runs a classifier once per object. It is one network, run once, that writes a box and scores in every cell of a grid.',
 transition:'To train boxes or to judge them, we first need to say when two boxes agree.',
 source:SRC(PHOTO,'Classifier: ResNet-50 on ImageNet, recorded from Lecture 26 (data/models.json).',YOLO),from:[26,'#pipeline/3']});

// ---------------------------------------------------------------------------------------- 7.2 IoU, live
chap('7.2','At test time: do two boxes agree?');
function iouOf(a,b){
 const w=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0])),h=Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1])),i=w*h;
 const u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i;
 return {w,h,i,u,iou:i/u,box:[Math.max(a[0],b[0]),Math.max(a[1],b[1]),Math.min(a[2],b[2]),Math.min(a[3],b[3])]};
}
const DOG=IU.dog;
const iouStatic=rect(DOG,{stroke:COL.target,w:6})+pill(DOG[0],DOG[1],'dog, annotated',{fill:COL.target,size:30,maxX:PH.W});
function iouLive(dx,dy){
 const g=[DOG[0]+dx,DOG[1]+dy,DOG[2]+dx,DOG[3]+dy],q=iouOf(DOG,g);
 return {svg:(q.i>0?rect(q.box,{stroke:'none',w:0,fill:COL.guess,op:.25}):'')+rect(g,{stroke:COL.guess,w:6,dash:'22 12'})+pill(g[2],g[3],'guess',{fill:COL.guess,size:30,anchor:'end',maxX:PH.W}),
  read:`<span>overlap ${num(q.w,0)} × ${num(q.h,0)} = ${big(q.i)}</span><span>union ${big(q.u)}</span><span>IoU <b>${num(q.iou)}</b></span>`};
}
const iouAlt='The annotated dog box in teal and a purple guess box that two sliders move; the overlap is shaded and its IoU printed';
register('p7-iou',(el,ctx)=>{
 const frame=(inner,read)=>`<div class="w-fig p7-iou">${courtyard(iouStatic+`<g class="live">${inner}</g>`,iouAlt)}<p class="p7-read">${read}</p></div>`;
 if(ctx.printing){const q=iouLive(...IU.shift);el.innerHTML=frame(q.svg,q.read);return;}
 el.innerHTML=frame('','')+`<div class="w-controls">${slider('dx','right',...IU.dx,0)}${slider('dy','down',...IU.dy,0)}</div>`;
 const live=el.querySelector('.live'),read=el.querySelector('.p7-read'),dx=el.querySelector('[data-k="dx"]'),dy=el.querySelector('[data-k="dy"]');
 const show=()=>{const q=iouLive(+dx.value,+dy.value);live.innerHTML=q.svg;read.innerHTML=q.read;
  el.querySelector('[data-o="dx"]').textContent=`${num(+dx.value,0)} px`;el.querySelector('[data-o="dy"]').textContent=`${num(+dy.value,0)} px`;};
 for(const s of [dx,dy]){s.addEventListener('input',show);s.addEventListener('change',()=>s.blur());}
 onBuild(el,b=>{[dx.value,dy.value]=b>=2?IU.shift:[0,0];show();});                // click 2 answers with the lecture's guess
 show();
});
const [SX,SY]=IU.shift;
S_('iou','det','Do two boxes agree? Intersection over union',{builds:2,
 body:cols(widget('p7-iou',iouAlt),stack(eq(R`\mathrm{IoU}(A,B)=\frac{\text{area}(A\cap B)}{\text{area}(A\cup B)}`),
  ps('It is 1 for the same box and 0 for boxes that do not touch.'),
  answer(2,`The lecture’s guess:${eq(R`\frac{${bt(IU.inter)}}{${bt(IU.union)}}=${tn(IU.iou)}`)}At least ${num(IU.match,1)}, so it counts as a match.`)),'1.25fr 1fr'),
 lines:[`The ${kt('annotated dog')} and a ${kp('guess')} the sliders move. Lecture 26’s guess sat ${SX} px right and ${SY} px down. What IoU would you give it?`,
  'A small shift already costs a lot. IoU decides three things today: which boxes NMS deletes, which detections count as right, and in training which cell owns a box.'],
 ask:[`What IoU does a guess ${SX} px right and ${SY} px down get?`,'Where else will IoU be used in this part?'],
 expected:[`${num(IU.iou)}: overlap ${big(IU.inter)} over union ${big(IU.union)}. Most of the class will guess higher.`,
  'In NMS, to find duplicates (7.4), and in evaluation, to call a detection right (7.5).'],
 point:'IoU is overlap over union: one number for how well two boxes agree, used in training, in NMS and in evaluation.',
 misconception:`That a box ${SX} px off is nearly perfect. Both boxes are ${big(IU.area)} px², and the shift already drops IoU to ${num(IU.iou)}.`,
 demo:`Start at 0 and 0: IoU 1. Drag right to ${SX} and down to ${SY}: ${num(IU.iou)}. Then drag right all the way left, until the boxes part: 0.`,
 transition:'Now train a network to write boxes: one target per grid cell.',
 source:SRC(PHOTO,'The dog box: Lecture 26’s annotation; the guess: its box moved by hand.'),from:[26,'#iou-union/3']});

// ---------------------------------------------------------------------------------------- 7.3 the grid and its loss
chap('7.3','Loss: one target per grid cell');
const [DR,DC]=G.dog_cell,S3=G.s;
const fx=v=>Number.isInteger(v)?String(v):num(v),fsq=v=>num(v,v<.01?3:2);
const gridLines=[...Array(G.cols-1)].map((_,k)=>line((k+1)*S3,0,(k+1)*S3,PH.H,{stroke:'#fff',w:7}))
 .concat([...Array(G.rows-1)].map((_,k)=>line(0,(k+1)*S3,PH.W,(k+1)*S3,{stroke:'#fff',w:7}))).join('');
const cellLoss=G.cell_loss.flatMap((row,r)=>row.map((v,c)=>label(c*S3+S3/2,r*S3+S3/2+22,num(v),{size:60,anchor:'middle',fill:COL.loss,weight:700,halo:true}))).join('');
const gridFig=courtyard(gridLines+rect(DOG,{stroke:COL.target,w:5})+dot(...G.dog_centre,14,{fill:COL.target,w:4})+
 rect([DC*S3,DR*S3,(DC+1)*S3,(DR+1)*S3],{stroke:COL.amber,w:12})+gat(3,cellLoss),
 `The photo cut into a ${G.cols} by ${G.rows} grid of ${S3}-pixel cells; the dog's centre lies in cell (${DR}, ${DC}); from click 3, each cell's loss`);
const head=['',...G.keys.map(k=>k.startsWith('b_')?m(k):k)];
const worst=G.sq.indexOf(Math.max(...G.sq)),wkey=G.keys[worst];
const emptyO=Math.sqrt(G.cell_loss[0][0]);                                // an empty cell pays (ô − 0)², so its ô is the square root
S_('det-loss','det','The loss, one grid cell at a time',{builds:3,
 body:cols(gridFig,stack(`<div class="table-sm xtight p7-rec">${table(head,[
   [`network ${m(R`\qcol{\hat y}`)}`,...G.pred.map(v=>kp(num(v)))],
   [`target ${m(R`\pcol{y}`)}`,...G.target.map(v=>kt(fx(v)))],
   [at(2,'squared'),...G.sq.map(v=>at(2,kl(fsq(v))))]])}</div>`,
  at(2,sm(eq(R`\textstyle\sum_{i=1}^{${G.keys.length}}(\qcol{\hat y_i}-\pcol{y_i})^2=\lcol{${tn(G.dog_loss)}}`))),
  at(3,sm(eq(R`\lcol{\mathcal L}=\text{the sum over all ${word(G.rows*G.cols)} cells}=\lcol{${tn(G.loss)}}`))),
  at(3,small(`After ${G.steps} Adam steps on this one photo: ${kl(num(G.loss_end,3))}.`))),'0.95fr 1.05fr'),
 lines:[`A ${G.cols} × ${G.rows} grid of ${S3}-pixel cells. The dog’s centre lies in cell (${DR}, ${DC}), so that cell’s ${kt('target')} holds its box. The untrained ${kp('network')} writes this.`,
  `Square each difference and add: the ${kl('loss')} of the dog’s cell. An empty cell pays only for its object? score.`,
  `Every cell’s loss is written in its cell. Gradient steps on their sum move the network’s table toward the target table.`],
 ask:['Which cell is responsible for the dog, and why that one?','Which entry of the dog’s record costs the most?','Why do the four empty cells cost anything at all?'],
 expected:[`Cell (${DR}, ${DC}): it holds the centre of the dog’s box, (${num(G.dog_centre[0],1)}, ${num(G.dog_centre[1],1)}).`,
  `${wkey}: the target is ${fx(G.target[worst])} and the untrained network says ${num(G.pred[worst])}, a squared error of ${fsq(G.sq[worst])}.`,
  `Their target object? is 0, and the untrained network says about ${num(emptyO)} everywhere, so each pays ${num(G.cell_loss[0][0])}. Their box and class entries are skipped.`],
 point:'The Data row now holds boxes, and the Loss row compares a target record with the network’s record in every cell. Lecture 26 used squared error throughout; real detectors use binary cross-entropy (Part 1’s −ln p) for object? and the class scores, and a box loss for the box.',
 misconception:'That every cell is trained to find the dog. Only the cell holding its centre gets the dog’s box; the others learn object? = 0.',
 transition:'After training, every cell writes a box. Most of them are not wanted.',
 source:SRC(PHOTO,'The network: Lecture 26’s small grid CNN; its table before training and its loss after training recorded from data/models.json.'),from:[26,'#target-photo/3']});

// ---------------------------------------------------------------------------------------- 7.4 threshold and NMS, live
chap('7.4','At test time: threshold, then NMS');
const CUTS=Object.keys(DT.by_cut),C0=CUTS.findIndex(k=>+k===DT.default),B0=DT.by_cut[CUTS[C0]];
function nmsLive(c,on){
 const B=DT.by_cut[c],ids=on?B.keep:B.pass;
 const near=i=>B.keep.reduce((a,j)=>DT.boxes[j].name===DT.boxes[i].name&&iouOf(DT.boxes[i].box,DT.boxes[j].box).iou>iouOf(DT.boxes[i].box,DT.boxes[a].box).iou?j:a,B.keep.find(j=>DT.boxes[j].name===DT.boxes[i].name));
 const tabs=on?'':B.keep.map(j=>{const n=B.pass.filter(i=>near(i)===j).length,k=DT.boxes[j];return pill(k.box[0],k.box[1],`${n} ${short(k.name)} box${n===1?'':'es'}`,{fill:colOf(k.name),size:30,maxX:PH.W});}).join('');
 return {svg:ids.map(i=>on?detBox(DT.boxes[i]):rect(DT.boxes[i].box,{stroke:colOf(DT.boxes[i].name),w:2})).join('')+tabs,
  read:`<span>score ≥ ${num(+c)}: <b>${B.pass.length}</b> boxes</span>${on?`<span>after NMS: <b>${B.keep.length}</b></span>`:'<span>NMS off</span>'}`};
}
const nmsAlt='The detector’s boxes on the courtyard photo: those that pass the score threshold, before or after NMS';
register('p7-nms',(el,ctx)=>{
 const frame=(inner,read)=>`<div class="w-fig p7-nms">${courtyard(`<g class="live">${inner}</g>`,nmsAlt)}<p class="p7-read">${read}</p></div>`;
 if(ctx.printing){const q=nmsLive(CUTS[C0],true);el.innerHTML=frame(q.svg,q.read);return;}
 el.innerHTML=frame('','')+`<div class="w-controls"><button type="button" data-nms aria-pressed="false">NMS</button>${slider('c','keep scores ≥',0,CUTS.length-1,1,C0)}</div>`;
 const live=el.querySelector('.live'),read=el.querySelector('.p7-read'),range=el.querySelector('[data-k="c"]'),out=el.querySelector('[data-o="c"]'),btn=el.querySelector('[data-nms]');
 let on=false;
 const show=()=>{const c=CUTS[+range.value],q=nmsLive(c,on);live.innerHTML=q.svg;read.innerHTML=q.read;out.textContent=num(+c);};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());
 const setOn=v=>{on=v;btn.setAttribute('aria-pressed',on);};
 btn.addEventListener('click',()=>{setOn(!on);btn.blur();show();});
 onBuild(el,b=>{range.value=C0;setOn(b>=3);show();});                                // click 3 shows the answer: NMS on
 show();
});
const keepAt=c=>DT.by_cut[c].keep.length,lowest=FIN.reduce((a,k)=>k.score<a.score?k:a);
const dropAt=CUTS.find(c=>keepAt(c)<FIN.length),oneAt=CUTS.find(c=>keepAt(c)===1);
S_('nms','det','Keep the confident boxes, then drop the duplicates',{builds:3,
 body:cols(widget('p7-nms',nmsAlt),stack(
  `<dl class="kv p7-kv"><dt>records, ${word(DT.per_level.length)} grids</dt><dd>${big(DT.records)}</dd><dt>score ≥ ${num(DT.default)}</dt><dd>${B0.pass.length}</dd><dt>after NMS</dt><dd><div>${only('1-2','?')}${only(3,String(B0.keep.length))}</div></dd></dl>`,
  at(2,`<ol class="steps cons p7-steps"><li>Keep the box with the best score.</li><li>Delete boxes of its class with IoU above ${num(DT.iou_thr,1)} with it.</li><li>Repeat on the boxes left.</li></ol>`),
  at(3,small(`The ${word(DT.n_dog-1)} other dog boxes had IoU ${num(DT.dog_iou_min)} or more with the kept one.`))),'1.25fr 1fr'),
 lines:[`YOLOv8n writes a box and scores in every cell of three grids. ${B0.pass.length} boxes pass a score of ${num(DT.default)}, several per object. How do we keep one each?`,
  'Non-maximum suppression (NMS) keeps the best box and deletes its near copies, one class at a time. Press NMS in the figure.',
  `${Word(B0.keep.length)} detections survive, one per object: the list of 7.1. A fixed number of records became a list of any length.`],
 ask:['Why does one dog get many boxes?','Why only boxes of the same class?','What happens as the score threshold rises?'],
 expected:[`Neighbouring cells all see the dog and each writes a box for it: ${DT.n_dog} dog boxes pass ${num(DT.default)}.`,
  'A person on a bicycle: the two boxes overlap and both are right. NMS run per class keeps both.',
  `Fewer boxes pass, and NMS still leaves one per object. At ${num(+dropAt)} the small ${lowest.name} (${num(lowest.score)}) is gone; at ${num(+oneAt)} only the ${DT.boxes[DT.by_cut[oneAt].keep[0]].name} is left.`],
 point:'At test time two choices turn a fixed grid of records into a list: a score threshold, and NMS with an IoU threshold.',
 misconception:'That NMS keeps only the single best box. It loops, so every object keeps its own best box, and boxes of other classes are never compared.',
 demo:`Start with NMS off at ${num(DT.default)}: ${B0.pass.length} boxes. Press NMS: ${B0.keep.length}. Then drag the threshold up and down, with NMS on and off.`,
 transition:'Is that list any good? Count what it got right.',
 source:SRC(PHOTO,YOLO,'Class-aware NMS rerun here at every threshold.'),from:[26,'#nms-dog/4']});

// ---------------------------------------------------------------------------------------- 7.5 precision, recall, AP, live
chap('7.5','At test time: precision, recall, AP');
const NP=PR.scores.length,EW=1152,EH=320,RY=62,RH=31;
const PP=panel({ox:430,W:EW-430,H:EH,L:72,R:28,T:66,B:56,xMin:0,xMax:1,yMin:0,yMax:1.05,ticksX:[0,.2,.4,.6,.8,1],ticksY:[0,.5,1],xLabel:'recall',yLabel:'precision',fs:20});
const ok=i=>PR.hit[i],vCol=i=>ok(i)?COL.green:COL.loss,hTxt=v=>String(+tn(v));
const listFig=label(14,32,'rank',{size:20})+label(110,32,'score',{size:20})+label(220,32,`IoU ≥ ${num(PR.match,1)}?`,{size:20})+line(10,44,390,44,{stroke:AX})+
 PR.scores.map((s,i)=>label(14,RY+i*RH+12,`#${i+1}`,{size:21,fill:COL.ink})+label(110,RY+i*RH+12,num(s),{size:21,fill:COL.ink})+
  label(220,RY+i*RH+12,ok(i)?'TP':'FP',{size:21,fill:vCol(i),weight:700})).join('');
const envRects=(()=>{let prev=0,s='';PR.recall.forEach((r,i)=>{if(r>prev){
  s+=`<rect x="${PP.x(prev).toFixed(1)}" y="${PP.y(PR.envelope[i]).toFixed(1)}" width="${(PP.x(r)-PP.x(prev)).toFixed(1)}" height="${(PP.y(0)-PP.y(PR.envelope[i])).toFixed(1)}" fill="${COL.guess}" fill-opacity=".14" stroke="${COL.guess}" stroke-width="2"/>`+
   label((PP.x(prev)+PP.x(r))/2,PP.y(PR.envelope[i]/2)+7,hTxt(PR.envelope[i]),{size:20,anchor:'middle',fill:COL.ink,weight:650});prev=r;}});return s;})();
const prStatic=listFig+PP.axes+gat(3,envRects);
/** The top k boxes kept: the rows below the cut faded, the curve drawn up to the k-th point, the counts at the top. */
function prLive(k){
 const i=k-1,yb=RY+k*RH-9,pts=PR.recall.slice(0,k).map((r,j)=>[PP.x(r),PP.y(PR.precision[j])]),rx=EW-28;
 return `<rect x="6" y="${yb}" width="390" height="${EH-yb}" fill="${PAPER}" opacity=".72"/>`+line(6,yb,396,yb,{stroke:COL.loss,w:3,dash:'8 6'})+
  path(pts,{stroke:COL.guess,w:3})+dot(...pts[i],14,{fill:COL.guess,w:3})+pts.map(([x,y],j)=>dot(x,y,7,{fill:vCol(j)})).join('')+
  label(rx-400,22,`TP ${PR.tp[i]}`,{size:21,fill:COL.green,weight:700})+label(rx-300,22,`FP ${PR.fp[i]}`,{size:21,fill:COL.loss,weight:700})+label(rx-200,22,`FN ${PR.fn[i]}`,{size:21,fill:COL.amber,weight:700})+
  label(rx,22,`${k} of ${NP} kept`,{size:21,fill:COL.ink,anchor:'end'})+
  label(rx-400,48,`precision ${num(PR.precision[i])}`,{size:21,fill:COL.guess,weight:700})+label(rx,48,`recall ${num(PR.recall[i])}`,{size:21,fill:COL.guess,weight:700,anchor:'end'});
}
const prAlt='Eight ranked boxes marked TP or FP beside the precision and recall curve; a slider sets how many boxes are kept';
register('p7-pr',(el,ctx)=>{
 const frame=inner=>`<div class="w-fig"><svg class="plot wsvg" viewBox="0 0 ${EW} ${EH}" role="img" aria-label="${prAlt}">${prStatic}<g class="live">${inner}</g></svg></div>`;
 if(ctx.printing){el.innerHTML=frame(prLive(NP));return;}
 el.innerHTML=frame('')+`<div class="w-controls">${slider('k','keep the top boxes',1,NP,1,1)}</div>`;
 const live=el.querySelector('.live'),range=el.querySelector('[data-k="k"]'),out=el.querySelector('[data-o="k"]');
 const show=()=>{const k=+range.value;live.innerHTML=prLive(k);out.textContent=`${k}, score ≥ ${num(PR.scores[k-1])}`;};
 range.addEventListener('input',show);range.addEventListener('change',()=>range.blur());
 onBuild(el,b=>{range.value=b>=2?NP:1;show();});                                     // clicks 2 and 3 talk about all the boxes
 show();
});
const L_=NP-1,steps=PR.recall.map((r,i)=>[r-(i?PR.recall[i-1]:0),PR.envelope[i]]).filter(([w])=>w>0);
const apTex=steps.map(([w,h])=>`${+tn(w,1)}\\times${+tn(h)}`).join('+');
const missed=PR.n_gt-PR.tp[L_];
S_('det-eval','det','How good is the list? Precision, recall and AP',{builds:3,
 body:stack(widget('p7-pr',prAlt),`<div>${only(2,`<div class="answer">All ${word(NP)} boxes: precision ${PR.tp[L_]}/${NP} = ${num(PR.precision[L_],1)}, recall ${PR.tp[L_]}/${PR.n_gt} = ${num(PR.recall[L_],1)}. Lowering the threshold moves the point right and, on the whole, down.</div>`)}${
  only(3,`<div class="answer">${m(R`\mathrm{AP}=${apTex}\approx${tn(PR.ap)}`)}</div>`)}</div>`),
 lines:[`Lecture 26’s teaching example: ${word(PR.n_gt)} dogs, ${word(NP)} ranked boxes, each a TP or FP by IoU ≥ ${num(PR.match,1)}. Keep more boxes, one at a time: where does the point go?`,
  'Each threshold is one point on the curve. Recall can only grow as boxes are added; precision falls whenever a wrong box comes in.',
  'One number for the whole curve: the area under its upper envelope, the average precision (AP).'],
 ask:['With only the top box kept, what are precision and recall?',`Why can recall never reach 1 here?`,'When would AP be 1?'],
 expected:[`Precision ${PR.tp[0]}/1 = ${num(PR.precision[0],0)}, recall ${PR.tp[0]}/${PR.n_gt} = ${num(PR.recall[0],1)}: right, but most dogs are missed.`,
  `${Word(missed)} dog, the small one far away, has no box at all, so no threshold can find it.`,
  'Only if every dog is found and every true positive is ranked above every false positive.'],
 point:'At test time we choose the threshold, and it trades precision against recall; AP scores the whole ranking at once.',
 misconception:'That accuracy would do. A true negative is “no box and no object”, and every empty patch of grass is one, so detection counts only TP, FP and FN.',
 demo:`Drag from the top box to all ${word(NP)}, reading precision and recall at each step. Stop at all ${word(NP)}: the lecture’s point.`,
 transition:'One question that needs IoU and NMS together.',
 source:SRC(`The example: Lecture 26’s dog park, ${word(PR.n_gt)} annotated dogs and ${word(NP)} made-up ranked boxes (data/dog-park.json); TP, FP, precision, recall and AP recomputed here.`),from:[26,'#ap-area/3']});

// ---------------------------------------------------------------------------------------- check
chap();
const GUESS=IU.guess;
S_('check-7','det','Check: two boxes on one dog',{builds:2,
 body:cols(courtyard(rect(DOG,{stroke:COL.guess,w:6})+pill(DOG[0],DOG[1],`dog ${num(CK.score_a)}`,{fill:COL.guess,size:30})+
   rect(GUESS,{stroke:COL.guess,w:6,dash:'22 12'})+pill(GUESS[2],GUESS[3],`dog ${num(CK.score_b)}`,{fill:COL.guess,size:30,anchor:'end',maxX:PH.W}),
   'Two dog boxes on the same dog: one on it, one moved right and down',{view:[330,40,560,600]}),
  stack(`<p class="p7-q">Two dog boxes: one on the dog (${num(CK.score_a)}) and the guess of 7.2 (${num(CK.score_b)}), with IoU ${num(CK.iou)}. NMS deletes same-class boxes above ${num(CK.nms,1)}; a match needs IoU ≥ ${num(CK.match,1)}. How many dogs are reported, and what is the precision?</p>`,
   answer(2,`${Word(CK.survive)}: ${num(CK.iou)} is below ${num(CK.nms,1)}, so NMS keeps both. The first box finds the dog; the second is a duplicate, a false positive. Precision 1/${CK.survive} = ${num(CK.precision,1)}.`)),'0.72fr 1.28fr'),
 lines:['Two ideas at once, IoU and NMS. Work it out with your neighbour before the next click.',
  `NMS let the duplicate through, and evaluation counted it against us. An NMS threshold below ${num(CK.iou)} would delete it and give precision 1.`],
 ask:['How many dogs are reported, and what is the precision?','What would you change to fix it, and what could go wrong?'],
 expected:[`${Word(CK.survive)} dogs; precision ${num(CK.precision,1)}.`,
  `Lower the NMS threshold below ${num(CK.iou)}. In a crowd that can delete a real neighbour whose box overlaps that much.`],
 point:'A duplicate that NMS misses costs precision; a threshold that is too low costs recall in crowds.',
 transition:'The part in one slide.',
 source:SRC(PHOTO,'The two scores are made up for this question; the IoU is the guess of 7.2.')});

// ---------------------------------------------------------------------------------------- recap
recap(7,[`Every grid cell writes a box and scores.`,
  `IoU compares boxes; the loss compares cells.`,
  `A threshold and NMS turn ${big(DT.records)} records into ${word(FIN.length)} boxes.`],
 'all seven parts side by side.',
 {card:RECIPES.det,title:'Object detector',sm:true,ask:'Which row of the recipe did detection leave alone?',
  expected:'The Update: the same gradient step as every other model in the course. Data, Output, Loss and At test time all changed.',
  point:'Detection is the course’s recipe with a new output: the training loop is unchanged, and the new work is in the targets and at test time.',
  transition:'Now all seven parts, side by side.'});

// ---------------------------------------------------------------------------------------- appendix: the two tools in code
O('p7-code','IoU and NMS in PyTorch',{builds:1,
 body:cols(`<div class="p7-code">${code(['def box_iou(a, b):  # a: (N, 4), b: (M, 4)','    lt = torch.max(a[:, None, :2], b[:, :2])','    rb = torch.min(a[:, None, 2:], b[:, 2:])',
   '    inter = (rb - lt).clamp(min=0).prod(-1)','    area_a = (a[:, 2:] - a[:, :2]).prod(-1)','    area_b = (b[:, 2:] - b[:, :2]).prod(-1)','    union = area_a[:, None] + area_b - inter','    return inter / union','',
   `a = torch.tensor([[${IU.dog.map(v=>v+'.').join(', ')}]])`,`b = torch.tensor([[${IU.guess.map(v=>v+'.').join(', ')}]])`,`box_iou(a, b)  # ${num(IU.iou,3)}`].join('\n'))}</div>`,
  `<div class="p7-code">${code([`def nms(boxes, scores, thr=${DT.iou_thr}):`,'    order = scores.argsort(descending=True)','    keep = []','    while order.numel() > 0:','        i, rest = order[:1], order[1:]',
   '        keep.append(i)','        ious = box_iou(boxes[i], boxes[rest])[0]','        order = rest[ious <= thr]','    return torch.cat(keep)'].join('\n'))}</div>`,'1fr 1fr'),
 ask:['Why does nms run once per class in a real detector?'],
 expected:['So that boxes of different classes never delete each other: torchvision’s batched_nms does exactly that.'],
 point:'Lecture 26’s two tools in a few lines each; the courtyard’s boxes go through them in scripts/sci/p7.py.',
 source:'Code: Lecture 26 (iou-code, nms-code).'});
