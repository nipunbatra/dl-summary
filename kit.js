// The authoring kit of the summary lecture. Every part file in parts/ imports from here and from nowhere else:
// the lecture's meta, the one slide list, the slide helpers, the route strip and the recipe card.
// AUTHORING.md explains how to use it.
import {at,esc,highlight,m} from './shared/author.js';
export * from './shared/author.js';
export * from './widgets.js';

export const meta={
 id:'dl-summary',number:27,title:'The course in one lecture',series:'ES 667 Deep Learning',
 subtitle:'Twenty-six lectures, one recipe',minutes:90,
 sections:[
  {id:'map',title:'Lecture map',minutes:7},
  {id:'found',title:'Part 1 · Foundations',minutes:11},
  {id:'grad',title:'Part 2 · Gradients and optimization',minutes:12},
  {id:'gen',title:'Part 3 · Generalization',minutes:9},
  {id:'attn',title:'Part 4 · Attention and language',minutes:12},
  {id:'vision',title:'Part 5 · Vision and language',minutes:13},
  {id:'genm',title:'Part 6 · Generative modelling',minutes:12},
  {id:'det',title:'Part 7 · Object detection',minutes:8},
  {id:'wrap',title:'The whole course',minutes:6},
  {id:'optional',title:'Appendix · More depth',minutes:0},
 ]};
/** The seven parts, in order: [section id, short name for the route strip, the lectures it covers, the recipe slots
 it changed]. The route slide and each part's recap tint those slots. */
export const PARTS=[['found','Foundations','1 to 4',['model','out','loss']],['grad','Gradients','5 to 9',['step']],
 ['gen','Generalization','10',['data','loss','test']],['attn','Attention','11 to 15',['data','model','out','test']],
 ['vision','Vision','16 to 22',['data','model','out','loss','test']],['genm','Generative','23 to 25',['data','model','out','loss','test']],
 ['det','Detection','26',['data','out','loss','test']]];

/** A data file written by a script in scripts/sci/. Slides never type a number; they read it from here. */
export async function data(name){
 const r=await fetch(new URL(`data/${name}.json`,import.meta.url),{cache:'no-store'});
 if(!r.ok)throw new Error(`data/${name}.json is missing: run its script in scripts/sci/`);
 return r.json();
}
/** The course as the schedule page lists it (scripts/sci/course.py). */
export const course=await data('course');
const lectureOf=n=>{const l=course.lectures.find(l=>l.n.includes(n));if(!l)throw new Error('No lecture '+n);return l;};
/** Where a slide's idea was taught in full: {href, label} for the top-right link of the slide.
 from(12) links Lecture 12's slides; from(25,'#fix-loss/4') adds a deep link; kind picks 'slides', 'notes', 'pdf' or 'recording'. */
export function from(n,hash='',kind='slides'){
 const l=lectureOf(n),a=l.links.find(x=>x.kind===kind)||l.links.find(x=>x.kind==='notes')||l.links[0];
 return {href:a.href.replace(/#.*$/,'')+hash,label:`From Lecture ${n}`};
}

// ---------------------------------------------------------------------------------------- small text helpers
export const R=String.raw;
export const p=s=>`<p class="lead">${s}</p>`;
export const ps=s=>`<p class="small">${s}</p>`;
export const small=s=>`<p class="caption">${s}</p>`;
export const stack=(...a)=>`<div class="equation-stack">${a.join('')}</div>`;
export const sm=(...a)=>`<div class="equation-stack eq-sm">${a.join('')}</div>`;
/** Colour words the way the maths is coloured: target (teal), prediction (purple), loss (red). */
export const kt=s=>`<span class="kt">${s}</span>`,kp=s=>`<span class="kp">${s}</span>`,kl=s=>`<span class="kl">${s}</span>`;
/** Code that appears a line per click: rows of [build, line]; build 0 shows the line from the start. */
export const codeSteps=rows=>`<pre class="code code-steps"><code>${rows.map(([b,l])=>`<span class="cl"${b?` data-at="${b}"`:''}>${highlight(l)||' '}</span>`).join('')}</code></pre>`;
export const takeaway=s=>`<div class="takeaway"><b>Takeaway.</b> ${s}</div>`;
/** The answer to a question the slide asks, shown from click b on. */
export const answer=(b,s)=>at(b,`<div class="answer">${s}</div>`);

// ---------------------------------------------------------------------------------------- the recipe card
/** The six slots of the recipe. Every model of the course fills them. */
export const SLOTS=[['data','Data'],['model','Model'],['out','Output'],['loss','Loss'],['step','Update'],['test','At test time']];
/** A recipe card. fill: {data, model, out, loss, step, test} (HTML; leave a slot out to drop its row).
 hi: the slots this part changed, drawn tinted. title: a small heading, e.g. the model's name. sm: smaller type. */
export function card(fill,{hi=[],title='',sm=false}={}){
 for(const k of Object.keys(fill))if(!SLOTS.some(s=>s[0]===k))throw new Error('Unknown recipe slot '+k);
 return `<div class="recipe${sm?' sm':''}">${title?`<h3>${title}</h3>`:''}<dl>${SLOTS.filter(([k])=>fill[k]!=null).map(([k,name])=>
  `<dt${hi.includes(k)?' class="hi"':''}>${name}</dt><dd${hi.includes(k)?' class="hi"':''}>${fill[k]}</dd>`).join('')}</dl></div>`;
}

// ---------------------------------------------------------------------------------------- slides
export const slides=[];
let chapter=null;
/** Name the chapter shown in the kicker of the slides that follow: chap('2.1','Backpropagation'). chap() clears it. */
export const chap=(k,title)=>{chapter=k?[k,title]:null;};
/** One slide. o: {builds, body, lines[], ask[], expected[], point, misconception, transition, source, from, optional}.
 lines, ask and expected need one entry per build. */
export function S_(id,section,title,o){
 const builds=o.builds||1;
 for(const k of ['lines','ask','expected'])if(!Array.isArray(o[k])||o[k].length!==builds)throw new Error(`${id}: ${k} needs ${builds} entries`);
 if(!meta.sections.some(s=>s.id===section))throw new Error(`${id}: unknown section ${section}`);
 if(slides.some(s=>s.id===id))throw new Error(`Slide id used twice: ${id}`);
 const ch=section==='optional'||section==='map'||section==='wrap'||o.nochap?null:chapter;
 const link=o.from==null?null:typeof o.from==='number'?from(o.from):Array.isArray(o.from)?from(...o.from):o.from;
 slides.push({id,section,title,builds,body:o.body,prompts:o.lines,chapter:ch?`${ch[0]} ${ch[1]}`:'',notebook:link,
  notes:{ask:o.ask,expected:o.expected,point:o.point||'',misconception:o.misconception||'',transition:o.transition||'',demo:o.demo||''},
  source:o.source||'',optional:!!o.optional});
}
/** An appendix slide: off the main route, shown with ?all and in the reading and print views. */
export const O=(id,title,o)=>S_(id,'optional',title,{...o,optional:true,lines:o.lines||o.ask});

/** The route strip: where part k sits among the seven. */
export const strip=k=>`<div class="act-strip" role="img" aria-label="Part ${k} of ${PARTS.length}">${PARTS.map((t,i)=>`<span class="${i+1<k?'done':i+1===k?'now':''}">${i+1} · ${t[1]}</span>`).join('')}</div>`;
/** The slide that opens part k: its aim in one sentence and the three or four steps it takes.
 o: {line, ask, expected, transition}. The slide's id is part-k. */
export function partBreak(k,title,lead,how,o){
 chapter=null;
 S_(`part-${k}`,PARTS[k-1][0],`Part ${k} · ${title}`,{builds:1,nochap:true,
  body:`<div class="part-break">${strip(k)}${p(lead)}<ol class="steps">${how.map(x=>`<li>${x}</li>`).join('')}</ol><p class="caption">${/ to /.test(PARTS[k-1][2])?'Lectures':'Lecture'} ${PARTS[k-1][2]}</p></div>`,
  lines:[o.line],ask:[o.ask],expected:[o.expected],transition:o.transition});
}
/** The slide that closes part k: what the part did (two or three lines) beside the recipe card with this part's
 slots tinted (hi defaults to the part's slots in PARTS). o: {card, hi, title, sm, ask, expected, point, transition}.
 The slide's id is recap-k. */
export function recap(k,did,next,o){
 S_(`recap-${k}`,PARTS[k-1][0],`Part ${k} in one slide`,{builds:1,nochap:true,
  body:`<div class="part-break">${strip(k)}<div class="columns" style="grid-template-columns:1fr 1.15fr;align-items:start"><ol class="steps tight">${did.map(x=>`<li>${x}</li>`).join('')}</ol>${card(o.card,{hi:o.hi||PARTS[k-1][3],title:o.title||'',sm:o.sm})}</div></div>`,
  lines:[`<b>Next:</b> ${next}`],ask:[o.ask],expected:[o.expected],point:o.point||'',transition:o.transition});
}
/** A number for use inside TeX: fixed decimals, an ASCII minus. tn(-0.2689) is '-0.27'. */
export const tn=(v,n=2)=>{const s=Number(v).toFixed(n);return /^-0(\.0*)?$/.test(s)?s.slice(1):s;};
/** Styles that belong to one part: css('.p3-key{color:var(--q)}'). Start every class name with the part, pN-. */
export const css=text=>{const el=document.createElement('style');el.textContent=text;document.head.append(el);};
const WORDS='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five twenty-six twenty-seven twenty-eight twenty-nine thirty'.split(' ');
/** A small count in words, as the slides write them: word(26) is 'twenty-six'; Word(26) starts a sentence. */
export const word=n=>WORDS[n]??String(n),Word=n=>word(n).replace(/^./,c=>c.toUpperCase());
/** The models the course built, in the order it built them: id, name, the lectures, what the model returns. */
export const MODELS=[
 {id:'mlp',name:'MLP classifier',lec:[4],out:'a class'},
 {id:'charlm',name:'Character language model',lec:[11],out:'the next character'},
 {id:'gpt',name:'Transformer language model',lec:[12,13,14,15],out:'the next token'},
 {id:'vit',name:'Vision Transformer',lec:[16],out:'the class of a photo'},
 {id:'clip',name:'CLIP',lec:[17],out:'which caption fits which photo'},
 {id:'vlm',name:'Vision-language model',lec:[18,19],out:'words about a photo'},
 {id:'cbm',name:'Concept bottleneck model',lec:[20,21],out:'concepts, then a class'},
 {id:'bnn',name:'Bayesian neural network',lec:[23,24],out:'a prediction and its uncertainty'},
 {id:'vae',name:'Variational autoencoder',lec:[25],out:'new images'},
 {id:'det',name:'Object detector',lec:[26],out:'a list of boxes'},
];
/** Guard for part files: a readable error if a number the slide needs is not in its data file. */
export const need=(obj,...keys)=>{for(const k of keys)if(obj?.[k]===undefined)throw new Error(`Data key missing: ${k}`);return obj;};
export {esc};

// ---------------------------------------------------------------------------------------- the ten recipes
const STEP=m(R`\theta\leftarrow\theta-\eta\,\nabla_{\!\theta}\,\lcol{\text{loss}}`),SAME='a gradient step, as before';
const P_=m(R`\qcol{p}`),NLL=s=>m(R`\lcol{-\ln p(\text{${s}})}`);
/** The recipe of every model in MODELS, slot by slot. A part's recap shows one of these (card: RECIPES.gpt, or
 {...RECIPES.mlp, step:'…'} to reword a slot); the closing table of the lecture shows all ten. */
export const RECIPES={
 mlp:{data:`pairs ${m('(x,\\,y)')}: an input and its class`,model:`an MLP gives scores ${m(R`z=f_\theta(x)`)}`,out:`probabilities ${m(R`\qcol{p}=\operatorname{softmax}(z)`)}`,
  loss:`${m(R`\lcol{-\ln p_{\pcol{y}}}`)}, the right class`,step:STEP,test:`the class with the largest ${P_}`},
 charlm:{data:'names: three characters in, the next one as the target',model:'three embeddings, joined, then an MLP',out:`${P_} over the 27 characters`,
  loss:NLL('next character'),step:SAME,test:'sample a character, slide along, repeat'},
 gpt:{data:'text as tokens; the target at each position is the next token',model:'embeddings, then blocks of masked self-attention',out:`${P_} over the vocabulary, at every position`,
  loss:`${NLL('next token')}, averaged over positions`,step:SAME,test:'sample a token, append it, repeat'},
 vit:{data:'photos with a class; each photo is cut into patches',model:'patch embeddings and [CLS], then attention blocks',out:`${P_} over the classes, read from [CLS]`,
  loss:`${m(R`\lcol{-\ln p_{\pcol{y}}}`)}, the right class`,step:SAME,test:`the class with the largest ${P_}`},
 clip:{data:'photos with their captions; no class labels',model:'an image encoder and a text encoder, one shared space',out:`for each photo, ${P_} over the captions in the batch`,
  loss:`${NLL('its own caption')}, and the same from caption to photo`,step:SAME,test:'write each class as a caption, pick the closest'},
 vlm:{data:'a photo, a question and the answer in words',model:'an image encoder, a projector, then a language model',out:`${P_} over the vocabulary, for each answer token`,
  loss:`${NLL('next token')}, on the answer`,step:SAME,test:'generate the answer one token at a time'},
 cbm:{data:`photos with concept labels ${m('c')} and a class ${m(R`\pcol{y}`)}`,model:`photo, then concepts ${m(R`\qcol{\hat c}`)}, then the class ${m(R`\qcol{\hat y}`)}`,
  out:`a probability per concept, then ${P_} over the classes`,loss:'a concept loss plus a class loss, each with a weight',step:SAME,test:'predict the concepts, then the class; a person can correct a concept'},
 bnn:{data:`a few pairs ${m('(x,\\,y)')}`,model:`a network whose weights are drawn from ${m(R`q(\theta)`)}`,out:`${m(R`\qcol{p(y\mid x,\theta)}`)} for each sampled ${m(R`\theta`)}`,
  loss:m(R`\lcol{-\ELBO}=-\E_q[\ln p(\mathcal D\mid\theta)]+\KL(q\,\|\,\text{prior})`),step:`a gradient step on ${m(R`\mu,\sigma`)}, through ${m(R`\theta=\mu+\sigma\epsilon`)}`,
  test:`average over samples of ${m(R`\theta`)}; their spread is the uncertainty`},
 vae:{data:'images alone, no labels',model:`an encoder ${m(R`q(z\mid x)`)} and a decoder ${m(R`p(x\mid z)`)}`,out:`a Gaussian over the code ${m('z')}, then over the image`,
  loss:m(R`\lcol{-\ELBO}=\text{reconstruction error}+\KL(q\,\|\,\N(0,I))`),step:`a gradient step, through ${m(R`z=\mu+\sigma\epsilon`)}`,test:`draw ${m(R`z\sim\N(0,I)`)} and decode it`},
 det:{data:'photos, each with its boxes and their classes',model:'a network with one output record per grid cell',out:'per cell: a box, an objectness score and class scores',
  loss:'box, objectness and class errors, summed over the cells',step:SAME,test:'keep the confident boxes, drop duplicates (NMS)'},
};
for(const M of MODELS)if(!RECIPES[M.id])throw new Error('No recipe for '+M.id);
