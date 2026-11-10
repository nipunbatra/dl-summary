// The SVG kit, the chart helpers and the live figures of the summary lecture.
// Every number a figure shows comes from a data/*.json file written by a script in scripts/sci/; nothing here invents one.
import {palette,esc} from './shared/author.js';

export const MUTED='#68717B',AX='#AAB3BC',PAPER=palette.paper;
export const COL={target:palette.p,guess:palette.q,loss:palette.loss,data:palette.data,amber:palette.noise,green:palette.d,ink:palette.ink};
/** Fixed decimals with a true minus sign and no "−0.00". */
export const num=(v,n=2)=>{const s=Number(v).toFixed(n);return (/^-0(\.0*)?$/.test(s)?s.slice(1):s).replace('-','−');};
/** Thousands separators, as the slides write large counts. */
export const big=v=>Number(v).toLocaleString('en-US');

/** A picture in its own pixel coordinates, with SVG drawn on top. view=[x,y,w,h] crops it. */
export function photo(name,W,H,inner,alt,{view=null,cls='',ext='jpg'}={}){
 const vb=view?view.join(' '):`0 0 ${W} ${H}`;
 return `<div class="diagram-scroll photo-fig ${cls}"><svg viewBox="${vb}" role="img" aria-label="${esc(alt)}"><image href="figures/${name}.${ext}" x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="none"/>${inner}</svg></div>`;
}
/** A rectangle from corners (x1, y1, x2, y2). */
export const rect=(b,{stroke=COL.guess,w=4,dash='',fill='none',op=1,rx=0,extra=''}={})=>
 `<rect x="${(+b[0]).toFixed(1)}" y="${(+b[1]).toFixed(1)}" width="${(b[2]-b[0]).toFixed(1)}" height="${(b[3]-b[1]).toFixed(1)}"${rx?` rx="${rx}"`:''} fill="${fill}" stroke="${stroke}" stroke-width="${w}"${dash?` stroke-dasharray="${dash}"`:''}${op<1?` opacity="${op}"`:''} ${extra}/>`;
/** A text label on a filled tab. */
export function pill(x,y,text,{fill=COL.guess,color='#fff',size=26,anchor='start',below=false,maxX=Infinity}={}){
 const w=String(text).replace(/<[^>]+>/g,'').length*size*.56+18,h=size*1.35;
 let x0=anchor==='end'?x-w:anchor==='middle'?x-w/2:x,y0=below?y:y-h;
 x0=Math.max(0,Math.min(x0,maxX-w));if(y0<0)y0=y;
 return `<g><rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="5" fill="${fill}"/><text x="${(x0+9).toFixed(1)}" y="${(y0+h*.74).toFixed(1)}" fill="${color}" font-size="${size}" font-weight="600">${text}</text></g>`;
}
/** Plain SVG text; content must already be safe. */
export const label=(x,y,s,{fill=MUTED,size=24,anchor='start',weight='',halo=false}={})=>
 `<text x="${(+x).toFixed(1)}" y="${(+y).toFixed(1)}" fill="${fill}" font-size="${size}" text-anchor="${anchor}"${weight?` font-weight="${weight}"`:''}${halo?` style="paint-order:stroke;stroke:${PAPER};stroke-width:7px;stroke-linejoin:round"`:''}>${s}</text>`;
export const line=(x1,y1,x2,y2,{stroke=MUTED,w=2,dash='',extra=''}={})=>`<line x1="${(+x1).toFixed(1)}" y1="${(+y1).toFixed(1)}" x2="${(+x2).toFixed(1)}" y2="${(+y2).toFixed(1)}" stroke="${stroke}" stroke-width="${w}"${dash?` stroke-dasharray="${dash}"`:''} ${extra}/>`;
/** An arrow with a head at (x2, y2). */
export function arrow(x1,y1,x2,y2,{stroke=COL.ink,w=3,head=14,both=false,dash=''}={}){
 const a=Math.atan2(y2-y1,x2-x1),h=(x,y,t)=>`<path d="M${x},${y} L${(x-head*Math.cos(t-.42)).toFixed(1)},${(y-head*Math.sin(t-.42)).toFixed(1)} L${(x-head*Math.cos(t+.42)).toFixed(1)},${(y-head*Math.sin(t+.42)).toFixed(1)}Z" fill="${stroke}"/>`;
 return line(x1,y1,x2,y2,{stroke,w,dash})+h(x2,y2,a)+(both?h(x1,y1,a+Math.PI):'');
}
/** A path through points [[x, y], ...]. */
export const path=(pts,{stroke=COL.guess,w=3,fill='none',dash='',op=1}={})=>
 `<path d="${pts.map((q,i)=>`${i?'L':'M'}${(+q[0]).toFixed(1)},${(+q[1]).toFixed(1)}`).join(' ')}" fill="${fill}" stroke="${stroke}" stroke-width="${w}" stroke-linejoin="round" stroke-linecap="round"${dash?` stroke-dasharray="${dash}"`:''}${op<1?` opacity="${op}"`:''}/>`;
export const dot=(x,y,r=7,{fill=COL.data,stroke='#fff',w=2}={})=>`<circle cx="${(+x).toFixed(1)}" cy="${(+y).toFixed(1)}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${w}"/>`;
export const svg=(W,H,inner,alt,cls='plot wsvg')=>`<svg class="${cls}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(alt)}">${inner}</svg>`;
export const fig=(W,H,inner,alt,cls='')=>`<div class="diagram-scroll ${cls}">${svg(W,H,inner,alt)}</div>`;
export const widget=(name,alt)=>`<div class="widget" data-widget="${name}" role="group" aria-label="${esc(alt)}"></div>`;
/** Reveals inside an SVG figure: an HTML wrapper would break out of the SVG, so use a group. */
export const gat=(b,svgInner)=>`<g data-at="${b}">${svgInner}</g>`,gonly=(b,svgInner)=>`<g data-only="${b}">${svgInner}</g>`;

/** One plotting panel with fixed axes. Returns the two scales and the axes' SVG. */
export function panel(o){
 const ox=o.ox??0,oy=o.oy??0,W=o.W??600,H=o.H??320,L=o.L??64,R=o.R??14,T=o.T??30,B=o.B??56,fs=o.fs??21;
 const {xMin,xMax}=o,yMin=o.yMin??0,yMax=o.yMax;
 const x=v=>ox+L+(v-xMin)/(xMax-xMin)*(W-L-R),y=v=>oy+H-B-(v-yMin)/(yMax-yMin)*(H-T-B);
 const tick=v=>String(+Number(v).toFixed(2)).replace('-','−');
 let axes=line(ox+L,oy+H-B,ox+W-R,oy+H-B,{stroke:AX})+line(ox+L,oy+T,ox+L,oy+H-B,{stroke:AX});
 for(const v of o.ticksX||[])axes+=label(x(v),oy+H-B+26,(o.fmtX||tick)(v),{size:fs,anchor:'middle'});
 for(const v of o.ticksY||[])axes+=label(ox+L-10,y(v)+7,(o.fmtY||tick)(v),{size:fs,anchor:'end'});
 if(o.xLabel)axes+=label(ox+W-R,oy+H-8,o.xLabel,{size:fs,anchor:'end'});
 if(o.yLabel)axes+=label(ox+L+8,oy+T-8,o.yLabel,{size:fs});
 return {x,y,axes};
}

/** A flowchart: rows of steps joined by arrows, left to right. rows: [{name, at, steps:[{k, t, n, hi, at, col}]}]; t may
 hold '\n', k is a small heading above it, n a small note under it, hi tints the step, col colours its border, at
 reveals the row (or one step) on that click. loop: [row, from, to, label] draws a return arrow under the row; out: [row, step, label] labels the arrow
 into a step. */
export function chart(rows,{W=1120,bh=96,gx=42,gy=26,fs=20,loop=null,out=null,alt='A flowchart'}={}){
 const all=[];let y=0;const pos=[];
 rows.forEach(row=>{const g=[];
  if(row.name){g.push(label(0,y+fs,esc(row.name),{size:fs*.9,fill:MUTED,weight:700}));y+=fs*1.5;}
  const n=row.steps.length,bw=(W-gx*(n-1))/n,ps=[];
  row.steps.forEach((s,i)=>{const x=i*(bw+gx),ls=s.t.split('\n'),lh=fs*1.22,kh=s.k?fs*1.12:0,th=kh+ls.length*lh+(s.n?fs*.9:0),y0=y+(bh-th)/2+fs*.92+kh,b=[];
   b.push(`<rect x="${x.toFixed(1)}" y="${y}" width="${bw.toFixed(1)}" height="${bh}" rx="12" fill="${s.hi?'#F1E8FC':'#fff'}" stroke="${s.col||(s.hi?COL.guess:'#AAB3BC')}" stroke-width="${s.hi||s.col?3.5:2}"/>`);
   if(s.k)b.push(label(x+bw/2,y0-kh-fs*.08,esc(s.k),{size:fs*.8,fill:s.col||MUTED,anchor:'middle',weight:700}));
   ls.forEach((l,j)=>b.push(label(x+bw/2,y0+j*lh,esc(l),{size:fs,fill:COL.ink,anchor:'middle',weight:s.hi?650:''})));
   if(s.n)b.push(label(x+bw/2,y0+ls.length*lh+1,esc(s.n),{size:fs*.78,fill:MUTED,anchor:'middle'}));
   if(i)b.push(arrow(x-gx+5,y+bh/2,x-5,y+bh/2,{stroke:COL.ink,w:2.5,head:11}));
   g.push(s.at?gat(s.at,b.join('')):b.join(''));ps.push([x,bw]);});
  pos.push([y,ps]);y+=bh+gy;all.push(row.at?gat(row.at,g.join('')):g.join(''));});
 let H=y-gy+4;
 if(loop){const [r,a,b,tx,atk]=loop,[yr,ps]=pos[r],xa=ps[a][0]+ps[a][1]/2,xb=ps[b][0]+ps[b][1]/2,yb=yr+bh,yl=yb+30;
  const g=line(xa,yb,xa,yl,{stroke:COL.ink,w:2.5})+line(xa,yl,xb,yl,{stroke:COL.ink,w:2.5})+arrow(xb,yl,xb,yb+3,{stroke:COL.ink,w:2.5,head:11})+label((xa+xb)/2,yl+fs+3,esc(tx),{size:fs*.85,fill:COL.ink,anchor:'middle',weight:600});
  all.push(atk?gat(atk,g):g);H=Math.max(H,yl+fs+10);}
 if(out){const [r,i,tx]=out,[yr,ps]=pos[r];all.push(label(ps[i][0]-gx/2,yr+bh/2-10,esc(tx),{size:fs*.8,fill:COL.ink,anchor:'middle',weight:600}));}
 return fig(W,Math.ceil(H),all.join(''),alt);
}

/** Vertical bars for a few named values (a distribution over classes, tokens, captions).
 items: [{name, v, col, note}]; values print above the bars with `dec` decimals. */
export function bars(items,{W=560,H=300,max=null,dec=2,fs=21,col=COL.guess,alt='A bar chart',gap=.34,ox=0,oy=0,bare=false,hi=-1}={}){
 const top=max??Math.max(...items.map(d=>d.v),1e-9),L=14,R=14,T=fs*1.6,B=fs*1.9,n=items.length,bw=(W-L-R)/n;
 let g=line(ox+L,oy+H-B,ox+W-R,oy+H-B,{stroke:AX});
 items.forEach((d,i)=>{const h=Math.max(0,d.v)/top*(H-T-B),x=ox+L+i*bw+bw*gap/2,w=bw*(1-gap),c=d.col||col;
  g+=`<rect x="${x.toFixed(1)}" y="${(oy+H-B-h).toFixed(1)}" width="${w.toFixed(1)}" height="${Math.max(h,1.5).toFixed(1)}" rx="4" fill="${c}"${hi>=0&&i!==hi?' opacity=".35"':''}/>`;
  g+=label(x+w/2,oy+H-B-h-8,d.note??num(d.v,dec),{size:fs,anchor:'middle',fill:COL.ink,weight:600});
  g+=label(x+w/2,oy+H-B+fs*1.25,esc(d.name),{size:fs,anchor:'middle'});});
 return bare?g:fig(W,H,g,alt);
}

/** A grid of numbers tinted by value (attention weights, a similarity matrix). rows/cols name the axes. */
export function grid(M,{rows=[],cols=[],cw=92,ch=56,fs=21,color=COL.guess,max=null,dec=2,L=null,T=null,alt='A grid of numbers',mark=null,bare=false,ox=0,oy=0,fmt=null}={}){
 const top=max??Math.max(...M.flat().map(Math.abs),1e-9),left=L??(rows.length?Math.max(...rows.map(r=>String(r).length))*fs*.56+18:0),head=T??(cols.length?fs*1.7:0);
 let g='';
 cols.forEach((c,j)=>{g+=label(ox+left+j*cw+cw/2,oy+head-fs*.55,esc(c),{size:fs,anchor:'middle'});});
 M.forEach((row,i)=>{if(rows[i]!==undefined)g+=label(ox+left-12,oy+head+i*ch+ch/2+fs*.36,esc(rows[i]),{size:fs,anchor:'end'});
  row.forEach((v,j)=>{const a=Math.min(.82,.06+.76*Math.abs(v)/top),on=mark&&mark(i,j);
   g+=`<rect x="${ox+left+j*cw+2}" y="${oy+head+i*ch+2}" width="${cw-4}" height="${ch-4}" rx="6" fill="${color}" fill-opacity="${a.toFixed(2)}" stroke="${on?COL.ink:'none'}" stroke-width="${on?3:0}"/>`;
   g+=label(ox+left+j*cw+cw/2,oy+head+i*ch+ch/2+fs*.36,fmt?fmt(v,i,j):num(v,dec),{size:fs,anchor:'middle',fill:a>.5?'#fff':COL.ink,weight:on?700:''});});});
 const W=left+(M[0]?.length||0)*cw,H=head+M.length*ch;
 return bare?{g,W,H}:fig(W,H,g,alt);
}

// ---------------------------------------------------------------------------------------- live figures
/** Live figures register here by name; a slide places one with widget(name, alt). Each takes (el, ctx, S) and must
 draw a sensible static state when ctx.printing is true. */
const registry={};
export const register=(name,fn)=>{if(registry[name])throw new Error('Widget registered twice: '+name);registry[name]=fn;};
export function mount(stage,ctx,S){
 for(const el of stage.querySelectorAll('.widget[data-widget]')){const f=registry[el.dataset.widget];if(!f)throw new Error('Unknown widget '+el.dataset.widget);f(el,ctx,S);}
}
/** A labelled range input for a live figure. Returns its HTML; read it with el.querySelector(`[data-k="${k}"]`). */
export const slider=(k,text,min,max,step,value)=>`<label class="w-slider"><span>${text}</span><input type="range" data-k="${k}" min="${min}" max="${max}" step="${step}" value="${value}"><output data-o="${k}"></output></label>`;
