// The summary lecture, assembled from its parts in order. Each file in parts/ adds its slides to the one list in kit.js.
// ?part=03 loads that one part on its own, for work on a single part: ?all&part=03.
import {meta,slides,mount as mountWidgets} from './kit.js';
const FILES=['00-opening','01-foundations','02-gradients','03-generalization','04-attention','05-vision','06-generative','07-detection','08-wrap'];
const one=new URLSearchParams(location.search).get('part');
if(one&&!FILES.some(f=>f.startsWith(one+'-')))throw new Error('No part '+one);
for(const f of FILES)if(!one||f.startsWith(one+'-'))await import(`./parts/${f}.js`);
export {meta,slides};
export const mount=(stage,ctx)=>mountWidgets(stage,ctx,{});
