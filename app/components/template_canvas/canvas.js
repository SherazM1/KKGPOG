/* Temporary pointer feedback only; every committed scene comes from TemplateDraft. */
(() => {
  const host = document.getElementById('canvas'), status = document.getElementById('status');
  const client = Math.random().toString(36).slice(2); let sequence = 0;
  let args, svg, selected, gesture, pending, handles;
  const post = (type, extra = {}) => parent.postMessage({isStreamlitMessage:true, type, ...extra}, '*');
  const rect = node => Object.fromEntries(['x','y','width','height'].map(k => [k, Number(node.getAttribute(k))]));
  const point = event => new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  function highlight(id) {
    selected = id;
    svg.querySelectorAll('[data-slot-id]').forEach(n => n.classList.toggle('selected', n.dataset.slotId === id));
    if (handles) handles.remove();
    handles = document.createElementNS(svg.namespaceURI, 'g'); svg.append(handles);
    const node = [...svg.querySelectorAll('[data-slot-id]')].find(n => n.dataset.slotId === id);
    if (!node) {status.textContent = 'Click a box to select it. Drag to move; use handles to resize.'; return;}
    const box = args.boxes[id], r = rect(node);
    status.textContent = `Box ${box.number} · Section: ${box.sections.join(', ')}${pending ? ' · Updating…' : ''}`;
    for (const [direction, x, y] of [['nw',0,0],['n',.5,0],['ne',1,0],['e',1,.5],['se',1,1],['s',.5,1],['sw',0,1],['w',0,.5]]) {
      const h = document.createElementNS(svg.namespaceURI, 'rect');
      Object.entries({x:r.x+x*r.width-5,y:r.y+y*r.height-5,width:10,height:10,class:'handle','data-handle':direction}).forEach(([k,v]) => h.setAttribute(k,v));
      h.style.cursor = `${direction}-resize`; handles.append(h);
    }
  }
  function emit(kind, id, rectangle) {
    pending = `${client}:${++sequence}`;
    post('streamlit:setComponentValue', {dataType:'json',value:{event_id:pending,kind,slot_id:id,
      template_id:args.template_id,facing_id:args.facing_id,revision:args.revision,...(rectangle ? {rect:rectangle} : {})}});
    highlight(id);
  }
  function paint(node, r) {
    Object.entries(r).forEach(([k,v]) => node.setAttribute(k,v));
    const label = node.nextElementSibling;
    if (label && label.tagName.toLowerCase() === 'text') {label.setAttribute('x',r.x+r.width/2);label.setAttribute('y',r.y+r.height/2);}
    highlight(node.dataset.slotId);
  }
  function down(event) {
    if (pending || gesture || event.button !== 0) return;
    const handle = event.target.closest('[data-handle]');
    const node = handle ? [...svg.querySelectorAll('[data-slot-id]')].find(n => n.dataset.slotId === selected) : event.target.closest('[data-slot-id]');
    if (!node) return;
    event.preventDefault(); const p = point(event);
    gesture = {node, id:node.dataset.slotId, start:p, original:rect(node), direction:handle?.dataset.handle, pointer:event.pointerId};
    highlight(gesture.id); svg.setPointerCapture(event.pointerId);
  }
  function move(event) {
    if (!gesture || gesture.pointer !== event.pointerId) return;
    const p = point(event), dx = p.x-gesture.start.x, dy = p.y-gesture.start.y;
    const r = {...gesture.original}, b = rect(svg.querySelector('[data-kind="facing"]')), d = gesture.direction;
    if (!d) {r.x=clamp(r.x+dx,b.x,b.x+Math.max(0,b.width-r.width));r.y=clamp(r.y+dy,b.y,b.y+Math.max(0,b.height-r.height));}
    else {
      const min = Math.min(4, b.width/100, b.height/100);
      if (d.includes('e')) r.width=Math.max(min,r.width+dx);
      if (d.includes('s')) r.height=Math.max(min,r.height+dy);
      if (d.includes('w')) {r.x=Math.min(r.x+dx,r.x+r.width-min);r.width=gesture.original.x+gesture.original.width-r.x;}
      if (d.includes('n')) {r.y=Math.min(r.y+dy,r.y+r.height-min);r.height=gesture.original.y+gesture.original.height-r.y;}
    }
    paint(gesture.node,r);
  }
  function finish(event, cancel=false) {
    if (!gesture || gesture.pointer !== event.pointerId) return;
    const g=gesture; gesture=null;
    if (cancel) {paint(g.node,g.original);return;}
    const r=rect(g.node), changed=Object.keys(r).some(k=>Math.abs(r[k]-g.original[k])>0.001);
    emit(changed ? (g.direction ? 'resize' : 'drag') : 'select',g.id,changed ? r : undefined);
  }
  window.addEventListener('message', event => {
    if (event.source !== parent || event.data.type !== 'streamlit:render') return;
    const next=event.data.args;
    if (pending && next.ack_id !== pending) return;
    pending=null;gesture=null;args=next;
    const parsed=new DOMParser().parseFromString(args.svg,'image/svg+xml');
    host.replaceChildren(document.importNode(parsed.documentElement,true));svg=host.querySelector('svg');handles=null;
    svg.querySelectorAll('[data-slot-id]').forEach(node => {
      node.setAttribute('tabindex','0');node.setAttribute('role','button');
      node.setAttribute('aria-label',`Box ${args.boxes[node.dataset.slotId].number}; ${args.boxes[node.dataset.slotId].sections.join(', ')}`);
      node.addEventListener('keydown', e=> {if (!pending && ['Enter',' '].includes(e.key)){e.preventDefault();emit('select',node.dataset.slotId);}});
    });
    svg.addEventListener('pointerdown',down);svg.addEventListener('pointermove',move);
    svg.addEventListener('pointerup',e=>finish(e));svg.addEventListener('pointercancel',e=>finish(e,true));
    highlight(args.selected_box_id);
    requestAnimationFrame(()=>post('streamlit:setFrameHeight',{height:Math.ceil(document.body.getBoundingClientRect().height)+4}));
  });
  new ResizeObserver(()=>post('streamlit:setFrameHeight',{height:Math.ceil(document.body.getBoundingClientRect().height)+4})).observe(host);
  post('streamlit:componentReady',{apiVersion:1});
})();
