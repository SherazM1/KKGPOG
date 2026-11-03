/* Pointer feedback only. Every committed scene comes from TemplateDraft. */
(() => {
  const host=document.getElementById('canvas'), status=document.getElementById('status');
  const mode=document.getElementById('selection-mode');
  const client=Math.random().toString(36).slice(2);let sequence=0;
  let args,svg,gesture,pending,lasso,selectMode=false,selected=new Set();
  const post=(type,extra={})=>parent.postMessage({isStreamlitMessage:true,type,...extra},'*');
  const rect=n=>Object.fromEntries(['x','y','width','height'].map(k=>[k,Number(n.getAttribute(k))]));
  const point=e=>new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());
  const nodes=()=>[...svg.querySelectorAll('[data-slot-id]')];
  const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
  function highlight() {
    nodes().forEach(n=>{const active=selected.has(n.dataset.slotId);n.classList.toggle('selected',active);n.setAttribute('aria-pressed',String(active));});
    svg.querySelectorAll('[data-region-id]').forEach(n=>n.classList.toggle('active-section',n.dataset.regionId===args.active_section_id));
    const ids=[...selected];
    if(ids.length===1){const b=args.boxes[ids[0]];status.textContent=`Box ${b.number} ? Section: ${b.sections.join(', ')}`;}
    else status.textContent=ids.length?`${ids.length} boxes selected. Drag a selected box to move the group.`:'Drag empty space to select boxes, or use Select boxes. Box size is fixed.';
    if(pending)status.textContent+=' ? Updating?';
    mode.disabled=!!pending;
  }
  mode.addEventListener('click',()=>{selectMode=!selectMode;mode.setAttribute('aria-pressed',String(selectMode));mode.textContent=selectMode?'Move boxes':'Select boxes';});
  function emit(kind,extra={}) {
    pending=`${client}:${++sequence}`;
    post('streamlit:setComponentValue',{dataType:'json',value:{event_id:pending,kind,
      template_id:args.template_id,facing_id:args.facing_id,revision:args.revision,...extra}});
    highlight();
  }
  function paint(node,r) {
    Object.entries(r).forEach(([k,v])=>node.setAttribute(k,v));
    const label=node.nextElementSibling;
    if(label?.tagName.toLowerCase()==='text'){label.setAttribute('x',r.x+r.width/2);label.setAttribute('y',r.y+r.height/2);}
  }
  function down(e) {
    if(pending||gesture||e.button!==0)return;
    const node=e.target.closest('[data-slot-id]'),p=point(e);
    e.preventDefault();
    if(!node||selectMode||e.shiftKey){
      gesture={kind:'lasso',start:p,pointer:e.pointerId,before:new Set(selected),base:e.shiftKey?new Set(selected):new Set()};
      lasso=document.createElementNS(svg.namespaceURI,'rect');lasso.setAttribute('class','selection-rectangle');svg.append(lasso);
    }else{
      if(e.ctrlKey||e.metaKey){
        if(selected.has(node.dataset.slotId))selected.delete(node.dataset.slotId);else selected.add(node.dataset.slotId);
        emit('select_many',{slot_ids:[...selected]});return;
      }
      if(!selected.has(node.dataset.slotId))selected=new Set([node.dataset.slotId]);
      gesture={kind:'move',start:p,pointer:e.pointerId,before:new Set(selected),originals:nodes().filter(n=>selected.has(n.dataset.slotId)).map(n=>({node:n,r:rect(n)})),dx:0,dy:0};
    }
    highlight();svg.setPointerCapture(e.pointerId);
  }
  function move(e) {
    if(!gesture||gesture.pointer!==e.pointerId)return;
    const viewport=host.getBoundingClientRect();
    if(e.clientY>viewport.bottom-20)host.scrollTop+=18;else if(e.clientY<viewport.top+20)host.scrollTop-=18;
    if(e.clientX>viewport.right-20)host.scrollLeft+=18;else if(e.clientX<viewport.left+20)host.scrollLeft-=18;
    const p=point(e),g=gesture;
    if(g.kind==='lasso'){
      const r={x:Math.min(p.x,g.start.x),y:Math.min(p.y,g.start.y),width:Math.abs(p.x-g.start.x),height:Math.abs(p.y-g.start.y)};
      Object.entries(r).forEach(([k,v])=>lasso.setAttribute(k,v));selected=new Set(g.base);
      nodes().forEach(n=>{const b=rect(n);if(b.x>=r.x&&b.y>=r.y&&b.x+b.width<=r.x+r.width&&b.y+b.height<=r.y+r.height)selected.add(n.dataset.slotId);});
    }else{
      const b=rect(svg.querySelector('[data-kind="facing"]'));
      const x=Math.min(...g.originals.map(o=>o.r.x)),y=Math.min(...g.originals.map(o=>o.r.y));
      const right=Math.max(...g.originals.map(o=>o.r.x+o.r.width)),bottom=Math.max(...g.originals.map(o=>o.r.y+o.r.height));
      g.dx=clamp(p.x-g.start.x,b.x-x,b.x+b.width-right);g.dy=clamp(p.y-g.start.y,b.y-y,b.y+b.height-bottom);
      g.originals.forEach(o=>paint(o.node,{...o.r,x:o.r.x+g.dx,y:o.r.y+g.dy}));
    }
    highlight();
  }
  function finish(e,cancel=false) {
    if(!gesture||gesture.pointer!==e.pointerId)return;
    const g=gesture;gesture=null;if(lasso){lasso.remove();lasso=null;}
    if(cancel){selected=g.before;if(g.originals)g.originals.forEach(o=>paint(o.node,o.r));highlight();return;}
    if(g.kind==='move'&&(Math.abs(g.dx)>0.001||Math.abs(g.dy)>0.001))emit('move_many',{slot_ids:[...selected],delta:{x:g.dx,y:g.dy}});
    else emit('select_many',{slot_ids:[...selected]});
  }
  window.addEventListener('message',e=>{
    if(e.source!==parent||e.data?.type!=='streamlit:render')return;
    const next=e.data.args;if(pending&&next.ack_id!==pending)return;
    const scroll={x:host.scrollLeft,y:host.scrollTop};pending=null;gesture=null;args=next;
    const parsed=new DOMParser().parseFromString(args.svg,'image/svg+xml');
    host.replaceChildren(document.importNode(parsed.documentElement,true));svg=host.querySelector('svg');
    svg.style.minWidth=Math.max(600,Number(svg.getAttribute('width'))*.75)+'px';
    selected=new Set((args.selected_box_ids||[]).filter(id=>args.boxes[id]));
    nodes().forEach(n=>{
      const b=args.boxes[n.dataset.slotId];n.setAttribute('tabindex','0');n.setAttribute('role','button');
      n.setAttribute('aria-label',`Box ${b.number}; ${b.sections.join(', ')}`);
      n.addEventListener('keydown',event=>{if(!pending&&['Enter',' '].includes(event.key)){event.preventDefault();selected=new Set([n.dataset.slotId]);emit('select_many',{slot_ids:[...selected]});}});
    });
    svg.addEventListener('pointerdown',down);svg.addEventListener('pointermove',move);
    svg.addEventListener('pointerup',e=>finish(e));svg.addEventListener('pointercancel',e=>finish(e,true));
    svg.addEventListener('keydown',e=>{if(e.key==='Escape'&&gesture)finish({pointerId:gesture.pointer},true);});
    highlight();requestAnimationFrame(()=>{host.scrollTop=scroll.y;host.scrollLeft=scroll.x;post('streamlit:setFrameHeight',{height:Math.ceil(document.body.getBoundingClientRect().height)+4});});
  });
  new ResizeObserver(()=>post('streamlit:setFrameHeight',{height:Math.ceil(document.body.getBoundingClientRect().height)+4})).observe(host);
  post('streamlit:componentReady',{apiVersion:1});
})();
