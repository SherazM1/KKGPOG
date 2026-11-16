/* Pointer feedback only. Every committed scene comes from TemplateDraft. */
(() => {
  const host=document.getElementById('canvas'), status=document.getElementById('status');
  const mode=document.getElementById('selection-mode'), deleteBoxes=document.getElementById('delete-boxes');
  const sidePreviews=document.getElementById('side-previews');
  const client=Math.random().toString(36).slice(2);let sequence=0;
  let args,svg,gesture,pending,lasso,selectMode=false,selected=new Set();
  let frameRequested=false,lastFrameHeight=0;
  const post=(type,extra={})=>parent.postMessage({isStreamlitMessage:true,type,...extra},'*');
  const rect=n=>Object.fromEntries(['x','y','width','height'].map(k=>[k,Number(n.getAttribute(k))]));
  const point=e=>new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());
  const nodes=()=>svg?[...svg.querySelectorAll('[data-slot-id]')]:[];
  const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
  function requestFrameHeight() {
    if(frameRequested)return;
    frameRequested=true;
    requestAnimationFrame(()=>{
      frameRequested=false;
      const height=Math.ceil(document.body.getBoundingClientRect().height)+4;
      if(height!==lastFrameHeight){lastFrameHeight=height;post('streamlit:setFrameHeight',{height});}
    });
  }
  function sceneHeight(height) {
    const width=svg.viewBox.baseVal.width;
    svg.setAttribute('height',height);
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    requestFrameHeight();
  }
  function growCanvas(bottom) {
    if(svg.dataset.growDown!=='true')return;
    const workspace=svg.querySelector('[data-kind="workspace"]');
    const bounds=rect(workspace),required=bottom+Number(svg.dataset.growPadding);
    if(required>bounds.y+bounds.height){
      workspace.setAttribute('height',required-bounds.y);
      sceneHeight(required+24);
    }
  }
  function renderSidePreviews() {
    sidePreviews.replaceChildren();
    for(const side of args.side_previews||[]){
      const button=document.createElement('button');
      button.type='button';button.className='side-thumbnail';button.dataset.facingId=side.id;
      button.setAttribute('aria-label',`Edit ${side.name}`);
      button.setAttribute('aria-pressed',String(side.id===args.facing_id));
      const preview=document.createElement('span');preview.className='thumbnail-image';
      if(side.svg){
        const parsed=new DOMParser().parseFromString(side.svg,'image/svg+xml');
        const image=document.importNode(parsed.documentElement,true);
        image.querySelectorAll('[data-slot-id], [data-region-id], [data-region-label], [data-kind]').forEach(node=>{
          for(const attr of ['data-slot-id','data-region-id','data-region-label','data-kind'])node.removeAttribute(attr);
        });
        image.setAttribute('aria-hidden','true');preview.append(image);
      }else preview.textContent='Check settings';
      const name=document.createElement('span');name.className='thumbnail-name';name.textContent=side.name;
      const detail=document.createElement('span');detail.className='thumbnail-detail';
      detail.textContent=`${side.box_count} ${side.box_count===1?'box':'boxes'}${side.id===args.facing_id?' - Editing':''}`;
      button.append(preview,name,detail);
      button.addEventListener('click',()=>{
        if(pending||gesture||side.id===args.facing_id)return;
        emit('activate_side',{target_facing_id:side.id});
      });
      sidePreviews.append(button);
    }
  }
  function highlight() {
    nodes().forEach(n=>{const active=selected.has(n.dataset.slotId);n.classList.toggle('selected',active);n.setAttribute('aria-pressed',String(active));});
    svg?.querySelectorAll('[data-region-id]').forEach(n=>n.classList.toggle('active-section',n.dataset.regionId===args.active_section_id));
    const ids=[...selected];
    if(ids.length===1){const b=args.boxes[ids[0]];status.textContent=`Box ${b.number} ? Section: ${b.sections.join(', ')}`;}
    else status.textContent=ids.length?`${ids.length} boxes selected. Drag a selected box to move the group.`:svg?'Drag empty space to select boxes, or use Select boxes. Box size is fixed.':'Choose a Side preview or repair this Side in the controls.';
    if(pending)status.textContent+=' ? Updating?';
    mode.disabled=!!pending||!svg;
    deleteBoxes.disabled=!!pending||!!gesture||!selected.size||!svg;
    sidePreviews.querySelectorAll('button').forEach(button=>{button.disabled=!!pending||!!gesture;});
  }
  mode.addEventListener('click',()=>{selectMode=!selectMode;mode.setAttribute('aria-pressed',String(selectMode));mode.textContent=selectMode?'Move boxes':'Select boxes';svg?.classList.toggle('selection-mode',selectMode);});
  deleteBoxes.addEventListener('click',()=>{
    if(!args||pending||gesture||!selected.size)return;
    emit('delete_many',{slot_ids:[...selected]});
  });
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
      gesture={kind:'move',start:p,pointer:e.pointerId,before:new Set(selected),originals:nodes().filter(n=>selected.has(n.dataset.slotId)).map(n=>({node:n,r:rect(n)})),dx:0,dy:0,
        height:svg.getAttribute('height'),workspaceHeight:svg.querySelector('[data-kind="workspace"]').getAttribute('height')};
    }
    highlight();svg.setPointerCapture(e.pointerId);
  }
  function move(e) {
    if(!gesture||gesture.pointer!==e.pointerId)return;
    const p=point(e),g=gesture;
    if(g.kind==='lasso'){
      const r={x:Math.min(p.x,g.start.x),y:Math.min(p.y,g.start.y),width:Math.abs(p.x-g.start.x),height:Math.abs(p.y-g.start.y)};
      Object.entries(r).forEach(([k,v])=>lasso.setAttribute(k,v));selected=new Set(g.base);
      nodes().forEach(n=>{const b=rect(n);if(b.x>=r.x&&b.y>=r.y&&b.x+b.width<=r.x+r.width&&b.y+b.height<=r.y+r.height)selected.add(n.dataset.slotId);});
    }else{
      const b=rect(svg.querySelector('[data-kind="facing"]'));
      const x=Math.min(...g.originals.map(o=>o.r.x)),y=Math.min(...g.originals.map(o=>o.r.y));
      const right=Math.max(...g.originals.map(o=>o.r.x+o.r.width)),bottom=Math.max(...g.originals.map(o=>o.r.y+o.r.height));
      g.dx=clamp(p.x-g.start.x,b.x-x,b.x+b.width-right);
      g.dy=svg.dataset.growDown==='true'?Math.max(b.y-y,p.y-g.start.y):clamp(p.y-g.start.y,b.y-y,b.y+b.height-bottom);
      g.originals.forEach(o=>paint(o.node,{...o.r,x:o.r.x+g.dx,y:o.r.y+g.dy}));
      growCanvas(bottom+g.dy);
    }
    highlight();
  }
  function finish(e,cancel=false) {
    if(!gesture||gesture.pointer!==e.pointerId)return;
    const g=gesture;gesture=null;if(lasso){lasso.remove();lasso=null;}
    if(cancel){
      selected=g.before;
      if(g.originals){
        g.originals.forEach(o=>paint(o.node,o.r));
        svg.querySelector('[data-kind="workspace"]').setAttribute('height',g.workspaceHeight);
        sceneHeight(g.height);
      }
      highlight();return;
    }
    if(g.kind==='move'&&(Math.abs(g.dx)>0.001||Math.abs(g.dy)>0.001))emit('move_many',{slot_ids:[...selected],delta:{x:g.dx,y:g.dy}});
    else emit('select_many',{slot_ids:[...selected]});
  }
  window.addEventListener('message',e=>{
    if(e.source!==parent||e.data?.type!=='streamlit:render')return;
    const next=e.data.args;if(pending&&next.ack_id!==pending)return;
    pending=null;gesture=null;args=next;
    renderSidePreviews();
    if(!args.svg){
      const empty=document.createElement('div');empty.className='canvas-empty';
      empty.textContent='Fix this Side\'s settings to restore its large preview. Other Sides are still available above.';
      host.replaceChildren(empty);svg=null;selected=new Set();highlight();requestFrameHeight();return;
    }
    const parsed=new DOMParser().parseFromString(args.svg,'image/svg+xml');
    host.replaceChildren(document.importNode(parsed.documentElement,true));svg=host.querySelector('svg');
    svg.classList.toggle('selection-mode',selectMode);
    selected=new Set((args.selected_box_ids||[]).filter(id=>args.boxes[id]));
    nodes().forEach(n=>{
      const b=args.boxes[n.dataset.slotId];n.setAttribute('tabindex','0');n.setAttribute('role','button');
      n.setAttribute('aria-label',`Box ${b.number}; ${b.sections.join(', ')}`);
      n.addEventListener('keydown',event=>{if(!pending&&['Enter',' '].includes(event.key)){event.preventDefault();selected=new Set([n.dataset.slotId]);emit('select_many',{slot_ids:[...selected]});}});
    });
    svg.addEventListener('pointerdown',down);svg.addEventListener('pointermove',move);
    svg.addEventListener('pointerup',e=>finish(e));svg.addEventListener('pointercancel',e=>finish(e,true));
    svg.addEventListener('keydown',e=>{if(e.key==='Escape'&&gesture)finish({pointerId:gesture.pointer},true);});
    highlight();requestFrameHeight();
  });
  new ResizeObserver(requestFrameHeight).observe(document.body);
  post('streamlit:componentReady',{apiVersion:1});
})();
