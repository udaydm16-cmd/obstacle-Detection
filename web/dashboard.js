const get = id => document.getElementById(id);
let revision = -1, lastEvent = 0, loaded = false, busy = false;
let frozen = false, lanePoints = [], calibrating = false;
let currentRun = null, uploadUrl = null, overlay = false;
let videoUnavailable = false;
const video = get('video');
const replayModes = ['traffic','pothole','blockage','bump','upload'];
const order = ['far-left','far-right','near-right','near-left'];
function renderLanes() {
  get('save-lanes').disabled = lanePoints.length !== 4;
  get('reset-lanes').disabled = !lanePoints.length;
  get('lane-guide').textContent = !frozen
    ? 'Aim the camera along two lanes. Freeze a clear frame to mark them.'
    : lanePoints.length < 4
      ? 'Click the '+order[lanePoints.length]+' corner. Far is where traffic exits the monitored section.'
      : 'Check the outline and center divider, then save the lanes.';
  const overlay = get('lane-overlay');
  overlay.replaceChildren();
  function shape(tag, attrs) {
    const element = document.createElementNS('http://www.w3.org/2000/svg',tag);
    for (const [key,value] of Object.entries(attrs)) element.setAttribute(key,value);
    overlay.append(element);
  }
  if (lanePoints.length > 1) shape(lanePoints.length===4?'polygon':'polyline',{
    points:lanePoints.map(point=>point.join(',')).join(' '), fill:lanePoints.length===4?'#ff965125':'none',
    stroke:'#ff995c','stroke-width':2});
  if (lanePoints.length===4) {
    const [a,b,c,d] = lanePoints;
    shape('line',{x1:(a[0]+b[0])/2,y1:(a[1]+b[1])/2,x2:(c[0]+d[0])/2,y2:(c[1]+d[1])/2,
                  stroke:'#fff','stroke-width':2,'stroke-dasharray':'5 3'});
  }
  lanePoints.forEach(([x,y])=>shape('circle',{cx:x,cy:y,r:4,fill:'#ff985a',stroke:'#fff'}));
}
get('freeze').onclick = () => {
  if (get('preview').hidden) return;
  frozen = true;
  get('freeze').disabled = true;
  get('viewer').classList.add('marking');
  renderLanes();
};
get('reset-lanes').onclick = () => { lanePoints=[]; renderLanes(); };
get('viewer').onclick = event => {
  if (!calibrating || !frozen || lanePoints.length===4) return;
  const rect=get('preview').getBoundingClientRect();
  const x=(event.clientX-rect.left)/rect.width, y=(event.clientY-rect.top)/rect.height;
  if (x>=0 && x<=1 && y>=0 && y<=1) {
    lanePoints.push([Math.round(x*319),Math.round(y*239)]);
    renderLanes();
  }
};
get('save-lanes').onclick = () => command('/api/calibration',JSON.stringify(lanePoints.flat()));
async function command(path, body) {
  try {
    busy = true;
    const response = await fetch(path, {method:'POST', headers:{'X-Road-Dashboard':'1'}, body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    return true;
  } catch (error) {
    get('message').textContent = error.message;
  } finally { busy = false; }
}
document.querySelectorAll('[data-source]').forEach(button => button.onclick = () => command('/api/start/'+button.dataset.source));
get('stop').onclick = () => command('/api/stop');
get('upload').onchange = async event => {
  const file = event.target.files[0];
  if (file) {
    if (uploadUrl) URL.revokeObjectURL(uploadUrl);
    uploadUrl = URL.createObjectURL(file);
    await command('/api/upload', file);
  }
  event.target.value = '';
};
get('overlay-toggle').onclick = async () => {
  if (await command('/api/overlay/'+(overlay?'off':'on'))) {
    overlay = !overlay;
    get('overlay-toggle').textContent = overlay ? 'Show smooth video' : 'Show tracking view';
    if (overlay) video.pause();
    else if (!videoUnavailable) video.play().catch(() => {});
  }
};
video.onerror = () => {
  videoUnavailable = true;
  if (!overlay) get('overlay-toggle').click();
};
get('sound').onchange = async event => {
  if (event.target.checked) {
    try { await get('tone').play(); } catch { event.target.checked = false; }
  }
};
function showEvents(events) {
  if (events[0]?.id === lastEvent && loaded) return;
  if (loaded && get('sound').checked && events.some(e => e.id > lastEvent && e.kind === 'blockage'))
    get('tone').play().catch(() => {});
  loaded = true;
  if (!events.length) return;
  lastEvent = events[0].id;
  get('events').replaceChildren(...events.map(event => {
    const card = document.createElement('article'); card.className = 'event';
    const badge = document.createElement('span'); badge.className = 'badge'+(event.kind==='blockage'?' urgent':'');
    badge.textContent = event.kind==='blockage' ? 'Urgent review' : event.kind==='pothole' ? 'Pothole alert' : 'Image suggestion';
    const title = document.createElement('h3'); title.textContent = event.reason;
    const detail = document.createElement('small'); detail.textContent = event.source+' · '+new Date(event.occurred*1000).toLocaleString();
    card.append(badge,title,detail);
    if (event.evidence) {
      const image = document.createElement('img'); image.src = event.evidence; image.alt = 'Saved road evidence'; image.loading = 'lazy';
      card.append(image);
    }
    return card;
  }));
}
async function refresh() {
  try {
    if (busy) return;
    const [state, events] = await Promise.all(['/api/status','/api/events'].map(async url => (await fetch(url)).json()));
    get('board').textContent = state.board;
    get('state').textContent = state.running ? 'Running' : 'Idle';
    get('source').textContent = state.source;
    get('message').textContent = state.message;
    get('frames').textContent = state.frames ? state.frames+' frames read' : '';
    get('passages').textContent = state.passages;
    get('stop').disabled = !state.running;
    const setup = state.running && state.mode==='calibrate';
    if (setup !== calibrating) {
      calibrating = setup;
      frozen = false;
      lanePoints = [];
      get('freeze').disabled = false;
      get('viewer').classList.remove('marking');
      for (const id of ['lane-tools','lane-guide','lane-overlay']) get(id).hidden = !setup;
      renderLanes();
    }
    document.querySelectorAll('[data-source]').forEach(button => button.disabled = state.running);
    get('upload').disabled = state.running;
    document.querySelector('[for=upload]').classList.toggle('disabled',state.running);
    state.counts.forEach(({lane,passages,slowed}) => {
      get('lane'+lane).textContent = state.site ? passages : '—';
      get('slow'+lane).textContent = slowed+' slowdowns';
    });
    get('calibration').textContent = state.site ? 'Only calibrated live traffic contributes to the 72-hour count.' : 'Set live lane boundaries to enable the 72-hour count.';
    const replay = replayModes.includes(state.mode);
    get('overlay-toggle').hidden = !replay || videoUnavailable;
    if (currentRun !== state.run_id) {
      currentRun = state.run_id;
      overlay = false;
      videoUnavailable = false;
      get('overlay-toggle').textContent = 'Show tracking view';
      video.pause();
      video.removeAttribute('src');
      video.load();
      if (replay) {
        const source = state.mode==='upload' ? uploadUrl : '/media/'+state.mode;
        if (source) {
          video.src = source;
          if (state.running) video.play().catch(() => {});
        } else {
          videoUnavailable = true;
          get('overlay-toggle').click();
        }
      }
    }
    const playingView = replay && !overlay && !videoUnavailable;
    video.hidden = !playingView;
    if (playingView) {
      get('preview').hidden = get('placeholder').hidden = true;
      if (state.message==='Stopped' || state.message==='Video ended early') video.pause();
    } else if (state.mode==='camera' || state.mode==='auto') {
      get('preview').hidden=true;
      get('placeholder').hidden=false;
      get('placeholder').textContent=state.running && state.mode==='auto' ? 'Checking camera for lane markings.' : state.running
        ? 'Camera analysis is running. Evidence appears in the alert list when an event is detected.'
        : 'Select a source to see the camera and tracking views.';
    } else if (!frozen && revision !== state.revision) {
      revision = state.revision;
      const image = new Image();
      image.onload = () => { if (frozen || revision !== state.revision || !video.hidden) return; get('preview').src=image.src; get('preview').hidden=false; get('placeholder').hidden=true; };
      image.onerror = () => { if (!video.hidden) return; get('preview').hidden=true; get('placeholder').hidden=false; };
      image.src = '/frame.jpg?v='+revision;
    }
    showEvents(events);
  } catch {
    get('message').textContent = 'Dashboard connection lost.';
  } finally { setTimeout(refresh,650); }
}
refresh();
