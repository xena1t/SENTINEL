import { loadStudy, sensors } from './data.js';

const $ = id => document.getElementById(id);
const video = $('camera');
const monitor = $('monitor');
const formatTime = time => time.toFixed(2).padStart(5, '0');
const maskLabel = mask => sensors.filter((_, index) => mask[index] === '1').map(sensor => sensor.label).join(' + ') || 'No evidence';
let study;
let state;
let mediaGeneration = 0;
let mediaEvents;
let frameRequest;
let lastRfSource = '';

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function hold() {
  video.pause();
  monitor.pause();
  const clip = study.clip(state.clipId);
  const index = state.windowIndex < 0 ? 0 : state.windowIndex;
  if (clip.mediaKind === 'snapshot') return 0;
  const end = clip.windows[index].end;
  if (video.readyState > 0) video.currentTime = end;
  if (clip.audio && monitor.readyState > 0) monitor.currentTime = end;
  return index;
}

function dispatch(action) {
  if (!study || !state) return;
  const clip = study.clip(state.clipId);
  if (action.type === 'clip') {
    hold();
    const nextClip = study.clip(action.clipId);
    const nextIndex = nextClip.mediaKind === 'snapshot' ? 0 : Math.max(0, study.windowAtTime(nextClip, nextClip.cue?.time ?? nextClip.windows[0].end));
    const changedClip = state.clipId !== nextClip.id;
    state = { ...state, clipId: nextClip.id, windowIndex: nextIndex, mask: nextClip.availableMask, transport: 'held', listening: false };
    if (changedClip) loadMedia(nextClip);
    else hold();
    render();
    announce();
    return;
  }
  if (action.type === 'sensor' || action.type === 'mask') {
    if (action.type === 'sensor' && !study.canToggle(clip.id, action.index)) return;
    const index = hold();
    const bits = state.mask.split('');
    if (action.type === 'sensor') bits[action.index] = bits[action.index] === '1' ? '0' : '1';
    state = { ...state, windowIndex: index, mask: action.type === 'mask' ? action.mask : bits.join(''), transport: 'held' };
  } else if (action.type === 'seek' || action.type === 'step') {
    if (clip.mediaKind !== 'video') return;
    hold();
    const index = Math.max(0, Math.min(clip.windows.length - 1, action.type === 'seek' ? action.index : state.windowIndex + action.direction));
    state = { ...state, windowIndex: index, transport: 'held' };
    hold();
  } else if (action.type === 'hold') {
    state = { ...state, windowIndex: hold(), transport: 'held' };
  } else if (action.type === 'play') {
    if (clip.mediaKind !== 'video') return;
    if (state.transport === 'playing') {
      dispatch({ type: 'hold' });
      return;
    }
    if (video.error) {
      $('announcement').textContent = 'Camera media is unavailable. Select and compare cached windows with the timeline.';
      return;
    }
    if (video.currentTime >= clip.duration - 0.02) {
      video.currentTime = 0;
      if (clip.audio) monitor.currentTime = 0;
      state = { ...state, windowIndex: -1 };
    }
    state = { ...state, transport: 'playing' };
    const generation = mediaGeneration;
    video.play().then(() => {
      if (generation !== mediaGeneration || state.transport !== 'playing') return;
      syncMonitor(true);
      clockFrame();
    }).catch(error => {
      if (generation !== mediaGeneration || error.name === 'AbortError') return;
      dispatch({ type: 'hold' });
      $('announcement').textContent = 'Playback could not start. Try Play recording again.';
    });
  } else if (action.type === 'clock') {
    if (state.transport !== 'playing') return;
    const index = study.windowAtTime(clip, video.currentTime);
    if (index === state.windowIndex) { renderClock(); return; }
    state = { ...state, windowIndex: index };
  } else if (action.type === 'ended') {
    state = { ...state, windowIndex: clip.windows.length - 1, transport: 'held' };
    monitor.pause();
  } else if (action.type === 'listen') {
    if (!clip.audio) return;
    state = { ...state, listening: !state.listening };
    syncMonitor(state.transport === 'playing');
  } else return;
  render();
  if (action.type !== 'clock' && action.type !== 'play' && action.type !== 'ended') announce();
}

function syncMonitor(play) {
  if (!study.clip(state.clipId).audio) { monitor.pause(); return; }
  monitor.muted = !state.listening;
  if (Math.abs(monitor.currentTime - video.currentTime) > 0.12 && monitor.readyState > 0) monitor.currentTime = video.currentTime;
  if (play && state.listening) monitor.play().catch(() => { $('announcement').textContent = 'Audio monitoring is unavailable for this recording.'; });
  else monitor.pause();
}

function loadMedia(clip) {
  mediaGeneration++;
  const generation = mediaGeneration;
  mediaEvents?.abort();
  mediaEvents = new AbortController();
  const options = { signal: mediaEvents.signal };
  cancelAnimationFrame(frameRequest);
  lastRfSource = '';
  $('video-error').hidden = true;
  $('rf-error').hidden = true;
  video.pause();
  monitor.pause();
  if (clip.mediaKind === 'video') {
    video.addEventListener('loadedmetadata', () => {
      if (generation !== mediaGeneration || video.currentSrc !== clip.video) return;
      video.currentTime = clip.windows[Math.max(0, state.windowIndex)].end;
      renderClock();
    }, options);
    video.addEventListener('error', () => {
      if (generation !== mediaGeneration || video.currentSrc !== clip.video) return;
      $('video-error').hidden = false;
      if (state.transport === 'playing') dispatch({ type: 'hold' });
    }, options);
    video.addEventListener('timeupdate', () => {
      if (generation !== mediaGeneration || video.currentSrc !== clip.video) return;
      if (state.transport === 'playing') dispatch({ type: 'clock' });
      syncMonitor(state.transport === 'playing');
    }, options);
    video.addEventListener('ended', () => {
      if (generation === mediaGeneration && video.currentSrc === clip.video) dispatch({ type: 'ended' });
    }, options);
  }
  $('camera-still').addEventListener('error', () => {
    if (generation === mediaGeneration && clip.mediaKind === 'snapshot' && $('camera-still').currentSrc === clip.poster) $('video-error').hidden = false;
  }, options);
  $('rf-image').addEventListener('error', () => {
    if (generation === mediaGeneration && $('rf-image').currentSrc === lastRfSource && lastRfSource) $('rf-error').hidden = false;
  }, options);
  if (clip.audio) monitor.addEventListener('loadedmetadata', () => {
    if (generation === mediaGeneration) monitor.currentTime = clip.windows[Math.max(0, state.windowIndex)].end;
  }, options);
  video.removeAttribute('src');
  monitor.removeAttribute('src');
  video.load();
  monitor.load();
  video.hidden = clip.mediaKind !== 'video';
  $('camera-still').hidden = clip.mediaKind !== 'snapshot';
  $('camera-still').removeAttribute('src');
  if (clip.mediaKind === 'video') { video.poster = clip.poster; video.src = clip.video; }
  else $('camera-still').src = clip.poster;
  if (clip.audio) monitor.src = clip.audio;
  $('rf-image').removeAttribute('src');
  $('rf-image').hidden = !clip.rfFrames.length;
  $('rf-empty').hidden = !!clip.rfFrames.length;
  $('waveform').toggleAttribute('hidden', !clip.audio);
  $('audio-empty').hidden = !!clip.audio;
  monitor.muted = !state.listening;
  renderWaveform(clip);
  renderProvenance(clip);
}

function clockFrame() {
  cancelAnimationFrame(frameRequest);
  if (state.transport !== 'playing') return;
  dispatch({ type: 'clock' });
  frameRequest = requestAnimationFrame(clockFrame);
}

function renderWaveform(clip) {
  $('waveform').hidden = !clip.audio;
  const path = clip.waveform.map((value, index) => {
    const x = index / Math.max(1, clip.waveform.length - 1) * 600;
    const amplitude = value * 31;
    return `M${x.toFixed(2)},${(35 - amplitude).toFixed(2)}V${(35 + amplitude).toFixed(2)}`;
  }).join('');
  $('wave-path').setAttribute('d', path);
}

function renderClock() {
  const clip = study.clip(state.clipId);
  const window = clip.windows[state.windowIndex];
  const time = state.transport === 'playing' ? video.currentTime : window?.end || 0;
  if (clip.mediaKind === 'snapshot') $('camera-clock').textContent = 'RGB STILL';
  else $('camera-clock').replaceChildren(document.createTextNode(`${formatTime(time)} `), element('small', '', `/ ${formatTime(clip.duration)} s`));
  const rf = study.rfForWindow(clip, state.windowIndex);
  if (rf && rf.src !== lastRfSource) {
    lastRfSource = rf.src;
    $('rf-error').hidden = true;
    $('rf-image').src = rf.src;
  } else if (!rf) {
    lastRfSource = '';
    $('rf-image').removeAttribute('src');
  }
  $('rf-image').hidden = !rf;
  $('rf-empty').hidden = !!rf;
  $('rf-empty').textContent = clip.rfFrames.length ? 'Awaiting a completed cached window' : 'RF was not recorded for this example';
  $('rf-time').textContent = clip.mediaKind === 'snapshot' ? '30 ms RF CAPTURE · LAST SOURCE CHUNK' : rf ? `WINDOW FRAME · ${formatTime(rf.time)} s` : 'NO PAIRED RF FRAME';
  $('wave-cursor').setAttribute('x1', time / clip.duration * 600);
  $('wave-cursor').setAttribute('x2', time / clip.duration * 600);
}

function renderChart(clip, comparison) {
  const rows = study.scores(clip.id, state.mask);
  for (const id of ['fusion', 'qwen']) {
    const path = clip.mediaKind === 'snapshot' ? '' : rows.map((row, index) => {
      const x = (row.end / clip.duration * 800).toFixed(2);
      const y = (84 - row[id] * 72).toFixed(2);
      return index ? `H${x}V${y}` : `M${x},${y}`;
    }).join('');
    $(`${id}-trace`).setAttribute('d', path);
    const point = $(`${id}-point`);
    point.hidden = comparison.kind !== 'cached';
    point.style.display = comparison.kind === 'cached' ? '' : 'none';
    if (comparison.kind === 'cached') {
      point.setAttribute('cx', clip.mediaKind === 'snapshot' ? 400 : comparison.window.end / clip.duration * 800);
      point.setAttribute('cy', 84 - comparison.selected[id] * 72);
    }
  }
  const window = clip.mediaKind === 'video' ? comparison.window : null;
  $('chart-cursor').style.display = clip.mediaKind === 'snapshot' ? 'none' : '';
  $('chart-window').setAttribute('x', window ? window.start / clip.duration * 800 : 0);
  $('chart-window').setAttribute('width', window ? (window.end - window.start) / clip.duration * 800 : 0);
  for (const attribute of ['x1', 'x2']) $('chart-cursor').setAttribute(attribute, window ? window.end / clip.duration * 800 : 0);
  $('wave-window').setAttribute('x', window ? window.start / clip.duration * 600 : 0);
  $('wave-window').setAttribute('width', window ? (window.end - window.start) / clip.duration * 600 : 0);
}

function renderPrediction(id, comparison) {
  const card = $(`${id}-card`);
  if (comparison.kind !== 'cached') {
    card.dataset.outcome = comparison.kind;
    $(`${id}-score`).textContent = '—';
    $(`${id}-score`).removeAttribute('title');
    $(`${id}-bar`).style.inlineSize = '0%';
    $(`${id}-verdict`).textContent = comparison.kind === 'no-evidence' ? 'No evidence selected' : comparison.kind === 'unsupported' ? 'Mask not in cache' : 'Awaiting complete window';
    $(`${id}-delta`).textContent = comparison.kind === 'no-evidence' ? 'Choose an input to inspect its cached output.' : comparison.kind === 'unsupported' ? 'No released output exists for this sensor selection.' : 'A prediction appears once a full window ends.';
    return;
  }
  const score = comparison.selected[id];
  const predicted = score >= study.threshold ? 1 : 0;
  const actual = comparison.window.label;
  const outcome = predicted && !actual ? 'false-positive' : !predicted && actual ? 'false-negative' : 'correct';
  card.dataset.outcome = outcome;
  $(`${id}-score`).textContent = score.toFixed(3);
  $(`${id}-score`).title = `Stored value ${score}`;
  $(`${id}-bar`).style.inlineSize = `${score * 100}%`;
  $(`${id}-verdict`).textContent = `${predicted ? 'Drone detected' : 'Background'}${outcome === 'false-positive' ? ' · False alarm' : outcome === 'false-negative' ? ' · Missed drone' : ' · Correct'}`;
  const delta = score - comparison.reference[id];
  const clip = study.clip(state.clipId);
  $(`${id}-delta`).textContent = clip.availableMask === '100' ? 'Camera-only released evaluation' : state.mask === comparison.referenceMask ? `All available inputs · ${maskLabel(comparison.referenceMask)}` : `${delta >= 0 ? '+' : ''}${delta.toFixed(3)} from all available inputs · same observation`;
}

function render() {
  const clip = study.clip(state.clipId);
  const comparison = study.compare(state.clipId, state.windowIndex, state.mask);
  $('clip-title').textContent = clip.title;
  $('dataset-label').textContent = `${clip.dataset} / ${clip.mediaKind === 'snapshot' ? 'SAVED OBSERVATION' : 'RECORDED VIDEO'}`;
  $('source-support').textContent = clip.mediaKind === 'snapshot' ? 'RGB STILL + 30 ms RF CAPTURE. The RF duration does not describe the still image.' : clip.availableMask === '100' ? 'CAMERA ONLY. This video has no paired RF or audio, so sensor dropout is not demonstrated.' : 'CAMERA + RF + AUDIO. Compare the same completed window with an input excluded.';
  $('camera-media-label').textContent = clip.mediaKind === 'snapshot' ? 'SAVED RGB STILL' : `${clip.dataset.toUpperCase()} VIDEO`;
  $('clip-select').value = clip.id;
  const actual = comparison.window?.label ?? clip.label;
  $('camera-truth').textContent = actual ? 'DRONE PRESENT' : 'BACKGROUND';
  $('truth-label').textContent = `Actual · ${actual ? 'Drone' : 'Background'}`;
  $('mask-code').replaceChildren(document.createTextNode('MASK '), element('b', '', state.mask));
  sensors.forEach((sensor, index) => {
    const enabled = state.mask[index] === '1';
    const available = study.canToggle(clip.id, index);
    const button = document.querySelector(`[data-sensor-switch="${sensor.id}"]`);
    button.setAttribute('aria-checked', String(enabled));
    button.disabled = !available;
    button.title = available ? `${sensor.label} evidence (${sensor.key}). Compare the same observation.` : `${sensor.label} was not recorded for this example.`;
    button.querySelector('small').textContent = available ? enabled ? 'INCLUDED' : 'EXCLUDED' : 'NOT RECORDED';
    document.querySelector(`[data-sensor-view="${sensor.id}"]`).dataset.available = available;
    document.querySelector(`[data-sensor-view="${sensor.id}"]`).dataset.enabled = enabled;
    document.querySelector(`[data-sensor-label="${sensor.id}"]`).textContent = !available ? 'NOT RECORDED' : enabled ? 'INCLUDED' : 'EXCLUDED';
  });
  document.querySelectorAll('[data-route], [data-node]').forEach(node => {
    const route = node.dataset.route || node.dataset.node;
    const enabled = route === 'any' ? state.mask !== '000' : state.mask[sensors.findIndex(sensor => sensor.id === route)] === '1';
    node.dataset.enabled = enabled;
  });
  const snapshot = clip.mediaKind === 'snapshot';
  const playing = state.transport === 'playing';
  for (const id of ['play', 'hold', 'previous', 'next']) { $(id).hidden = snapshot; $(id).disabled = snapshot; }
  $('seek').closest('label').hidden = snapshot;
  $('seek').disabled = snapshot;
  $('timeline-start').hidden = snapshot;
  $('timeline-duration').hidden = snapshot;
  $('chart-heading').textContent = snapshot ? 'ONE SAVED OBSERVATION' : 'DRONE SCORE OVER TIME';
  $('play').replaceChildren(document.createTextNode(playing ? 'Ⅱ ' : '▶ '), element('span', '', playing ? 'Pause recording' : 'Play recording'));
  $('play').setAttribute('aria-label', playing ? 'Pause recording' : 'Play recording');
  $('transport-state').textContent = snapshot ? 'SAVED OBSERVATION' : playing ? 'RECORDED PLAYBACK' : 'HELD WINDOW';
  $('window-time').textContent = snapshot ? 'RGB still + 30 ms RF capture' : comparison.window ? `${formatTime(comparison.window.start)} to ${formatTime(comparison.window.end)} s` : 'Awaiting complete window';
  $('listen').disabled = !clip.audio;
  $('audio-meta').textContent = clip.audio ? `RECORDED WAVEFORM · ${clip.duration} s` : 'NO PAIRED SOURCE AUDIO';
  $('listen').setAttribute('aria-pressed', String(state.listening));
  $('listen').replaceChildren(document.createTextNode('◖ Listen '), element('span', '', state.listening ? 'ON' : 'OFF'));
  $('listen').title = 'Monitor the source audio. This does not change audio evidence supplied to either model.';
  $('seek').max = clip.windows.length - 1;
  $('seek').value = Math.max(0, state.windowIndex);
  $('seek').setAttribute('aria-valuetext', comparison.window ? `Window ${formatTime(comparison.window.start)} to ${formatTime(comparison.window.end)} seconds` : 'Awaiting first complete window');
  $('previous').disabled = snapshot || state.windowIndex <= 0;
  $('next').disabled = snapshot || state.windowIndex >= clip.windows.length - 1;
  $('timeline-duration').textContent = `${clip.duration} s`;
  $('timeline-note').textContent = state.mask === '000' ? 'NO EVIDENCE SELECTED · NO PREDICTION' : snapshot ? 'RGB STILL / 30 ms RF CAPTURE' : `${clip.windowSeconds} s WINDOW / ${clip.hopSeconds} s HOP`;
  renderPrediction('fusion', comparison);
  renderPrediction('qwen', comparison);
  renderChart(clip, comparison);
  renderClock();
  renderMaskTable(clip, comparison);
  document.querySelectorAll('[data-example-id]').forEach(node => node.setAttribute('aria-pressed', String(node.dataset.exampleId === clip.id)));
  const atCue = clip.cue && comparison.window && comparison.window.start <= clip.cue.time && Math.abs(comparison.window.end - clip.cue.time) < 0.001;
  $('story-note').querySelector('p').textContent = state.mask === '000' ? 'Every available input is excluded. Both predictions are cleared. The source evidence stays visible.' : atCue && state.mask === clip.cue.mask && clip.cue.caption ? clip.cue.caption : clip.availableMask === '100' ? 'This released evaluation uses camera evidence alone. RF and audio were not paired with this video.' : `${maskLabel(state.mask)} supplied to both models for this exact saved observation.`;
}

function renderMaskTable(clip, comparison) {
  $('mask-summary').textContent = `Inspect ${clip.supportedMasks.length} released sensor ${clip.supportedMasks.length === 1 ? 'selection' : 'combinations'}`;
  $('table-caption').textContent = clip.mediaKind === 'snapshot' ? 'Exact saved RGB still and 30 ms RF observation. Click inputs to compare.' : comparison.window ? `Exact cached window ${formatTime(comparison.window.start)} to ${formatTime(comparison.window.end)} s. Click inputs to hold and compare.` : 'Awaiting the first completed window.';
  const body = $('mask-table');
  if (body.dataset.clipId !== clip.id) {
    body.replaceChildren();
    body.dataset.clipId = clip.id;
    for (const mask of [...clip.supportedMasks].sort().reverse()) {
      const row = element('tr');
      row.dataset.mask = mask;
      const inputCell = element('td');
      const button = element('button', 'mask-select');
      button.append(element('b', '', mask), document.createTextNode(maskLabel(mask)));
      button.addEventListener('click', () => dispatch({ type: 'mask', mask }));
      inputCell.append(button);
      row.append(inputCell, element('td'), element('td'));
      body.append(row);
    }
  }
  for (const row of body.children) {
    const mask = row.dataset.mask;
    row.dataset.selected = String(state.mask === mask);
    const values = study.compare(clip.id, state.windowIndex, mask);
    ['fusion', 'qwen'].forEach((id, index) => {
      const cell = row.children[index + 1];
      cell.textContent = values.kind === 'cached' ? values.selected[id].toFixed(6) : '—';
      cell.title = values.kind === 'cached' ? String(values.selected[id]) : '';
    });
  }
}

function renderProvenance(clip) {
  const target = $('provenance');
  target.replaceChildren();
  target.append(element('p', '', clip.description || `Real released evidence from ${clip.dataset}.`));
  const sourceList = element('dl');
  for (const [key, value] of Object.entries(clip.source || {})) {
    sourceList.append(element('dt', '', key.replace(/([a-z])([A-Z])/g, '$1 $2')), element('dd', '', typeof value === 'object' ? JSON.stringify(value) : String(value)));
  }
  target.append(sourceList);
  const notes = study.evidence[clip.dataset] || (clip.dataset === 'TRIDENT' ? study.evidence : {});
  for (const value of Object.values(notes)) if (typeof value === 'string' && !/^https?:/.test(value)) target.append(element('p', '', value));
  target.append(element('p', '', clip.mediaKind === 'snapshot' ? 'This is one saved observation. The camera is a still image. The 30 ms interval belongs to the RF capture and is not camera duration or inference latency.' : `Timing uses the latest completed ${clip.windowSeconds} s window at or before the video clock. Holding and stepping snap to its end. Sensor switches preserve this interval. ${clip.rfFrames.length ? 'The RF pane shows the last source frame included in that exact released window.' : 'No paired RF or audio is supplied for this video.'}`));
  target.append(element('p', '', 'Mask bits are Camera, RF, Audio. Each score comes directly from the corresponding saved V2 evaluation row. Threshold 0.5 produces the displayed class. A false alarm means a score of at least 0.5 for a window labeled background. Qwen uses normalized lexical yes/no scores; these outputs are not calibrated confidence. No browser inference or attention measurement runs.'));
  const link = element('a', '', 'Inspect the pinned released implementation ↗');
  link.href = clip.source?.sourceUrl || study.evidence.sourceUrl || '';
  if (!/^https:\/\//.test(link.href)) link.removeAttribute('href');
  link.target = '_blank';
  link.rel = 'noreferrer';
  if (link.hasAttribute('href')) target.append(link);
}

function announce() {
  const clip = study.clip(state.clipId);
  const comparison = study.compare(state.clipId, state.windowIndex, state.mask);
  const support = clip.mediaKind === 'snapshot' ? 'Saved still and RF observation.' : comparison.window ? `Held window ${comparison.window.start.toFixed(2)} to ${comparison.window.end.toFixed(2)} seconds.` : '';
  $('announcement').textContent = comparison.kind === 'cached' ? `${clip.dataset}. ${maskLabel(state.mask)}. ${support} Fusion classifier ${comparison.selected.fusion.toFixed(3)}. Qwen detector ${comparison.selected.qwen.toFixed(3)}.` : comparison.kind === 'no-evidence' ? 'No evidence selected. Both model predictions are cleared.' : comparison.kind === 'unsupported' ? 'No released output exists for this sensor selection.' : 'Awaiting the first complete window.';
}

function buildControls() {
  for (const clip of study.clips) {
    const option = element('option', '', `${clip.dataset} · ${clip.label ? 'YES, drone' : 'NO, background'} · ${clip.title}`);
    option.value = clip.id;
    $('clip-select').append(option);
  }
  sensors.forEach((sensor, index) => {
    const button = element('button', 'sensor-toggle');
    button.dataset.sensorSwitch = sensor.id;
    button.setAttribute('role', 'switch');
    button.setAttribute('aria-checked', 'true');
    button.setAttribute('aria-label', `${sensor.label} evidence supplied to both models`);
    button.title = `${sensor.label} evidence (${sensor.key}). Changing inputs holds the same window.`;
    button.append(element('span', 'switch-track'), document.createTextNode(sensor.label), element('small', '', 'INCLUDED'));
    button.addEventListener('click', () => dispatch({ type: 'sensor', index }));
    $('sensor-switches').append(button);
  });
  $('example-pairs').replaceChildren();
  for (const pair of study.pairs) {
    const card = element('section', 'dataset-card');
    card.setAttribute('aria-label', `${pair.label} examples`);
    const positive = study.clip(pair.positiveId);
    card.append(element('h2', '', pair.label), element('p', 'dataset-media', positive.mediaKind === 'snapshot' ? 'RGB still + 30 ms RF' : positive.availableMask === '100' ? 'Camera video only' : 'Camera video + RF + audio'));
    const buttons = element('div', 'pair-buttons');
    for (const id of [pair.positiveId, pair.negativeId]) {
      const clip = study.clip(id);
      const button = element('button', 'example-button');
      button.dataset.exampleId = id;
      button.setAttribute('aria-pressed', 'false');
      button.setAttribute('aria-label', `${pair.label}, ${clip.label ? 'YES, drone present' : 'NO, background'}`);
      button.append(element('strong', '', clip.label ? 'YES' : 'NO'), element('span', '', clip.label ? 'Drone present' : 'Background'));
      button.addEventListener('click', () => dispatch({ type: 'clip', clipId: id }));
      buttons.append(button);
    }
    card.append(buttons);
    $('example-pairs').append(card);
  }
}

async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch {
    $('announcement').textContent = 'Fullscreen is unavailable in this browser window.';
  }
}

$('fullscreen').addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', () => $('fullscreen').setAttribute('aria-label', document.fullscreenElement ? 'Exit fullscreen' : 'Enter fullscreen'));
$('play').addEventListener('click', () => dispatch({ type: 'play' }));
$('hold').addEventListener('click', () => dispatch({ type: 'hold' }));
$('previous').addEventListener('click', () => dispatch({ type: 'step', direction: -1 }));
$('next').addEventListener('click', () => dispatch({ type: 'step', direction: 1 }));
$('seek').addEventListener('input', event => dispatch({ type: 'seek', index: Number(event.target.value) }));
$('listen').addEventListener('click', () => dispatch({ type: 'listen' }));
$('clip-select').addEventListener('change', event => dispatch({ type: 'clip', clipId: event.target.value }));
document.addEventListener('keydown', event => {
  if (event.repeat || event.altKey || event.ctrlKey || event.metaKey || event.target.closest('input,select,textarea,button,a,summary,[contenteditable=true]')) return;
  const sensorIndex = sensors.findIndex(sensor => sensor.key === event.key);
  if (event.code === 'Space') { event.preventDefault(); dispatch({ type: 'play' }); }
  else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); dispatch({ type: 'step', direction: event.key === 'ArrowRight' ? 1 : -1 }); }
  else if (sensorIndex >= 0) { event.preventDefault(); dispatch({ type: 'sensor', index: sensorIndex }); }
  else if (event.key.toLowerCase() === 'f') { event.preventDefault(); toggleFullscreen(); }
});

try {
  study = await loadStudy('./data/demo.json');
  const clip = study.clip(study.pairs[0].positiveId);
  state = { clipId: clip.id, windowIndex: Math.max(0, study.windowAtTime(clip, clip.cue?.time ?? clip.windows[0].end)), mask: clip.availableMask, transport: 'held', listening: false };
  buildControls();
  for (const model of study.models) if ($(`${model.id}-label`)) $(`${model.id}-label`).textContent = model.label;
  loadMedia(clip);
  render();
  $('instrument').hidden = false;
  $('details-section').hidden = false;
  $('load-status').hidden = true;
} catch (error) {
  $('load-status').classList.add('error');
  $('load-status').replaceChildren(element('strong', '', 'Replay data unavailable'), element('p', '', error.message), element('p', '', 'Serve the demo folder over local HTTP and keep data/demo.json with its original assets. No replacement scores or media are generated.'));
}
