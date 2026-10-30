export const sensors = Object.freeze([
  { id: 'camera', label: 'Camera', key: '1' },
  { id: 'rf', label: 'RF', key: '2' },
  { id: 'audio', label: 'Audio', key: '3' },
]);

export async function loadStudy(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Replay pack could not be loaded (${response.status}).`);
  const pack = await response.json();
  const require = (condition, message) => { if (!condition) throw new Error(`Invalid replay pack. ${message}`); };
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const near = (a, b) => Math.abs(a - b) < 0.001;
  const subset = (mask, available) => /^[01]{3}$/.test(mask) && [...mask].every((bit, index) => bit === '0' || available[index] === '1');
  const expected = {
    TRIDENT: { available: '111', masks: ['001', '010', '011', '100', '101', '110', '111'], kind: 'video', window: 1, hop: 0.25 },
    RFVS: { available: '110', masks: ['010', '100', '110'], kind: 'snapshot', window: 0.03, hop: null },
    Halmstad: { available: '100', masks: ['100'], kind: 'video', window: 1, hop: 1 },
  };
  require(pack.version === 2 && pack.experiment === 'Version 2' && pack.threshold === 0.5, 'Expected schema 2, Version 2 evaluations and threshold 0.5.');
  require(pack.maskOrder?.join(',') === 'camera,rf,audio', 'Mask order must be camera, RF, audio.');
  require(Array.isArray(pack.models) && ['fusion', 'qwen'].every(id => pack.models.some(model => model.id === id)), 'Both model definitions are required.');
  require(Array.isArray(pack.clips) && pack.clips.length === 6, 'Six evidence examples are required.');
  require(Array.isArray(pack.pairs) && pack.pairs.length === 3, 'Three dataset pairs are required.');
  const ids = new Set();
  const mediaUrl = path => {
    require(typeof path === 'string' && path.length > 0 && !/^(?:[a-z]+:|\/|\\)/i.test(path) && !path.split(/[\/\\]/).includes('..'), 'Media must reference a local asset.');
    return new URL(path, new URL('./', document.baseURI)).href;
  };
  for (const clip of pack.clips) {
    require(typeof clip.id === 'string' && !ids.has(clip.id), 'Example IDs must be unique.');
    ids.add(clip.id);
    const spec = expected[clip.dataset];
    require(spec && clip.mediaKind === spec.kind && clip.availableMask === spec.available, `${clip.id} has invalid source availability.`);
    require(typeof clip.title === 'string' && finite(clip.duration) && clip.duration > 0 && [0, 1].includes(clip.label), `${clip.id} has invalid metadata.`);
    require(near(clip.windowSeconds, spec.window) && (spec.hop === null ? clip.hopSeconds === null : near(clip.hopSeconds, spec.hop)), `${clip.id} has invalid source cadence.`);
    require(Array.isArray(clip.supportedMasks) && clip.supportedMasks.length === spec.masks.length && spec.masks.every(mask => clip.supportedMasks.includes(mask)), `${clip.id} must contain its exact released masks.`);
    require(Array.isArray(clip.windows) && clip.windows.length === (clip.dataset === 'TRIDENT' ? 37 : clip.dataset === 'Halmstad' ? 10 : 1), `${clip.id} has an invalid observation count.`);
    require(near(clip.duration, clip.mediaKind === 'snapshot' ? 0.03 : 10), `${clip.id} has invalid source duration.`);
    require(Array.isArray(clip.rfFrames) && Array.isArray(clip.waveform), `${clip.id} has invalid media collections.`);
    require(clip.rfFrames.every((frame, index) => finite(frame.time) && frame.time >= 0 && (index === 0 || frame.time > clip.rfFrames[index - 1].time)), `${clip.id} has invalid RF timestamps.`);
    require(clip.waveform.every(value => finite(value) && value >= 0 && value <= 1), `${clip.id} has invalid source waveform.`);
    const hasRf = clip.availableMask[1] === '1';
    const hasAudio = clip.availableMask[2] === '1';
    require(hasRf ? clip.rfFrames.length > 0 : clip.rfFrames.length === 0, `${clip.id} has inconsistent RF availability.`);
    require(hasAudio ? typeof clip.audio === 'string' && clip.waveform.length > 0 : clip.audio === null && clip.waveform.length === 0, `${clip.id} has inconsistent audio availability.`);
    require(clip.mediaKind === 'video' ? typeof clip.video === 'string' : clip.video === null, `${clip.id} has inconsistent camera media.`);
    clip.windows.forEach((window, index) => {
      const expectedStart = clip.mediaKind === 'snapshot' ? 0 : index * clip.hopSeconds;
      require(finite(window.start) && finite(window.end) && near(window.start, expectedStart) && near(window.end - window.start, clip.windowSeconds) && window.end <= clip.duration + 0.001 && window.label === clip.label, `${clip.id} has an invalid cached interval.`);
      require(window.scores && Object.keys(window.scores).length === clip.supportedMasks.length && Object.keys(window.scores).every(mask => clip.supportedMasks.includes(mask) && mask !== '000' && subset(mask, clip.availableMask)), `${clip.id} has an unsupported score mask.`);
      for (const mask of clip.supportedMasks) for (const model of ['fusion', 'qwen']) {
        const score = window.scores[mask]?.[model];
        require(finite(score) && score >= 0 && score <= 1, `${clip.id}, observation ${index}, mask ${mask}, ${model} has no valid cached score.`);
      }
      require(Array.isArray(window.rfIndices) && (hasRf ? window.rfIndices.length > 0 : window.rfIndices.length === 0) && window.rfIndices.every(index => Number.isInteger(index) && index >= 1 && index <= clip.rfFrames.length), `${clip.id} has invalid cached RF frame indices.`);
    });
    if (clip.cue) require(finite(clip.cue.time) && clip.cue.time >= clip.windows[0].end && clip.cue.time <= clip.duration + 0.001 && clip.supportedMasks.includes(clip.cue.mask), `${clip.id} has an invalid recorded cue.`);
    clip.video = clip.video === null ? null : mediaUrl(clip.video);
    clip.poster = mediaUrl(clip.poster);
    clip.audio = clip.audio === null ? null : mediaUrl(clip.audio);
    clip.rfFrames = clip.rfFrames.map(frame => ({ ...frame, src: mediaUrl(frame.src) }));
  }
  const clipById = new Map(pack.clips.map(clip => [clip.id, clip]));
  const pairedIds = new Set();
  const datasets = new Set();
  for (const pair of pack.pairs) {
    const positive = clipById.get(pair.positiveId);
    const negative = clipById.get(pair.negativeId);
    require(typeof pair.id === 'string' && positive?.label === 1 && negative?.label === 0 && positive.dataset === negative.dataset && pair.label === positive.dataset && !datasets.has(pair.label), 'Dataset pairs must each map one drone and one background example.');
    require(!pairedIds.has(pair.positiveId) && !pairedIds.has(pair.negativeId), 'Each example must belong to one pair.');
    pairedIds.add(pair.positiveId); pairedIds.add(pair.negativeId); datasets.add(pair.label);
  }
  const windowAtTime = (clip, time) => {
    require(clip.mediaKind === 'video', 'Snapshots do not have a playback clock.');
    let index = -1;
    for (let i = 0; i < clip.windows.length && clip.windows[i].end <= time + 0.0001; i++) index = i;
    return index;
  };
  return {
    experiment: pack.experiment, evidence: pack.evidence || {}, threshold: pack.threshold,
    models: pack.models, clips: pack.clips, pairs: pack.pairs,
    clip: id => clipById.get(id), windowAtTime,
    canToggle: (id, index) => clipById.get(id).availableMask[index] === '1',
    compare(id, index, mask) {
      const clip = clipById.get(id);
      const window = clip.windows[index];
      if (mask === '000') return { kind: 'no-evidence', window };
      if (!subset(mask, clip.availableMask) || !clip.supportedMasks.includes(mask)) return { kind: 'unsupported', window };
      if (!window) return { kind: 'awaiting' };
      return { kind: 'cached', window, selected: window.scores[mask], reference: window.scores[clip.availableMask], referenceMask: clip.availableMask };
    },
    scores(id, mask) {
      const clip = clipById.get(id);
      if (!clip.supportedMasks.includes(mask)) return [];
      return clip.windows.map(window => ({ start: window.start, end: window.end, ...window.scores[mask] }));
    },
    rfForWindow(clip, index) {
      const window = clip.windows[index];
      return window?.rfIndices.length ? clip.rfFrames[window.rfIndices.at(-1) - 1] : null;
    },
  };
}
