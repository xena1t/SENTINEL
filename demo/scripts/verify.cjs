const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const read = file => JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'));
const pack = read('data/demo.json');
const url = process.env.SENTINEL_DEMO_URL || 'http://127.0.0.1:4173';
const output = path.resolve(root, '../.audit/sentinel-demo-external');
fs.mkdirSync(output, { recursive: true });
assert.equal(pack.version, 2);
assert.equal(pack.clips.length, 6);
assert.equal(pack.clips.filter(clip => clip.label === 1).length, 3);
assert.equal(pack.clips.filter(clip => clip.label === 0).length, 3);
assert.deepEqual(pack.pairs.map(pair => pair.label), ['TRIDENT', 'RFVS', 'Halmstad']);
const expectedComparisons = pack.clips.reduce((total, clip) => total + clip.windows.length * clip.supportedMasks.length * 2, 0);
const tridentFusion = new Map(read('data/published/research__test-v2__fusion-predictions.json').rows.map(row => [`${row.sample_id}/${row.mask}`, row]));
const tridentQwen = new Map(read('data/published/research__test-v2__qwen-predictions.json').rows.map(row => [`${row.sample_id}/${row.mask}`, row]));
const external = new Map(['rfvs', 'halmstad-video'].flatMap(dataset => read(`data/external/published/research__external__${dataset}__predictions.json`).rows).map(row => [`${row.sample_id}/${row.mask}`, row]));
for (const clip of pack.clips) for (const window of clip.windows) for (const mask of clip.supportedMasks) {
  const key = `${window.sampleId}/${mask}`;
  const f = clip.dataset === 'TRIDENT' ? tridentFusion.get(key) : external.get(key);
  const q = clip.dataset === 'TRIDENT' ? tridentQwen.get(key) : external.get(key);
  assert(f && q, `Missing actual released row ${key}`);
  assert.equal(window.label, f.label);
  assert.equal(window.scores[mask].fusion, f.fusion_probability);
  assert.equal(window.scores[mask].qwen, q.probability);
}

async function values(page) {
  return page.evaluate(() => ({
    clip: document.querySelector('#clip-select').value,
    index: Number(document.querySelector('#seek').value),
    mask: document.querySelector('#mask-code b').textContent,
    fusion: document.querySelector('#fusion-score').textContent,
    qwen: document.querySelector('#qwen-score').textContent,
    fusionTitle: document.querySelector('#fusion-score').getAttribute('title'),
    qwenTitle: document.querySelector('#qwen-score').getAttribute('title'),
  }));
}
function assertScores(actual) {
  const clip = pack.clips.find(clip => clip.id === actual.clip);
  const scores = clip.windows[actual.index].scores[actual.mask];
  for (const model of ['fusion', 'qwen']) {
    assert.equal(actual[model], scores[model].toFixed(3));
    assert.equal(actual[`${model}Title`], `Stored value ${scores[model]}`);
  }
}
async function selected(page, clip) {
  await page.locator(`[data-example-id="${clip.id}"]`).click();
  if (clip.mediaKind === 'video') {
    await page.waitForFunction(videoPath => {
      const video = document.querySelector('video');
      return video.currentSrc.endsWith(videoPath) && video.readyState >= 2 && !video.seeking;
    }, clip.video);
  } else {
    await page.waitForFunction(poster => {
      const image = document.querySelector('#camera-still');
      return image.src.endsWith(poster) && image.complete && image.naturalWidth > 0;
    }, clip.poster);
  }
  const current = await values(page);
  assert.equal(current.clip, clip.id);
  assert.equal(current.mask, clip.availableMask);
  assertScores(current);
}
async function run() {
  const browser = await chromium.launch({ headless: true, ...(process.env.SENTINEL_BROWSER_PATH ? { executablePath: process.env.SENTINEL_BROWSER_PATH } : {}) });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const errors = [], externalRequests = [];
    await context.route('**/*', route => {
      if (new URL(route.request().url()).origin === new URL(url).origin) return route.continue();
      externalRequests.push(route.request().url());
      return route.abort();
    });
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.goto(url);
    await page.locator('#instrument').waitFor({ state: 'visible' });
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator('[data-example-id]').count(), 6);
    assert.equal(await page.locator('#clip-select option').count(), 6);
    await page.locator('#details-section details').first().locator('summary').click();
    let comparisons = 0;
    const examples = [];
    for (const clip of pack.clips) {
      await selected(page, clip);
      assert.equal(await page.locator('#mask-table tr').count(), clip.supportedMasks.length);
      for (const [index, sensor] of ['camera', 'rf', 'audio'].entries()) {
        const button = page.locator(`[data-sensor-switch="${sensor}"]`);
        assert.equal(await button.isDisabled(), clip.availableMask[index] === '0');
      }
      assert.equal(await page.locator('#waveform').isVisible(), Boolean(clip.audio));
      if (clip.mediaKind === 'snapshot') {
        assert.equal(await page.locator('#play').isVisible(), false);
        assert.equal(await page.locator('video').getAttribute('src'), null);
        assert.equal(await page.locator('audio').getAttribute('src'), null);
        assert.equal(await page.locator('#fusion-trace').getAttribute('d'), '');
        assert.match(await page.locator('#camera-clock').textContent(), /STILL/);
        assert.match(await page.locator('#window-time').textContent(), /30 ms/);
      } else {
        await page.locator('#seek').evaluate(input => { input.value = 0; input.dispatchEvent(new Event('input', { bubbles: true })); });
        if (clip.audio) await page.locator('#listen').click();
        else assert.equal(await page.locator('audio').getAttribute('src'), null);
        await page.locator('#play').click();
        await page.waitForFunction(() => !document.querySelector('video').paused && document.querySelector('video').currentTime > 1.2);
        if (clip.audio) assert.equal(await page.locator('audio').evaluate(audio => audio.paused), false);
        await page.locator('#hold').click();
        if (clip.audio) await page.locator('#listen').click();
        assertScores(await values(page));
      }
      const result = await page.evaluate(clip => {
        const issues = [];
        let checked = 0;
        for (let index = 0; index < clip.windows.length; index++) {
          if (clip.mediaKind === 'video') {
            const seek = document.querySelector('#seek');
            seek.value = index;
            seek.dispatchEvent(new Event('input', { bubbles: true }));
          }
          if (clip.rfFrames.length) {
            const expected = clip.rfFrames[clip.windows[index].rfIndices.at(-1) - 1].src;
            if (!document.querySelector('#rf-image').src.endsWith(expected)) issues.push(`${index}/RF`);
          } else if (document.querySelector('#rf-image').hasAttribute('src')) issues.push('Unpaired RF source loaded');
          for (const mask of clip.supportedMasks) {
            document.querySelector(`tr[data-mask="${mask}"] button`).click();
            for (const model of ['fusion', 'qwen']) {
              const expected = clip.windows[index].scores[mask][model];
              const element = document.querySelector(`#${model}-score`);
              if (element.textContent !== expected.toFixed(3) || element.title !== `Stored value ${expected}`) issues.push(`${index}/${mask}/${model}`);
              checked++;
            }
          }
        }
        return { checked, issues };
      }, clip);
      assert.deepEqual(result.issues, []);
      comparisons += result.checked;
      await page.locator(`tr[data-mask="${clip.availableMask}"] button`).click();
      const indexBefore = (await values(page)).index;
      for (const [index, sensor] of ['camera', 'rf', 'audio'].entries()) if (clip.availableMask[index] === '1') await page.locator(`[data-sensor-switch="${sensor}"]`).click();
      const empty = await values(page);
      assert.equal(empty.mask, '000');
      assert.equal(empty.index, indexBefore);
      for (const model of ['fusion', 'qwen']) {
        assert.equal(empty[model], '—');
        assert.equal(empty[`${model}Title`], null);
      }
      await page.locator('body').click({ position: { x: 2, y: 2 } });
      for (let index = 0; index < 3; index++) if (clip.availableMask[index] === '0') {
        await page.keyboard.press(String(index + 1));
        assert.equal((await values(page)).mask, '000');
      }
      await selected(page, clip);
      await page.waitForFunction(() => !document.querySelector('video').seeking);
      await page.screenshot({ path: path.join(output, `${clip.id}.png`), fullPage: true });
      examples.push({ id: clip.id, dataset: clip.dataset, label: clip.label, mediaKind: clip.mediaKind, supportedMasks: clip.supportedMasks, verified: true });
    }
    assert.equal(comparisons, expectedComparisons);
    await selected(page, pack.clips.find(clip => clip.id === 'rfvs-background'));
    await page.locator('tr[data-mask="100"] button').click();
    assert.equal(await page.locator('#qwen-card').getAttribute('data-outcome'), 'false-positive');
    assert.doesNotMatch(await page.locator('#story-note p').textContent(), /Qwen rejects/);
    await selected(page, pack.clips.find(clip => clip.id === 'drone'));
    await page.locator('tr[data-mask="010"] button').click();
    assert.equal(await page.locator('#qwen-card').getAttribute('data-outcome'), 'false-negative');
    assert.doesNotMatch(await page.locator('#story-note p').textContent(), /Both cached models detect/);
    await page.locator('#details-section details').first().locator('summary').click();
    await selected(page, pack.clips.find(clip => clip.id === 'rfvs-background'));
    assert.equal(await page.locator('#fusion-card').getAttribute('data-outcome'), 'false-positive');
    assert.equal(await page.locator('#qwen-card').getAttribute('data-outcome'), 'correct');
    await selected(page, pack.clips.find(clip => clip.id === 'halmstad-background'));
    assert.equal((await values(page)).index, 5);
    assert.equal(await page.locator('#fusion-card').getAttribute('data-outcome'), 'false-positive');
    assert.equal(await page.locator('#qwen-card').getAttribute('data-outcome'), 'correct');
    await page.evaluate(() => {
      for (let i = 0; i < 3; i++) for (const id of ['drone', 'rfvs-drone', 'halmstad-background', 'rfvs-background']) document.querySelector(`[data-example-id="${id}"]`).click();
    });
    await page.waitForFunction(() => document.querySelector('#camera-still').complete && document.querySelector('#camera-still').naturalWidth > 0);
    assert.equal((await values(page)).clip, 'rfvs-background');
    assert.equal(await page.locator('video').getAttribute('src'), null);
    assertScores(await values(page));
    await page.locator('#fullscreen').click();
    await page.waitForFunction(() => document.fullscreenElement !== null);
    await page.locator('#fullscreen').click();
    await page.waitForFunction(() => document.fullscreenElement === null);
    const viewports = [];
    for (const [width, height] of [[1440,1000],[1280,900],[768,1024],[390,844]]) {
      await page.setViewportSize({width,height});
      const layout = await page.evaluate(() => ({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));
      assert(layout.scrollWidth <= width+1);
      viewports.push(layout);
      await page.screenshot({path:path.join(output,`viewport-${width}.png`),fullPage:true});
    }
    await page.emulateMedia({reducedMotion:'reduce'});
    await page.locator('[data-sensor-switch="camera"]').click();
    assert.equal(await page.evaluate(() => document.getAnimations().filter(animation => animation.playState==='running').length),0);
    const corrupt = await context.newPage();
    const invalid = structuredClone(pack);
    invalid.clips.find(clip=>clip.id==='halmstad-drone').windows[0].scores['110']={fusion:0,qwen:0};
    await corrupt.route('**/data/demo.json',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(invalid)}));
    await corrupt.goto(url);
    await corrupt.locator('#load-status.error').waitFor();
    assert.equal(await corrupt.locator('#instrument').isVisible(),false);
    await corrupt.close();
    assert.deepEqual(errors,[]);
    assert.deepEqual(externalRequests,[]);
    const report={passed:true,clips:6,exactDisplayedProbabilities:comparisons,exactReleasedSourceProbabilities:expectedComparisons,examples,emptyEvidence:true,unsupportedSensors:true,snapshotNoPlayback:true,sourceSpecificMasks:true,sourceSpecificCadence:true,pairedAudioOnly:true,rapidSourceSwitching:true,maskSpecificCaptions:true,fullscreen:true,reducedMotion:true,invalidMaskRejected:true,errors,externalRequests,viewports};
    fs.writeFileSync(path.join(output,'browser-report.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));
  } finally { await browser.close(); }
}
run().catch(error=>{console.error(error);process.exitCode=1;});
