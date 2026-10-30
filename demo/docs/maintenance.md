# Verify or rebuild the demo

Run these developer commands from the repository directory.
To present the existing demo, use the [startup guide](../README.md) instead.

## Verify the evidence pack

Install FFmpeg so that `ffprobe` is on your executable search path.
Then run:

```sh
python3 demo/scripts/prepare_data.py --verify-only
```

Check that the JSON report contains `"passed": true`.
The command checks source membership, source hashes, media, and exact cached probabilities.
It also runs the external verifier and regenerates both `verification.json` reports, including example and observation counts.

## Verify browser behavior

Install Playwright and its Chromium browser in your developer environment.
Start `python3 demo/serve.py` in another terminal, then run:

```sh
node demo/scripts/verify.cjs
```

If the server uses another port, set `SENTINEL_DEMO_URL` to its URL.
If Playwright is outside normal Node module resolution, set `NODE_PATH` to its module directory.
If Chromium is outside Playwright's default cache, set `SENTINEL_BROWSER_PATH` to its executable.

Inspect the report and screenshots under `.audit/sentinel-demo-external/`.
The check compares browser scores with the released source rows and exercises playback, sensor selection, example changes, and responsive layouts.

## Regenerate the assets

Install NumPy, SciPy, Matplotlib, Pillow, and FFmpeg before regeneration.
Provide the original TRIDENT media at the paths in the released source manifest or in the builder's media cache.
Provide the source font files under `presentations/sentinel/source/`.
The presentation package contains rendered assets but does not include every TRIDENT source frame or the presentation's source font directory.

Build the external examples first, then assemble the combined pack:

```sh
python3 demo/scripts/prepare_external.py
python3 demo/scripts/prepare_data.py
```

To fetch missing external sources, add `--download` to `prepare_external.py`.
To fetch missing TRIDENT camera and RF frames, add `--download-media` to `prepare_data.py`.
Provide the TRIDENT WAV files separately at their manifest paths.
These commands package released predictions. They do not train models or run inference.

For pinned source identities and packaged hashes, inspect [the data contract](../data/README.md) and `demo/data/provenance.json`.
