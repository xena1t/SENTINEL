# Cached demo architecture

The demo runs offline so that a presentation does not depend on GPU access or model startup.
The browser reads packaged media and exact saved Version 2 predictions. It does not run inference.

## Source-specific observations

`data/demo.json` defines dataset pairs, examples, available inputs, supported masks, and observation windows.
TRIDENT provides video, RF, and paired audio with one-second windows at 0.25-second hops.
RFVS provides one camera still and a 30 ms RF capture per example.
Halmstad provides camera-only video with nonoverlapping one-second windows.

These differences determine the controls. RFVS remains a fixed observation, and inputs that were not recorded stay disabled.
Each example's reference uses every input recorded for that source.

## Browser state and score selection

`data.js` loads and validates the packaged contract. `app.js` owns the selected example, input mask, window, and playback state.
The browser selects both model scores by the same observation identifier and input mask.
Unsupported combinations have no substituted score. An all-missing mask clears the predictions.

Video time determines the latest completed window. Hold and step controls seek to that window's end.
RFVS has no playback clock or fabricated score history.
The RF pane shows the final source frame included in the selected observation.

## Local delivery

`serve.py` serves this folder over HTTP on `127.0.0.1` and supports byte-range requests for video seeking.
All runtime scripts, fonts, media, and scores are local.
There is no inference API or GPU connection.

The builders package source-backed examples and verify released probabilities.
Pinned source files, hashes, and media provenance remain under `data`.
The [evidence reference](docs/evidence.md) describes score meaning and source limitations.
