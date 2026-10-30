# SENTINEL evidence pack

`demo.json` is schema version 2. It contains six examples in three explicit drone/background pairs. The experiment name Version 2 refers separately to the released model evaluation. The browser selects actual cached outputs and does not run inference.

| Dataset | Positive source | Negative source | Observation unit | Supported masks |
| --- | --- | --- | --- | --- |
| TRIDENT | `Train/Drone/Drone_16/1` | `Test/Background/Background_1/3` | 37 one-second windows per recording, 0.25-second hop | `111`, `110`, `101`, `011`, `100`, `010`, `001` |
| RFVS | `AIR2s_58296527` | `house_background_124157_73` | One camera still and 30 ms RF capture per example | `110`, `100`, `010` |
| Halmstad | `V_DRONE_001` | `V_BIRD_022` | Ten nonoverlapping one-second video windows per example | `100` |

Mask bits mean camera, RF, and audio, in that order. `000` is a UI state with no score. An unavailable sensor cannot be enabled. Supported masks map to exact released rows for both models. Missing mask evaluations never receive zero or a substituted score.

## Data contract

`pairs` maps each dataset to one `positiveId` and one `negativeId`. `clips` contains the six examples. Each example declares `dataset`, `mediaKind`, `availableMask`, `supportedMasks`, its actual cadence, and local media. `mediaKind` discriminates a playable video from a fixed snapshot. Absent media are null or empty collections.

Each `windows` entry preserves the exact `sampleId`, label, support, and cached `scores[mask].fusion` and `scores[mask].qwen`. The field holds temporal video windows or a single RFVS paired observation. RFVS's start/end describe RF support only. Its hop is null because there is no observation sequence.

RF frame indices in the browser contract are one-based positions in `rfFrames`. TRIDENT preserves released indices. RFVS normalizes the four adapter chunk positions to one-based indices. Halmstad has an empty index list because RF is unpaired. The RF pane resolves the final actual frame included in the selected observation.

## Source evidence

All prediction and adapter files are pinned to SENTINEL commit `7fc465480f4f5b1ad348ffef93c744712e06b098`. `published-index.json` records the TRIDENT Git blob and SHA256 identities. `external/provenance.json` records the external pinned sources, raw public media, preprocessing, and packaged hashes.

TRIDENT membership comes from the released V2 role=test manifest, not its publisher folder name. The protocol repeats previously observed Version 1 recordings. Labels inherit recording presence and do not prove visibility in each frame.

RFVS samples use exact-stem RGB, RF IQ, and label members from the public archive. Archive offsets, sizes, CRCs, and source hashes are recorded. Four true spectrogram chunks use the exact released adapter. The still is not animated into a video. Audio is absent.

Halmstad sources come from the pinned public dataset. Original drone and bird videos have 301 and 323 frames at 30 fps. The released adapter evaluates ten complete one-second windows in each. Local excerpts retain the first 300 frames and source offset zero. The bird cue ends at 6 seconds and selects original sample `halmstad-video:V_BIRD_022:s05`. No unpaired audio or RF is attached.

## Verification

The combined verification checks six examples, one positive and negative per dataset, 94 video windows, two fixed observations, and 1,088 exact probability values. Source-specific checks validate original media, packaged asset hashes, cadence, mask legality, and source row identity.

```sh
python3 demo/scripts/prepare_data.py --verify-only
```

`provenance.json` binds the combined JSON and asset hashes to the external provenance receipt. `verification.json` records the latest completed check. The six examples are purposefully selected illustrations, not a random sample, an accuracy estimate, or proof of generalization.
