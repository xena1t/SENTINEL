# External examples

This pack contains one drone and one background example from each of RFVS and Halmstad. These are purposefully selected illustrations, not a performance estimate.

The prediction caches and adapters are exact Git blobs from SENTINEL commit `7fc465480f4f5b1ad348ffef93c744712e06b098`. The pack retains the released sample identifiers, labels and floating point probabilities. `provenance.json` records SHA-256 hashes for every original source and displayed asset.

RFVS provides an RGB still and a 30 ms RF capture for each sample. The released adapter repeats the still across eight model frame slots and converts the RF capture into four consecutive 7.5 ms spectrogram panels. These examples support masks `110`, `100` and `010`. Each is one sample, with no video transport and no audio. Original image, exact-stem label file and raw RF capture are retained under `source/rfvs`. The public archive is accessed through six bounded HTTP ranges validated against its pinned ZIP manifest, local headers, member sizes and CRCs.

Halmstad provides camera video without paired RF or audio. The two original files are pinned to the dataset repository commit `e7a6eaf9760fa3f9ecc37d09cd11a8cd20be90dd`. The drone file has 301 frames and the bird file has 323 frames, both at 30 fps. Displayed videos contain their first 300 frames, encoded as 10 s excerpts with source offset zero. The released adapter uses ten nonoverlapping one-second windows. Each window samples original frame offsets 0, 4, 8, 12, 17, 21, 25 and 29. Only mask `100` is available.

The bird example's cue ends at six seconds, which selects released sample `halmstad-video:V_BIRD_022:s05`. Qwen detector rejects that bird window while Fusion classifier produces a false alarm. The RFVS background example also retains a real disagreement. All displayed scores come from the released V2 caches.

Run verification without model libraries or inference:

```sh
python3 demo/scripts/prepare_external.py --verify-only
```

The verifier needs `ffprobe` on the executable search path. Regeneration also needs NumPy, SciPy, Matplotlib, Pillow and `ffmpeg`:

```sh
python3 demo/scripts/prepare_external.py
```

Add `--download` to acquire missing public files. Downloads are limited to the pinned source blobs, two original Halmstad videos and six RFVS members. The full datasets are not downloaded. No model inference or training runs.
