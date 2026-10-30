#!/usr/bin/env python3
"""Package original TRIDENT media and released V2 predictions without inference."""
from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
DATA = DEMO / "data"
ASSETS = DEMO / "assets"
MASKS = ["111", "110", "101", "011", "100", "010", "001"]
MODELS = [{"id": "fusion", "label": "Fusion classifier"}, {"id": "qwen", "label": "Qwen detector"}]
CLIPS = [
    {"id": "drone", "recording": "Train/Drone/Drone_16/1", "title": "Drone presence",
     "description": "A real outdoor drone recording. Both models consistently detect presence with all sensors available.",
     "cue": {"time": 4, "mask": "111", "caption": "Both cached models detect drone presence throughout this recording."}},
    {"id": "background", "recording": "Test/Background/Background_1/3", "title": "Quiet background",
     "description": "A real background recording. The all-sensor predictions stay below the detection threshold.",
     "cue": {"time": 4, "mask": "111", "caption": "Both models correctly reject the background with all three sensors."}},

]
PAIRS = [
    {"id": "trident", "label": "TRIDENT", "positiveId": "drone", "negativeId": "background"},
    {"id": "rfvs", "label": "RFVS", "positiveId": "rfvs-drone", "negativeId": "rfvs-background"},
    {"id": "halmstad", "label": "Halmstad", "positiveId": "halmstad-drone", "negativeId": "halmstad-background"},
]


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path: Path, algorithm="sha256"):
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def published_inputs():
    index = read_json(DATA / "published-index.json")
    files = {}
    for item in index["files"]:
        path = DATA / item["local"]
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        assert blob == item["gitBlobSha1"], f"Published blob changed: {path}"
        assert digest(path) == item["sha256"], f"Published SHA256 changed: {path}"
        files[item["path"]] = path
    fusion = read_json(files["research/test-v2/fusion-predictions.json"])
    qwen = read_json(files["research/test-v2/qwen-predictions.json"])
    sources = read_json(files["data/windows/test-v2/sources.json"])
    windows = [json.loads(line) for line in files["data/windows/test-v2/manifest.jsonl"].read_text().splitlines() if line]
    assert len(windows) == 1961 and len(sources) == 53
    assert len(fusion["rows"]) == len(qwen["rows"]) == 1961 * 7
    by_model = {"fusion": {}, "qwen": {}}
    for model, result, key in [("fusion", fusion, "fusion_probability"), ("qwen", qwen, "probability")]:
        for row in result["rows"]:
            identity = (row["sample_id"], row["mask"])
            assert identity not in by_model[model], f"Duplicate cached row: {identity}"
            assert 0 <= row[key] <= 1 and math.isfinite(row[key])
            by_model[model][identity] = row
    assert by_model["fusion"].keys() == by_model["qwen"].keys()
    for identity in by_model["fusion"]:
        f, q = by_model["fusion"][identity], by_model["qwen"][identity]
        assert f["fusion_probability"] == q["fusion_probability"]
        assert f["label"] == q["label"] and f["recording"] == q["recording"]
    return index, sources, windows, by_model


def original_frame(frame, recording, modality, cache, download):
    candidates = [ROOT / frame["path"], ROOT / "data/trident/images" / modality / recording / frame["name"],
                  cache / modality / recording / frame["name"]]
    for candidate in candidates:
        if candidate.exists() and digest(candidate, "md5") == frame["md5"]:
            return candidate
    target = candidates[-1]
    if not download:
        raise FileNotFoundError(f"Missing original {frame['name']}. Run with --download-media.")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = "https://drive.google.com/uc?export=download&id=" + frame["id"]
    last_error = None
    for attempt in range(4):
        temporary = target.with_suffix(".download")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                temporary.write_bytes(response.read())
            assert digest(temporary, "md5") == frame["md5"], f"Original MD5 mismatch: {frame['name']}"
            temporary.replace(target)
            return target
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(attempt + 1)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def waveform(path: Path):
    with wave.open(str(path)) as stream:
        assert stream.getnchannels() == 1 and stream.getsampwidth() == 2
        assert stream.getframerate() == 44100 and stream.getnframes() == 441000
        values = array("h", stream.readframes(stream.getnframes()))
    if sys.byteorder != "little":
        values.byteswap()
    bins = [max(abs(value) for value in values[i:i + 1470]) / 32768 for i in range(0, len(values), 1470)]
    peak = max(bins)
    return [round(value / peak, 6) for value in bins] if peak else bins


def package_media(source, spec, cache, download, workers):
    recording = spec["recording"]
    folder = ASSETS / "clips" / spec["id"]
    folder.mkdir(parents=True, exist_ok=True)
    rgb, rf = sorted(source["rgb_files"], key=lambda f: f["index"]), sorted(source["rf_files"], key=lambda f: f["index"])
    assert [f["index"] for f in rgb] == list(range(1, 301))
    assert [f["index"] for f in rf] == list(range(1, 41))
    jobs = [(frame, "rgb") for frame in rgb] + [(frame, "rf") for frame in rf]
    resolved = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(original_frame, frame, recording, modality, cache, download): (frame, modality)
                   for frame, modality in jobs}
        count = 0
        for future in as_completed(pending):
            frame, modality = pending[future]
            resolved[(modality, frame["index"])] = future.result()
            count += 1
            if count % 60 == 0:
                print(f"{spec['id']}: verified {count}/340 original frames", flush=True)
    originals = []
    for frame, modality in jobs:
        path = resolved[(modality, frame["index"])]
        originals.append({"modality": modality, "index": frame["index"], "name": frame["name"],
                          "driveId": frame["id"], "url": "https://drive.google.com/uc?export=download&id=" + frame["id"],
                          "md5": frame["md5"], "sha256": digest(path), "bytes": path.stat().st_size,
                          "nominalTime": (frame["index"] - 1) / (30 if modality == "rgb" else 4)})
    video = folder / "camera.mp4"
    with tempfile.TemporaryDirectory(prefix="sentinel-frames-") as temporary:
        sequence = Path(temporary)
        for frame in rgb:
            (sequence / f"{frame['index']:04}.jpg").symlink_to(resolved[("rgb", frame["index"])])
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "30", "-start_number", "1",
                        "-i", str(sequence / "%04d.jpg"), "-frames:v", "300", "-vf", "scale=640:640",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-map_metadata", "-1", "-movflags", "+faststart", str(video)], check=True)
    shutil.copyfile(resolved[("rgb", 1)], folder / "poster.jpg")
    rf_frames = []
    for frame in rf:
        target = folder / f"rf-{frame['index']:02}.jpg"
        shutil.copyfile(resolved[("rf", frame["index"])], target)
        rf_frames.append({"time": (frame["index"] - 1) / 4, "src": str(target.relative_to(DEMO))})
    audio = ROOT / source["audio"]["path"]
    assert digest(audio) == source["audio"]["sha256"]
    shutil.copyfile(audio, folder / "audio.wav")
    originals.append({"modality": "audio", "driveId": source["audio"]["id"], "sha256": digest(audio),
                      "pcmSha256": source["audio"]["pcm_sha256"], "bytes": audio.stat().st_size})
    return {"video": str(video.relative_to(DEMO)), "poster": str((folder / "poster.jpg").relative_to(DEMO)),
            "audio": str((folder / "audio.wav").relative_to(DEMO)), "rfFrames": rf_frames,
            "waveform": waveform(audio)}, originals


def build(download, cache, workers):
    index, sources, windows, by_model = published_inputs()
    clips, provenance = [], []
    for spec in CLIPS:
        source = next(row for row in sources if row["clip_id"] == spec["recording"])
        assert source["role"] == "test"
        rows = sorted([row for row in windows if row["clip_id"] == spec["recording"]], key=lambda row: row["start_tick"])
        assert len(rows) == 37
        media, originals = package_media(source, spec, cache, download, workers)
        clip_windows = []
        for row in rows:
            scores = {}
            for mask in MASKS:
                identity = (row["sample_id"], mask)
                f, q = by_model["fusion"][identity], by_model["qwen"][identity]
                assert f["label"] == q["label"] == row["label"] == source["label"]
                scores[mask] = {"fusion": f["fusion_probability"], "qwen": q["probability"]}
            clip_windows.append({"sampleId": row["sample_id"], "start": row["start_tick"] / 120,
                                 "end": row["end_tick"] / 120, "label": row["label"], "scores": scores,
                                 "rgbIndices": row["rgb_indices"], "rfIndices": row["rf_indices"]})
        clips.append({key: spec[key] for key in ["id", "title", "description", "cue"]} | media |
                     {"label": source["label"], "dataset": "TRIDENT", "mediaKind": "video",
                      "availableMask": "111", "supportedMasks": MASKS, "windowSeconds": 1, "hopSeconds": 0.25,
                      "duration": 10, "windows": clip_windows,
                      "source": {"dataset": "TRIDENT", "recording": source["clip_id"], "evaluationRole": "V2 observed test",
                                 "publisherSplit": source["published_split"], "group": source["group_proxy"],
                                 "labelProvenance": source["label_provenance"], "videoReconstruction": "300 original camera JPEG frames at nominal 30 fps",
                                 "cameraFps": 30, "cameraFrames": 300, "rfFrames": 40, "rfFps": 4,
                                 "audioSampleRate": 44100, "timingStatus": "Nominal alignment; acquisition timestamps and streaming latency are not verified.",
                                 "releaseCommit": index["commit"]}})
        provenance.append({"id": spec["id"], "recording": spec["recording"], "originals": originals})
    clips.extend(read_json(DATA / "external/examples.json")["clips"])
    fonts = ASSETS / "fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    for name in ["display700.ttf", "display800.ttf", "body.ttf", "Barlow-Condensed-OFL.txt", "IBM-Plex-Sans-OFL.txt"]:
        shutil.copyfile(ROOT / "presentations/sentinel/source" / name, fonts / name)
    result = {"version": 2, "experiment": "Version 2", "threshold": 0.5,
              "masks": MASKS, "maskOrder": ["camera", "rf", "audio"], "models": MODELS, "pairs": PAIRS, "clips": clips,
              "evidence": {"mode": "Cached evaluation replay", "releaseCommit": index["commit"],
                           "sourceUrl": "https://github.com/xena1t/SENTINEL/tree/" + index["commit"],
                           "totalTestRecordings": 53, "totalTestWindows": 1961,
                           "selection": "Three purposefully selected drone/background pairs from TRIDENT, RFVS, and Halmstad. These six examples are not a random sample or a performance estimate.",
                           "testDisclosure": "TRIDENT repeats evaluation of recordings already evaluated in Version 1. External examples retain the released no-retraining evaluation. These selections are not an untouched confirmatory holdout.",
                           "labels": "TRIDENT windows inherit presence labels. RFVS uses released sample labels. Halmstad uses recording classes. Frame-level visibility is not independently established here.",
                           "timing": "TRIDENT uses 1 s windows at 0.25 s hops. Halmstad has nonoverlapping 1 s camera windows. RFVS is a camera still paired with 30 ms RF, without video playback. None is a latency measurement.",
                           "scores": "Exact released cached probability values, without interpolation, recalibration, or new inference."}}
    write_json(DATA / "demo.json", result)
    assets = [{"path": str(path.relative_to(DEMO)), "sha256": digest(path), "bytes": path.stat().st_size}
              for path in sorted(ASSETS.rglob("*")) if path.is_file() and not path.name.startswith("._")]
    write_json(DATA / "provenance.json", {"schema": "sentinel-demo-provenance-v1", "release": index,
                                         "clips": provenance, "assets": assets, "demoSha256": digest(DATA / "demo.json"),
                                         "external": {"path": "data/external/provenance.json", "sha256": digest(DATA / "external/provenance.json")}})
    verify()


def verify():
    index, sources, windows, by_model = published_inputs()
    demo = read_json(DATA / "demo.json")
    provenance = read_json(DATA / "provenance.json")
    assert digest(DATA / "demo.json") == provenance["demoSha256"]
    assert demo["masks"] == MASKS and demo["maskOrder"] == ["camera", "rf", "audio"] and demo["threshold"] == 0.5
    assert demo["version"] == 2 and demo["pairs"] == PAIRS and demo["models"] == MODELS
    assert len(demo["clips"]) == len({clip["id"] for clip in demo["clips"]}) == 6
    by_id = {clip["id"]: clip for clip in demo["clips"]}
    for pair in PAIRS:
        assert by_id[pair["positiveId"]]["label"] == 1 and by_id[pair["negativeId"]]["label"] == 0
        assert by_id[pair["positiveId"]]["dataset"] == by_id[pair["negativeId"]]["dataset"] == pair["label"]
    trident_clips = [clip for clip in demo["clips"] if clip["dataset"] == "TRIDENT"]
    assert [clip["id"] for clip in trident_clips] == [spec["id"] for spec in CLIPS]
    assert [clip["source"]["recording"] for clip in trident_clips] == [spec["recording"] for spec in CLIPS]
    selected_sources = [next(row for row in sources if row["clip_id"] == spec["recording"]) for spec in CLIPS]
    selected_windows = [row for row in windows if row["clip_id"] in {spec["recording"] for spec in CLIPS}]
    expected_probabilities = len(selected_windows) * len(MASKS) * len(MODELS)
    window_index = {row["sample_id"]: row for row in windows}
    checked = 0
    for clip in trident_clips:
        source = next(row for row in sources if row["clip_id"] == clip["source"]["recording"])
        assert source["role"] == "test" and clip["label"] == source["label"]
        released_windows = sorted([row for row in windows if row["clip_id"] == source["clip_id"]], key=lambda row: row["start_tick"])
        assert [row["sampleId"] for row in clip["windows"]] == [row["sample_id"] for row in released_windows]
        assert len(clip["windows"]) == 37 and len(clip["rfFrames"]) == 40
        assert [row["start"] for row in clip["windows"]] == [tick / 4 for tick in range(37)]
        assert all(row["end"] - row["start"] == 1 for row in clip["windows"])
        assert clip["source"]["cameraFrames"] == len(source["rgb_files"])
        assert clip["source"]["rfFrames"] == len(source["rf_files"])
        assert digest(DEMO / clip["audio"]) == source["audio"]["sha256"]
        assert digest(DEMO / clip["poster"], "md5") == source["rgb_files"][0]["md5"]
        for frame, original in zip(clip["rfFrames"], sorted(source["rf_files"], key=lambda row: row["index"])):
            assert frame["time"] == (original["index"] - 1) / 4
            assert digest(DEMO / frame["src"], "md5") == original["md5"]
        for row in clip["windows"]:
            original = window_index[row["sampleId"]]
            assert original["clip_id"] == clip["source"]["recording"]
            assert row["start"] == original["start_tick"] / 120 and row["end"] == original["end_tick"] / 120
            assert row["label"] == original["label"]
            assert row["rgbIndices"] == original["rgb_indices"] and row["rfIndices"] == original["rf_indices"]
            assert set(row["scores"]) == set(MASKS)
            for mask in MASKS:
                assert set(row["scores"][mask]) == {model["id"] for model in MODELS}
                for model, key in [("fusion", "fusion_probability"), ("qwen", "probability")]:
                    assert row["scores"][mask][model] == by_model[model][(row["sampleId"], mask)][key]
                    checked += 1
        metadata = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,nb_frames,r_frame_rate,width,height,duration", "-of", "json", str(DEMO / clip["video"])]))["streams"][0]
        assert metadata["codec_name"] == "h264" and metadata["nb_frames"] == "300" and metadata["r_frame_rate"] == "30/1"
        assert metadata["width"] == metadata["height"] == 640
        assert float(metadata["duration"]) == 10
        assert clip["waveform"] == waveform(DEMO / clip["audio"]) and len(clip["waveform"]) == 300
    for asset in provenance["assets"]:
        assert digest(DEMO / asset["path"]) == asset["sha256"], f"Asset changed: {asset['path']}"
    external_clips = read_json(DATA / "external/examples.json")["clips"]
    assert [clip for clip in demo["clips"] if clip["dataset"] != "TRIDENT"] == external_clips
    assert digest(DATA / "external/provenance.json") == provenance["external"]["sha256"]
    subprocess.run([sys.executable, str(DEMO / "scripts/prepare_external.py"), "--verify-only"], check=True, capture_output=True)
    external_report = read_json(DATA / "external/verification.json")
    assert checked == expected_probabilities
    checked += external_report["exactCachedProbabilities"]
    report = {"passed": True, "releaseCommit": index["commit"], "clips": len(demo["clips"]),
              "windows": sum(len(clip["windows"]) for clip in demo["clips"] if clip["mediaKind"] == "video"),
              "observations": sum(len(clip["windows"]) for clip in demo["clips"]),
              "datasets": [pair["label"] for pair in PAIRS], "exactPublishedProbabilities": checked,
              "publishedBlobsVerified": len(index["files"]), "externalVerification": external_report,
              "assetsVerified": len(provenance["assets"]),
              "videos": sum(clip["mediaKind"] == "video" for clip in demo["clips"]),
              "snapshots": sum(clip["mediaKind"] == "snapshot" for clip in demo["clips"]),
              "tridentCameraFrames": sum(len(source["rgb_files"]) for source in selected_sources),
              "tridentRfFrames": sum(len(source["rf_files"]) for source in selected_sources)}
    write_json(DATA / "verification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--download-media", action="store_true", help="Fetch missing public original frames and verify their published MD5.")
    parser.add_argument("--cache-root", type=Path, default=Path(tempfile.gettempdir()) / "sentinel-demo-originals")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    verify() if args.verify_only else build(args.download_media, args.cache_root.resolve(), args.workers)
