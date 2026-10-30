#!/usr/bin/env python3
"""Build four external examples from pinned V2 caches and public source media.

Run with --download once, then rerun offline to verify and regenerate the pack.
Requires NumPy, SciPy, Matplotlib, Pillow, ffmpeg and ffprobe. No inference runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
DATA = DEMO / "data/external"
ASSETS = DEMO / "assets/external"
COMMIT = "7fc465480f4f5b1ad348ffef93c744712e06b098"
DATASET_COMMIT = "e7a6eaf9760fa3f9ecc37d09cd11a8cd20be90dd"
RFVS_URL = "https://drive.usercontent.google.com/download?id=1ANz2rfjkWlMiWKkTb31dbqQhCApGPFtK&export=download&confirm=t"
RFVS_BYTES = 5343324308
PINNED = {
    "research/external/rfvs/predictions.json": "09d9939e923d53a635625c7068b17c45e3cb8f88",
    "research/external/halmstad-video/predictions.json": "4e3564c3fc5822a3d14bbfdb55f85b87e4dc9f7d",
    "src/drone_presence/adapters/rfvs.py": "b4db1cdfcb0c6205360e1b1a74c91d42b1790d2e",
    "src/drone_presence/adapters/halmstad.py": "818bce21d4d1df27efbcbc23c660e106e11c9a59",
    "scripts/setup/download_external.py": "f1a48299322eb8d218096b9c868b350f060b9887",
    "data/audit/rfvs/archive-manifest.json": "2445b61bc47f3300984f1f4141f200a7afb884c9",
}
RFVS = [
    {"id": "rfvs-drone", "stem": "AIR2s_58296527", "label": 1, "title": "RFVS drone snapshot",
     "description": "A released drone RGB still paired with its 30 ms RF capture. Both models detect presence with camera and RF.",
     "caption": "One genuine RFVS sample. The single RGB image fills all eight model frame slots. Four RF panels cover consecutive 7.5 ms intervals."},
    {"id": "rfvs-background", "stem": "house_background_124157_73", "label": 0, "title": "RFVS house background",
     "description": "A released background RGB still and its 30 ms RF capture. Qwen detector rejects the background while Fusion classifier raises a false alarm.",
     "caption": "The released label file is empty. With camera and RF, Qwen detector rejects this background and Fusion classifier raises a false alarm."},
]
HALMSTAD = [
    {"id": "halmstad-drone", "recording": "V_DRONE_001", "label": 1, "title": "Halmstad drone video",
     "gitBlobSha1": "08a75180c1daf4b383a8234f97a1d8bdf4ec9012", "time": 4,
     "description": "A 10 s excerpt of a released camera recording labelled drone. Both models detect presence throughout these windows.",
     "caption": "Both cached models detect drone presence in this camera-only window. Halmstad video has no paired RF or audio."},
    {"id": "halmstad-background", "recording": "V_BIRD_022", "label": 0, "title": "Halmstad bird video",
     "gitBlobSha1": "f9bbadc6f83f3e8506ed017ff245d13013d52162", "time": 6,
     "description": "A 10 s excerpt of a released bird recording. At 5-6 s, Qwen detector rejects the bird while Fusion classifier raises a false alarm.",
     "caption": "The released class is bird. In the 5-6 s window, Fusion classifier scores 96.18% and Qwen detector scores 0.67%."},
]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def rel(path):
    return str(path.relative_to(DEMO))


def pinned_path(path):
    return DATA / "published" / path.replace("/", "__")


def get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120) as response:
        limit = 16 * 1024 * 1024
        if int(response.headers.get("Content-Length", "0")) > limit:
            raise ValueError("Response exceeds the selected source size bound")
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Response exceeds the selected source size bound")
        return response.status, dict(response.headers), raw


def acquire_pinned(item):
    path, expected = item
    local = pinned_path(path)
    if not local.exists():
        _, _, raw = get(f"https://raw.githubusercontent.com/xena1t/SENTINEL/{COMMIT}/{path}")
        assert git_sha(raw) == expected, path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(raw)
    assert git_sha(local.read_bytes()) == expected, path


def acquire_rfvs_member(row):
    local = DATA / "source/rfvs" / row["name"]
    if local.exists():
        raw = local.read_bytes()
        assert len(raw) == row["size"] and zlib.crc32(raw) == row["crc"]
        return
    start = row["offset"]
    end = min(RFVS_BYTES - 1, start + 30 + len(row["name"].encode()) + 4096 + row["compressed"] - 1)
    status, headers, body = get(RFVS_URL, {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    normalized = {key.lower(): value for key, value in headers.items()}
    assert status == 206 and normalized["content-range"] == f"bytes {start}-{end}/{RFVS_BYTES}"
    assert len(body) == end - start + 1
    signature, _, flags, method, _, _, crc, compressed, size, nlen, xlen = struct.unpack_from("<4s5H3L2H", body)
    assert signature == b"PK\x03\x04" and not flags & (1 | 64) and method == row["method"]
    assert body[30:30 + nlen].decode() == row["name"] and xlen <= 4096
    extra = body[30 + nlen:30 + nlen + xlen]
    position = 0
    while position < len(extra):
        tag, length = struct.unpack_from("<HH", extra, position)
        position += 4
        if tag == 1:
            original_size, original_compressed = struct.unpack_from("<QQ", extra, position)
            assert (original_size, original_compressed) == (row["size"], row["compressed"])
            size, compressed = original_size, original_compressed
        position += length
    assert (size, compressed) == (row["size"], row["compressed"]) or flags & 8
    assert crc in (0, row["crc"])
    offset = 30 + nlen + xlen
    packed = body[offset:offset + row["compressed"]]
    raw = zlib.decompress(packed, -15) if method == 8 else packed
    assert len(raw) == row["size"] and zlib.crc32(raw) == row["crc"]
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(raw)
    print(f"Acquired {row['name']} ({len(raw)} bytes)", flush=True)


def download():
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(acquire_pinned, PINNED.items()))
    manifest = json.loads(pinned_path("data/audit/rfvs/archive-manifest.json").read_text())
    names = {name for example in RFVS for name in (f"images/test/{example['stem']}.jpg", f"labels/test/{example['stem']}.txt", f"RF_raw/test/{example['stem']}")}
    selected = [row for row in manifest if row["name"] in names]
    assert len(selected) == 6
    for row in selected:
        acquire_rfvs_member(row)
    for example in HALMSTAD:
        path = DATA / "source/halmstad" / f"{example['recording']}.mp4"
        if not path.exists():
            _, _, raw = get(f"https://raw.githubusercontent.com/DroneDetectionThesis/Drone-detection-dataset/{DATASET_COMMIT}/Data/Video_V/{path.name}")
            assert git_sha(raw) == example["gitBlobSha1"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        assert git_sha(path.read_bytes()) == example["gitBlobSha1"]


def window(rows, start, end, rf_indices):
    assert len({row["sample_id"] for row in rows}) == 1
    assert len({row["label"] for row in rows}) == 1
    return {"sampleId": rows[0]["sample_id"], "start": start, "end": end, "label": rows[0]["label"],
            "scores": {row["mask"]: {"fusion": row["fusion_probability"], "qwen": row["probability"]} for row in rows},
            "rfIndices": rf_indices}


def package():
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sentinel-external-matplotlib")
    sys.dont_write_bytecode = True
    receipts = []
    for path, expected in PINNED.items():
        local = pinned_path(path)
        assert git_sha(local.read_bytes()) == expected
        receipts.append({"path": path, "local": rel(local), "gitBlobSha1": expected, "sha256": sha(local),
                         "url": f"https://github.com/xena1t/SENTINEL/blob/{COMMIT}/{path}"})
    cache = {name: json.loads(pinned_path(f"research/external/{name}/predictions.json").read_text()) for name in ("rfvs", "halmstad-video")}
    spec = importlib.util.spec_from_file_location("released_rfvs", pinned_path("src/drone_presence/adapters/rfvs.py"))
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    manifest = json.loads(pinned_path("data/audit/rfvs/archive-manifest.json").read_text())
    clips, sources = [], []
    for example in RFVS:
        folder = ASSETS / example["id"]
        folder.mkdir(parents=True, exist_ok=True)
        stem = example["stem"]
        original = DATA / "source/rfvs"
        image, label, rf = original / f"images/test/{stem}.jpg", original / f"labels/test/{stem}.txt", original / f"RF_raw/test/{stem}"
        assert int(bool(label.read_text().strip())) == example["label"]
        for path in (image, label, rf):
            name = str(path.relative_to(original))
            row = next(row for row in manifest if row["name"] == name)
            assert len(path.read_bytes()) == row["size"] and zlib.crc32(path.read_bytes()) == row["crc"]
            sources.append({"clipId": example["id"], "local": rel(path), "archiveMember": name, "sha256": sha(path), "bytes": path.stat().st_size,
                            "archive": RFVS_URL, "archiveBytes": RFVS_BYTES, "archiveManifest": row,
                            "httpRange": f"bytes={row['offset']}-{row['offset'] + 30 + len(name.encode()) + 4096 + row['compressed'] - 1}"})
        shutil.copyfile(image, folder / "poster.jpg")
        frames = adapter.spectrogram_images(rf)
        for i, frame in enumerate(frames):
            frame.save(folder / f"rf-{i + 1:02d}.png")
        rows = [row for row in cache["rfvs"]["rows"] if row["sample_id"] == f"rfvs:{stem}"]
        assert {row["mask"] for row in rows} == {"100", "010", "110"}
        assert all(row["label"] == example["label"] for row in rows)
        clips.append({"id": example["id"], "title": example["title"], "description": example["description"],
                      "label": example["label"], "dataset": "RFVS", "mediaKind": "snapshot", "availableMask": "110", "supportedMasks": ["110", "100", "010"],
                      "duration": .03, "windowSeconds": .03, "hopSeconds": None, "sourceOffsetSeconds": 0,
                      "video": None, "poster": rel(folder / "poster.jpg"), "audio": None, "waveform": [],
                      "rfFrames": [{"time": i * .0075, "start": i * .0075, "end": (i + 1) * .0075, "src": rel(folder / f"rf-{i + 1:02d}.png")} for i in range(4)],
                      "windows": [window(rows, 0, .03, [1, 2, 3, 4])],
                      "cue": {"time": .03, "mask": "110", "caption": example["caption"]},
                      "source": {"recording": stem, "labelBasis": "Released exact-stem YOLO label file, nonempty means drone.",
                                 "temporalBasis": "One still image and one 30 ms RF sample. This is not a video window.", "selection": "Purposefully selected illustration; not a performance estimate."}})
    for example in HALMSTAD:
        folder = ASSETS / example["id"]
        folder.mkdir(parents=True, exist_ok=True)
        original = DATA / "source/halmstad" / f"{example['recording']}.mp4"
        assert git_sha(original.read_bytes()) == example["gitBlobSha1"]
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames,avg_frame_rate,duration", "-of", "json", str(original)]))["streams"][0]
        assert probe["avg_frame_rate"] == "30/1" and int(probe["nb_frames"]) // 30 == 10
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(original), "-frames:v", "300", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(folder / "camera.mp4")], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(int(example["time"])), "-i", str(original), "-frames:v", "1", "-q:v", "2", str(folder / "poster.jpg")], check=True)
        rows = [row for row in cache["halmstad-video"]["rows"] if row["recording"] == example["recording"]]
        assert len(rows) == 10 and all(row["label"] == example["label"] and row["mask"] == "100" for row in rows)
        windows = []
        for second in range(10):
            selected = [row for row in rows if row["sample_id"] == f"halmstad-video:{example['recording']}:s{second:02d}"]
            assert len(selected) == 1
            windows.append(window(selected, second, second + 1, []))
        sources.append({"clipId": example["id"], "local": rel(original), "sha256": sha(original), "gitBlobSha1": example["gitBlobSha1"], "bytes": original.stat().st_size,
                        "videoProbe": probe, "excerptStartFrame": 0, "excerptEndFrameExclusive": 300,
                        "url": f"https://github.com/DroneDetectionThesis/Drone-detection-dataset/blob/{DATASET_COMMIT}/Data/Video_V/{original.name}"})
        clips.append({"id": example["id"], "title": example["title"], "description": example["description"], "label": example["label"],
                      "dataset": "Halmstad", "mediaKind": "video", "availableMask": "100", "supportedMasks": ["100"], "duration": 10,
                      "windowSeconds": 1, "hopSeconds": 1, "sourceOffsetSeconds": 0,
                      "video": rel(folder / "camera.mp4"), "poster": rel(folder / "poster.jpg"), "audio": None, "rfFrames": [], "waveform": [], "windows": windows,
                      "cue": {"time": example["time"], "mask": "100", "caption": example["caption"]},
                      "source": {"recording": example["recording"], "labelBasis": "Released filename class, DRONE=1, BIRD=0.",
                                 "temporalBasis": f"First 300 frames of the original {probe['nb_frames']}-frame 30 fps video, source offset zero. Ten nonoverlapping 1 s released windows. Eight model frames per window at offsets 0,4,8,12,17,21,25,29. No RF or paired audio.",
                                 "selection": "Purposefully selected illustration; not a performance estimate."}})
    examples = {"version": 1, "experiment": "Version 2", "sourceCommit": COMMIT, "threshold": .5, "selection": "Purposefully selected illustrations, not a performance estimate.", "clips": clips}
    dump(DATA / "examples.json", examples)
    dump(DATA / "provenance.json", {"repository": "xena1t/SENTINEL", "commit": COMMIT, "publishedFiles": receipts, "sources": sources,
                                   "modelProvenance": {key: value["provenance"] for key, value in cache.items()},
                                   "media": [{"local": rel(path), "bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(ASSETS.rglob("*")) if path.is_file() and not path.name.startswith("._")]})
    verify()


def verify():
    examples = json.loads((DATA / "examples.json").read_text())
    provenance = json.loads((DATA / "provenance.json").read_text())
    for item in provenance["publishedFiles"] + provenance["sources"] + provenance["media"]:
        local = DEMO / item["local"]
        assert sha(local) == item["sha256"], item["local"]
        if "gitBlobSha1" in item:
            assert git_sha(local.read_bytes()) == item["gitBlobSha1"], item["local"]
    rows = {}
    for dataset in ("rfvs", "halmstad-video"):
        cache = json.loads(pinned_path(f"research/external/{dataset}/predictions.json").read_text())
        for row in cache["rows"]:
            rows[row["sample_id"], row["mask"]] = row
    scores = windows = 0
    assert len(examples["clips"]) == 4
    for clip in examples["clips"]:
        assert not clip["audio"] and not clip["waveform"]
        assert clip["supportedMasks"] == (["110", "100", "010"] if clip["mediaKind"] == "snapshot" else ["100"])
        assert len(clip["windows"]) == (1 if clip["mediaKind"] == "snapshot" else 10)
        if clip["mediaKind"] == "snapshot":
            assert clip["video"] is None and clip["duration"] == .03
            assert len(clip["rfFrames"]) == 4
            assert (clip["windows"][0]["start"], clip["windows"][0]["end"]) == (0, .03)
            assert clip["windows"][0]["rfIndices"] == [1, 2, 3, 4]
        else:
            assert not clip["rfFrames"] and clip["windowSeconds"] == clip["hopSeconds"] == 1
            assert clip["duration"] == 10 and clip["sourceOffsetSeconds"] == 0
            for second, w in enumerate(clip["windows"]):
                assert w["sampleId"].endswith(f":s{second:02d}")
                assert (w["start"], w["end"]) == (second, second + 1)
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames,avg_frame_rate,duration", "-of", "json", str(DEMO / clip["video"])]))["streams"][0]
            assert probe["nb_frames"] == "300" and probe["avg_frame_rate"] == "30/1" and float(probe["duration"]) == 10
        if clip["id"] == "halmstad-background":
            completed = [w for w in clip["windows"] if w["end"] <= clip["cue"]["time"]]
            assert completed[-1]["sampleId"] == "halmstad-video:V_BIRD_022:s05"
        for w in clip["windows"]:
            windows += 1
            assert set(w["scores"]) == set(clip["supportedMasks"])
            for mask, values in w["scores"].items():
                row = rows[w["sampleId"], mask]
                assert w["label"] == row["label"] == clip["label"]
                assert values == {"fusion": row["fusion_probability"], "qwen": row["probability"]}
                scores += 2
    result = {"passed": True, "sourceCommit": COMMIT, "clips": 4, "releasedSamplesOrWindows": windows, "exactCachedProbabilities": scores,
              "verifiedPublishedFiles": len(provenance["publishedFiles"]), "verifiedRawSources": len(provenance["sources"]),
              "verifiedMedia": len(provenance["media"]), "noInference": True, "selection": "Purposefully selected illustrations, not a performance estimate."}
    dump(DATA / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    if args.verify_only:
        verify()
    else:
        package()
