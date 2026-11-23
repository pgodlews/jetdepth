# JetDepth: Depth Pro 2 on Jetson Orin with TensorRT

<p align="center">
  <a href="https://youtu.be/jc_cAkoPv6M"><img src="https://img.youtube.com/vi/jc_cAkoPv6M/maxresdefault.jpg" alt="Watch the JetDepth demo video on YouTube" width="100%"></a>
  <br>
  <a href="https://youtu.be/jc_cAkoPv6M"><strong>▶ Watch the demo on YouTube</strong></a>
</p>

Scripts that convert the Depth Pro 2 monocular depth model to NVIDIA TensorRT on Jetson Orin, check
the converted engine against PyTorch, and benchmark it.

This is an independent project. It is not affiliated with, endorsed by, or supported by Apple,
Meta, NVIDIA or Hugging Face. It contains no model weights and no upstream source code.

> **Depth from this model is an estimate, not a calibrated measurement.** Do not use it for safety,
> navigation, medical, legal or metrology purposes.

## What it provides

| File | Purpose |
|---|---|
| `convert.py` | Exports ONNX, compiles a TensorRT engine, and verifies it against PyTorch |
| `benchmark.py` | Measures a verified engine and writes one JSON result |
| `examples/run_depth.py` | Minimal runtime example, including temporal memory |
| `dp2trt/runtime.py` | `DepthEngine`, the small runtime class |
| `scripts/setup_jetson.sh` | Creates a project-local Python environment |
| `scripts/run_matrix.sh` | Converts and benchmarks a list of configurations |
| `scripts/temporal_stability.py` | Compares frame-to-frame stability of 1-frame and 4-frame engines |
| `scripts/make_results_table.py` | Builds the results tables from the JSON files |
| `tests/test_gate.py` | Checks that the verification gate refuses bad output |
| `results/` | Measured results for the two devices below |

## Tested environment

- Jetson Orin Nano Super 8 GB and Jetson AGX Orin 64 GB
- JetPack 6.2 (L4T 36.5.2), CUDA 12.6, TensorRT 10.3.0 from JetPack
- Python 3.10, torch 2.8.0 from the Jetson AI Lab index, NumPy below 2
- Depth Pro 2 at upstream commit `aa389cb`

Three version constraints matter:

- The official PyTorch aarch64 CUDA wheels do not include code for the Orin GPU. Use the Jetson index.
- torch 2.11 from that index needs `libcudss.so.0`, which was missing on the test system.
- Upstream declares Python 3.11 or later. JetPack 6.2's TensorRT bindings are for Python 3.10, so
  the scripts import upstream from a pinned checkout on Python 3.10. Everything reported here ran
  that way, but upstream does not declare it as supported.

## Setup

```bash
scripts/setup_jetson.sh work
export PYTHONPATH="$PWD" TORCH_HOME="$PWD/work/cache/torch"
PY=work/venv/bin/python
```

The script creates a virtual environment, links JetPack's TensorRT bindings into it, and checks out
upstream at the pinned commit. It makes no system changes.

**Weights.** This repository does not include or mirror them. On first use, upstream's loader
downloads them from Apple into `TORCH_HOME`. Read upstream's `LICENSE_MODEL` first: downloading or
using the weights means accepting it.

## Convert

```bash
$PY convert.py --model s --width 640 --height 480 --window 4 --precision fp16
```

| Option | Values |
|---|---|
| `--model` | `s`, `b`, `l`, `h` (H+) |
| `--width`, `--height` | Input size; the engine is fixed to it |
| `--window` | `1` (single frame) or `4` (three previous frames plus the current one) |
| `--precision` | S: `fp16` (default) or `fp32`. B, L and H+: `fp32` only |

Each stage runs in its own process and logs to `work/engines/<name>.log`. The engine keeps the
suffix `.unverified` until it passes. The exit status is 0 when verified, 2 when verification
failed, and 3 when a stage failed or was killed.

### Verification

An engine passes only if, on 16 consecutive frames of upstream's sample video, compared with
upstream's own float32 PyTorch stream:

- the median per-pixel relative difference in inverse depth is at most **1%**
- the 99th percentile is at most **5%**
- the correlation on every frame is at least **0.999**
- there are **no NaN or Inf** values, on those frames or on five synthetic stress frames
- for temporal engines, repeating the sequence after a reset reproduces the output

The frames used to fit the 16-bit rescaling are different from the frames used for scoring.
The synthetic frames are checked for NaN and Inf only, because they are far outside what the model
was designed for. Their accuracy is recorded in each conversion report.

## Run

```bash
$PY examples/run_depth.py --engine work/engines/s_640x480_w4_fp16.engine \
  --video work/ml-depth-pro-2/assets/example.mp4
```

```python
from dp2trt.frames import to_tensor
from dp2trt.runtime import DepthEngine

engine = DepthEngine("work/engines/s_640x480_w4_fp16.engine")
engine.reset()                                 # at the start of each video or camera session
inverse_depth = engine(to_tensor(rgb, "cuda")) # rgb: uint8 (H, W, 3); output: (1, H, W)
result = inverse_depth.cpu().numpy().copy()    # the output buffer is reused on the next call
```

Two details cause most integration bugs:

- **Contiguous memory.** TensorRT reads raw memory. A permuted, non-contiguous tensor produces
  scrambled input without any error, which looks like a conversion fault. `DepthEngine` rejects it.
- **Temporal memory.** TensorRT engines hold no state, so the memory is explicit input and output
  tensors. `DepthEngine` keeps two sets and swaps them after each step. Call `reset()` whenever the
  scene source changes, or old frames will influence the new ones.

The output is focal-normalised inverse depth. With a focal length `f_px` in pixels of the resized
frame, estimated depth in metres is `f_px / (width * inverse_depth)`.

## Benchmark

```bash
$PY benchmark.py --engine work/engines/s_640x480_w4_fp16.engine \
  --device-label "Jetson Orin Nano Super 8 GB" --output results/my-device/s_640x480_w4_fp16.json
```

Each JSON file records model-only FPS and latency (median, p95, p99), memory, power rails,
temperatures, GPU clock and load, power mode, software versions, accuracy against PyTorch, and
whether the GPU was shared during the run. The script only reads device state.

## Results

<!-- results:start -->
### Jetson AGX Orin 64 GB

Power mode: MAXN. L4T 36.5.2, TensorRT 10.3.0.

| Model | Input | Frames | Precision | FPS | Latency median / p95 (ms) | Difference from PyTorch, median / p99 | Power (W) | Peak GPU temp (C) | Engine (MB) | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| S | 320x240 | 1 | fp16 | 327.5 | 3.1 / 3.1 | 0.076% / 2.39% | 27.2* | 53.2 | 52 | verified |
| S | 320x240 | 1 | fp32 | 142.9 | 7.0 / 7.0 | 0.007% / 0.18% | 29.2* | 56.8 | 102 | verified |
| S | 320x240 | 4 | fp16 | 194.9 | 5.1 / 5.2 | 0.048% / 2.00% | 26.8* | 54.0 | 55 | verified |
| S | 320x240 | 4 | fp32 | 117.7 | 8.5 / 8.5 | 0.008% / 0.17% | 30.4* | 58.0 | 102 | verified |
| S | 640x480 | 1 | fp16 | 110.1 | 9.1 / 9.1 | 0.045% / 1.53% | 38.0* | 57.4 | 50 | verified |
| S | 640x480 | 1 | fp32 | 34.3 | 29.1 / 29.2 | 0.005% / 0.17% | 36.8* | 59.6 | 102 | verified |
| S | 640x480 | 4 | fp16 | 35.6 | 28.1 / 28.1 | 0.028% / 1.13% | 35.6* | 58.4 | 50 | verified |
| S | 640x480 | 4 | fp32 | 22.6 | 44.2 / 44.3 | 0.004% / 0.13% | 36.8* | 60.3 | 102 | verified |
| S | 848x480 | 1 | fp16 | 79.0 | 12.7 / 12.7 | 0.043% / 1.37% | 38.8* | 58.2 | 56 | verified |
| S | 848x480 | 1 | fp32 | 22.3 | 44.9 / 45.0 | 0.006% / 0.25% | 38.0* | 60.7 | 102 | verified |
| S | 848x480 | 4 | fp16 | 18.7 | 53.6 / 53.6 | 0.024% / 0.91% | 34.0* | 57.5 | 53 | verified |
| S | 848x480 | 4 | fp32 | 13.5 | 74.1 / 74.2 | 0.004% / 0.14% | 36.4* | 60.8 | 102 | verified |
| B | 640x480 | 1 | fp32 | 16.3 | 61.4 / 61.5 | 0.011% / 0.13% | 41.2* | 62.0 | 398 | verified |
| B | 640x480 | 4 | fp32 | 11.0 | 91.0 / 91.2 | 0.009% / 0.11% | 40.0* | 62.2 | 371 | verified |
| L | 640x480 | 1 | fp32 | 5.9 | 168.4 / 168.5 | 0.003% / 0.09% | 44.4* | 63.5 | 1277 | verified |
| L | 640x480 | 4 | fp32 | 4.2 | 236.3 / 236.5 | 0.002% / 0.09% | 43.2* | 63.9 | 1278 | verified |
| H+ | 640x480 | 1 | fp32 | 3.1 | 325.1 / 326.6 | 0.003% / 0.09% | 45.9* | 65.7 | 3331 | verified |
| H+ | 640x480 | 4 | fp32 | 2.2 | 459.7 / 461.5 | 0.003% / 0.09% | 43.2* | 65.8 | 3331 | verified |

### Jetson Orin Nano Super 8 GB

Power mode: MAXN_SUPER. L4T 36.5.2, TensorRT 10.3.0.

| Model | Input | Frames | Precision | FPS | Latency median / p95 (ms) | Difference from PyTorch, median / p99 | Power (W) | Peak GPU temp (C) | Engine (MB) | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| S | 320x240 | 1 | fp16 | 169.8 | 5.9 / 5.9 | 0.066% / 2.20% | 16.9 | 61.3 | 53 | verified |
| S | 320x240 | 1 | fp32 | 70.0 | 14.3 / 14.3 | 0.010% / 0.18% | 18.3 | 63.4 | 102 | verified |
| S | 320x240 | 4 | fp16 | 99.4 | 10.1 / 10.1 | 0.037% / 1.84% | 17.0 | 60.7 | 54 | verified |
| S | 320x240 | 4 | fp32 | 57.7 | 17.3 / 17.4 | 0.008% / 0.17% | 18.3 | 62.7 | 102 | verified |
| S | 448x336 | 4 | fp16 | 46.3 | 21.6 / 21.6 | 0.028% / 1.30% | 19.4 | 63.5 | 52 | verified |
| S | 512x384 | 4 | fp16 | 33.8 | 29.6 / 29.7 | 0.023% / 1.23% | 19.8 | 64.2 | 54 | verified |
| S | 640x480 | 1 | fp16 | 49.0 | 20.4 / 20.5 | 0.052% / 1.60% | 20.1 | 64.8 | 54 | verified |
| S | 640x480 | 1 | fp32 | 15.3 | 65.5 / 66.1 | 0.005% / 0.24% | 20.2 | 64.8 | 102 | verified |
| S | 640x480 | 4 | fp16 | 16.2 | 61.8 / 61.8 | 0.031% / 1.07% | 19.9 | 63.4 | 54 | verified |
| S | 640x480 | 4 | fp32 | 10.0 | 100.1 / 100.9 | 0.005% / 0.14% | 20.1 | 64.5 | 102 | verified |
| S | 848x480 | 1 | fp16 | 35.2 | 28.4 / 28.4 | 0.042% / 1.37% | 20.4 | 64.6 | 56 | verified |
| S | 848x480 | 1 | fp32 | 10.0 | 99.7 / 100.6 | 0.006% / 0.25% | 20.7 | 65.3 | 95 | verified |
| S | 848x480 | 4 | fp16 | 8.6 | 116.5 / 116.7 | 0.025% / 1.11% | 19.7 | 63.6 | 56 | verified |
| S | 848x480 | 4 | fp32 | 6.0 | 166.9 / 168.1 | 0.004% / 0.14% | 20.0 | 65.1 | 97 | verified |

Sustained runs:

| Model | Input | Frames | Precision | Duration (s) | FPS including upload | Latency median / p95 (ms) | Peak GPU temp (C) | GPU clock min (MHz) | Lowest free memory (MB) | GPU shared |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| S | 512x384 | 4 | fp16 | 300 | 33.6 | 29.7 / 29.7 | 70.9 | 1020 | 5324 | no |

Frame-to-frame stability on the sample video (lower is steadier; the camera moves, so part of each change is real):

| Model | Input | Precision | Frames | Change, median | Change, p95 | Scale jump, median | Scale jump, p95 | Scale jump, max |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S | 320x240 | fp16 | 1 | 0.88% | 13.61% | 2.98% | 7.07% | 9.32% |
| S | 320x240 | fp16 | 4 | 0.37% | 9.63% | 0.91% | 2.74% | 4.04% |
| S | 640x480 | fp16 | 1 | 0.62% | 12.41% | 2.36% | 6.27% | 6.88% |
| S | 640x480 | fp16 | 4 | 0.31% | 8.03% | 0.84% | 2.06% | 3.51% |

FPS is `1000 / median latency` for TensorRT execution alone. Power on the Orin Nano is the module input rail (`VDD_IN`). Power marked * is the GPU and SoC rail (`VDD_GPU_SOC`) only, so the two devices' power figures are not comparable. Every rail is in the JSON files.
<!-- results:end -->

## Building the 4-frame model in 16-bit

With TensorRT 10.3, building the unmodified 4-frame graph with the FP16 flag ends in a segmentation
fault inside `libnvinfer`. It reproduced on both devices and is not a memory limit: the 64 GB device
still had more than 38 GB available when it crashed. The same graph builds without the FP16 flag.

Bisecting the ONNX graph stopped at the first cross-frame attention product, where the attention
weights multiply the concatenated temporal values. `convert.py` replaces each such product with one
product per frame slot followed by a sum:

```text
W @ concat(V0, V1, ..., Vn)  =  W[slot 0] @ V0 + W[slot 1] @ V1 + ... + W[slot n] @ Vn
```

The rewrite is algebraically exact, and the rewritten graph builds and passes verification. Why the
original product crashes inside TensorRT is unknown, because the library is closed source.

## Limitations

**What failed or was not run**

- **B, L and H+ are 32-bit only for now.** Their 16-bit engines failed verification at 640x480 on
  the AGX, at both window sizes, with p99 differences from 6.3% to 45%, so `convert.py` refuses that
  combination. The failed conversion reports remain in `results/`. The fp32 engines passed, and the
  rescaled PyTorch models still matched the reference (p99 at most 0.034%), so the error appears in
  the compiled 16-bit engines. Its cause has not been identified.
- B, L and H+ were run only on the AGX and only at 640x480.
- 448x336 and 512x384 were run only on the Orin Nano, with 4 frames in fp16.
- No live camera pipeline was benchmarked, so capture, resize, display and end-to-end latency are
  not included in any figure.

**How to read the numbers**

- Agreement with PyTorch shows that the conversion is faithful. It says nothing about how accurate
  the depth is in the real world.
- The reference is upstream's float32 path. Upstream's default on GPUs is bfloat16 autocast.
- Accuracy was checked on one public sample video. The 16-bit rescaling is fitted to frames from
  that video, so very different content could behave differently. On synthetic stress frames no
  engine produced NaN or Inf, but agreement was lower (lowest frame correlation 0.942, for S
  640x480 with 4 frames in fp16).
- Most timings are single 30-second runs. One configuration was run for 5 minutes. Clocks were not
  pinned; the GPU clock stayed constant in every run.
- Orin shares memory between CPU and GPU and reports no per-process GPU memory. The JSON files
  record the drop in system memory on load and the process peak instead.
- The stability figures come from one clip with a moving camera, so they include real scene change.
- 848x480 stretches the 4:3 sample video slightly.
- Engines depend on the TensorRT version and GPU. Build them on the device that will run them.

## Licence and notices

**This project's own code and documentation** are under the MIT License in [`LICENSE`](LICENSE).
That licence does not cover Depth Pro 2 code or weights, DINOv3, TIMM, or any ONNX file or TensorRT
engine converted from the weights. Their own terms, described below, still apply.

**Depth Pro 2 code** is under Apple's licence in upstream's `LICENSE`. It is not copied here; the
scripts import it from a checkout pinned to commit `aa389cb`.

**Depth Pro 2 weights** are under the Apple Machine Learning Research Model License Agreement
(`LICENSE_MODEL` upstream), which limits use to non-commercial research and excludes commercial
exploitation, product development and use in commercial products or services. Those terms stay in
force for anyone using these scripts. ONNX files and TensorRT engines converted from the weights
should be treated as subject to the same terms, and are not distributed here. Anyone who does
redistribute the model or a derivative must include that agreement and the attribution notice it
requires, and must identify derivatives as modified.

**Third-party terms.** Depth Pro 2 builds on DINOv3 (Meta) and TIMM, whose licence terms also
apply. Upstream's `ACKNOWLEDGEMENTS` file lists them in full.

**Names.** Apple, Depth Pro, NVIDIA, Jetson, TensorRT and other names belong to their owners and
are used only to identify the software and hardware involved. Nothing here implies endorsement.

If you use the model in research, cite the Depth Pro 2 paper as described in the upstream README.
