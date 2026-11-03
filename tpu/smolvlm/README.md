# SmolVLM on the SG2000 TPU

SmolVLM-256M answers questions about the camera frame on one Milk-V Duo S, beside a
YOLO11n person detector. Both run on the TPU, with the ARM core in charge. One resident
program holds the whole VLM, and a page on the board takes questions from any browser on
the local network.

Measured on one Duo S in the ARM boot (Cortex-A53), with a 384 MiB CMA pool and
tpu-mlir 1.30.2, in November 2026.

## Results

### Accuracy

150 held-out VQA v2 "other" questions about people, on COCO val2014 images. The score is
VQA accuracy, with a human answer counted when the output contains it.

| Answers from | Runs on | Score |
|---|---|---:|
| YOLO11n boxes as JSON, SmolLM2-135M, no image | PC | 6.9 |
| SmolVLM-256M, float | PC | 50.9 |
| Encoder and decoder on the TPU, swapped in and out | Board | 47.1 |
| Resident build, everything on the board | Board | 48.5 |

The resident build minus float: -2.5 points, paired bootstrap 95% interval [-8.4, 3.2].
150 questions cannot tell the two apart.

### Time

Detector set to 15 fps. Time runs from the question, or from the detection for the
trigger path. Frame age is the age of the frame the answer is about.

| Path | Time to answer | Frame age | Detector |
|---|---|---|---|
| `camera/ask.sh`: load the models for each question | 18.4 s | 18.4 s | off |
| `resident/watch_fast.py`: a detection triggers two questions | 3.7 to 3.8 s first, 4.1 to 4.3 s both | same | 12.9 fps |
| `resident/live.py`, new frame every 10 s | 0.36 s median, 0.66 s at 90% | 8.9 s median | 12.5 fps |
| `resident/live.py`, new frame every 60 s | 0.49 s median, 1.09 s at 90% | 30.1 s median | 14.5 fps |
| `resident/live.py`, new frames back to back | 0.59 s median, 0.94 s at 90% | 6.5 s median | 9.6 fps |

- One new frame, beside the detector (medians): canvas 118 ms on the CPU, patch embedding
  34 ms, 12 SigLIP blocks 2758 ms, connector 68 ms, image prefill 448 ms. With the
  detector off, the 12 blocks take about 2.52 s.
- The decoder decodes 12.1 tok/s alone and 100 to 150 ms per token beside the detector.
- The detector's worst latency stays near 0.3 s. The TPU does not preempt, and the longest
  VLM call is a 205 ms block.
- A real person triggered the trigger path three times: both answers came 4.5, 4.8 and
  5.3 s after the detector first saw them.
- From a PC over Wi-Fi, four questions to `live.py` took 0.41 to 0.75 s round trip.

### Memory

The detector and the VLM take 335 MiB of the 384 MiB pool, and the detector with its
camera 14 MiB of that, with 3 camera buffers. About 9 MiB stay free.

## What made it fit and fast

- **CMA pool 320 to 384 MiB** (`/boot/armbianEnv.txt`, `cma=384M`).
- **Linear layers as 1x1 convolutions.** On block 6, the convolution build takes 200.9 ms
  and the MatMul build 388.6 ms (`encoder/export_block_mm.py`, `encoder/bench_mm.sh`).
  In BF16 the MatMul build wins, so the gain comes from INT8 convolution.
- **Attention one head at a time** (`encoder/export_grouped.py`). The shared activation
  buffer falls from 53.5 to 17.3 MB, and the outputs are the same bytes.
- **Block 5 split by channel** (`resident/export_split.py`). 15 of its 3072 fc1 channels
  exceed 8, up to 31. A BF16 branch takes them, and the rest runs in INT8: 608.8 to
  204.9 ms.
- **Decoder in INT8 weight products, 16 tokens per call, 128 cache slots, a one-row head.**
  The five fixed prompt tokens before the image carry activations INT8 cannot hold, so
  their K/V comes from a float prefix file (`VLM_TEXT_PREFIX=1`). That lifted the score
  from 31.3 to 50.4.
- **Patch embedding and the connector on the TPU** (`resident/export_pp.py`): 1.28 s to
  23 ms, and 2.27 s to 54 ms.
- **One resident program** (`resident/duosight.c`). Every model loads once. Loading the
  decoder first, the largest encoder block next, and sharing I/O buffers at once leaves
  the fewest gaps in the pool.
- **Answers on demand.** `bg` encodes a frame in steps and answers waiting questions
  between steps, about the last frame. It then prefills the image part of the prompt.
  The K/V kept stops at the start of a 16-token pass, so every answer is the same ids as
  a full prefill: 150 of 150 on the scored questions. Keeping every shared token gave a
  different answer to the same question asked twice.
- **The detector decides when to take a new frame** (`resident/live.py`). A new frame starts
  at once when the person count, stable for 3 detector frames, changes, or a person moves
  by 15% of the width. Otherwise it starts after 10 s with a person in view, or 60 s
  without one.
- **The tokenizer in C** (`resident/bpe.h`). It gives the same ids as `camera/tok.py` on 175
  test questions, in 28 ms for all of them.

## Layout

| Path | Contents |
|---|---|
| `encoder/` | The 12 SigLIP blocks: export, compile, the board runner `chain.c`, scoring, the MatMul comparison |
| `encoder/wsl/` | The per-block fixes of the final encoder build |
| `decoder/` | The text model: export, compile, the board runner `vlm.c`, scoring |
| `camera/` | The camera path: raw frame to canvas (`frame.c`), one question with models swapped (`ask.sh`), triggered questions (`watch.py`), the tokenizer in Python (`tok.py`) |
| `resident/` | The resident program `duosight.c`, the C tokenizer `bpe.h`, the question page `live.py`, the trigger path `watch_fast.py`, and their export, compile and check scripts |
| `quality/` | Float quality tests on the PC, with the rule fixed before the runs (`PREDICTIONS.md`) |

## Build and run the resident program

PC side, in the TPU-MLIR environment:

1. Encoder blocks with attention per head: `encoder/export_grouped.py ROOT 1 OLD_DIR OUT_DIR`,
   then `encoder/compile_g.sh DIR LAYER` for each block. Block 5:
   `resident/export_split.py ROOT LEVEL OLD_DIR OUT_DIR l05`, then
   `resident/compile_s.sh DIR l05`.
2. Patch embedding and connector: `resident/export_pp.py W_DIR OUT_DIR`, then
   `resident/compile_pp.sh patch "[[1,3,512,512]]"` and the same for `post`.
3. Decoder: `VLM_S=128 VLM_T=16 VLM_MIX2=all VLM_HEAD_T=1 VLM_TEXT_PREFIX=1
   decoder/export_vlm.py ROOT TEXT_DIR OUT_DIR`, then `resident/compile_t16.sh NAME` for
   `l00` to `l29` and `head`. `resident/stage_dec.sh OUT BUILD BOARD_DIR` copies them to
   the board.
4. Tables: `resident/make_vocab.py VOCAB.json vocab.bin` and
   `resident/make_bpe.py VOCAB.json bpe.bin`.

Board side:

5. `gcc -O2 -Iinc -o duosight duosight.c -lcviruntime -lm`
6. `python3 live.py --gain 0` serves the page on port 8080. `watch_fast.py` runs the
   trigger path instead.

`duosight` reads commands on stdin: `bg raw|rgb` (a frame in steps), `say TEXT` (a
question in words), `ask N ID...` (a prompt as ids), `raw`, `rgb` and `rows` (a frame at
once). Its header lists the replies and the environment variables.

## The detector

`live.py`, `watch_fast.py` and `camera/watch.py` run
[duos-detect](https://github.com/leafpmpmp/duos-detect) with two changes: SIGUSR1 saves the
next raw frame to a file, and `DETECT_BUFFERS` sets the number of camera buffers (3 here).
They expect the build at `/root/vlm/duos-detect-tap`. The patch is not in this
repository, because duos-detect is AGPL-3.0 and this repository is MIT.

## Limits

- Answers on demand are about a frame at least 4.5 s old. A person who just stepped in
  appears in answers about 4.5 s after the detector sees them, longer if a frame was
  already in steps.
- Three real-person events only. "What is the person wearing?" was answered wrong twice.
- The camera has no automatic exposure: daylight needs `--gain 0`, a dim room the default
  940.
- The questions assume a person, so a frame without one gets invented answers.
- About 9 MiB of the pool stay free. More services on the board can make the load fail.
- The scripts keep the paths of the machine they ran on (`D:/duosight-eval`, and
  `~/duosight-tpu` in WSL), and the scoring scripts read question files that are not in
  this branch.
