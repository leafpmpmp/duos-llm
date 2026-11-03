# SmolVLM on the SG2000: results and data

Every experiment behind `tpu/smolvlm`, in the order they ran, with the data to check
them. All board runs: one Milk-V Duo S (SG2000) in the ARM boot (Cortex-A53), tpu-mlir
1.30.2, 2026-11-01 to 2026-11-03. The detector is the YOLO11n person model at
448x320 from duos-detect. The camera is a Raspberry Pi Camera Module 3 (IMX708).

## How the scores are measured

- **Questions.** VQA v2 val2014 questions that name a person (man, woman, person, people,
  boy, girl, child, player, ...), on COCO images where YOLO11n finds a person at
  confidence 0.4. One question per image.
- **Sample** (seed 42): 150 "other", 100 "yes/no" and 50 "number" questions. Used for the
  test whose rule was fixed in advance (`quality/PREDICTIONS.md`) and for the exploratory
  runs.
- **Held-out set** (seed 43): 150 "other" questions on images that are not in the sample.
  Every board score uses it.
- **Metric.** VQA accuracy: min(matches / 3, 1), averaged over leaving out each of the 10
  human answers. A human answer matches when it appears as whole words in the output.
  This lenient match was fixed before the full run, because SmolLM2 answers in sentences.
- **Prompt.** "QUESTION Answer using a single word or phrase.", at most 24 new tokens.
- **Differences** come with paired bootstrap 95% intervals (10000 resamples).
- **JSON + LLM baseline.** SmolLM2-135M-Instruct gets the YOLO11n boxes as JSON, as
  fractions of the frame, and no image.

## Data

| File | Contents |
|---|---|
| `data/sample-300.json` | The 300 sampled questions, their 10 human answers, the YOLO11n boxes, and the answers of every PC condition |
| `data/holdout-150.json` | The 150 held-out questions and the answers of every build |
| `data/trigger-events.jsonl` | Each event of the trigger path on the board: steps, times, answers, boxes |
| `data/live.jsonl` | Each frame and each question of `resident/live.py` on the board |

Answers sit under `out`, one key per condition. The score function is `vqa()` in
`resident/score_fast.py`, and it gives these scores on the "other" questions:

| `holdout-150.json` key | Condition | Score |
|---|---|---:|
| `json_llm` | JSON + LLM, no image | 6.9 |
| `up256_bilinear`, `up256_nearest` | Float, 256 px, features resampled to 64 tokens | 22.5, 25.6 |
| `up320_bilinear` | Float, 320 px, resampled | 32.4 |
| `up384_bilinear` | Float, 384 px, resampled | 41.0 |
| `native512` | Float, 512 px | 50.9 |
| `tpu512` | Encoder on the TPU, final build; float decoder | 51.7 |
| `tpu_decoder` | Encoder and decoder on the TPU, swap build | 47.1 |
| `e2e_t16p` | Resident decoder with the float prefix, block 5 with BF16 fc2 | 50.4 |
| `e2e_split` | Resident build, final | 48.5 |

| `sample-300.json` key | Condition | Score |
|---|---|---:|
| `json_llm` | JSON + LLM | 12.5 |
| `full256`, `crop256` | Whole image, largest person: 256 px, 16 tokens | 19.9, 19.7 |
| `full512`, `crop512` | Same at 512 px, 64 tokens | 47.6, 44.6 |
| `official512` | The model's own processor, 512 px | 50.3 |
| `masked512` ... `masked256` | Padding masked, 64 to 16 tokens | 45.9 to 14.9 |
| `pixels256_tokens64` | 256 px of detail, upscaled to 512 px | 41.9 |
| `maskcrop512`, `maskcrop384` | Largest person, padding masked | 36.7, 23.1 |
| `native512` | Same preprocessing as `holdout-150.json` | 53.7 |

The questions and human answers come from VQA v2, under CC BY 4.0. The images are not
included; `image` gives the COCO file name. Times in the `.jsonl` files are the board's
clock (UTC). The first lines of `trigger-events.jsonl` record failed loads of the swap
build, before the page cache fix described below.

## 1. Is a VLM worth porting? Float tests on the PC

### The test with a rule fixed in advance

Sample, "other" questions:

| Condition | Score |
|---|---:|
| JSON + LLM | 12.5 |
| Whole image, 256 px | 19.9 |
| Largest person, 256 px | 19.7 |
| Whole image, 512 px | 47.6 |
| Largest person, 512 px | 44.6 |

- The crop at 256 px minus JSON + LLM: 7.1 points, [0.6, 13.6]. The rule asked for 10.
- The whole image at 512 px minus the crop at 256 px: 27.9 points, [20.0, 36.1]. The rule
  allowed at most 10.
- All three predictions failed: the VLM needs 512 px and the whole frame.
- Yes/no questions need their own prompt. With "Answer yes or no." the model scores 67.7
  at 256 px, 72.3 at 384 px and 76.5 at 512 px; always "no" scores 65.6.

### Exploratory runs, same sample

| Encoder input | Image tokens | Score |
|---|---:|---:|
| 512 px | 64 | 45.9 to 53.7 |
| 448 px | 49 | 33.4 |
| 384 px | 36 | 25.3 |
| 320 px | 25 | 18.1 |
| 256 px | 16 | 14.9 |
| 256 px of detail, upscaled to 512 px | 64 | 41.9 |
| 384 px, patch features resampled | 64 | 37.7 |
| 320 px, patch features resampled | 64 | 34.7 |
| 256 px, patch features resampled | 64 | 24.9 |

- The loss at small inputs comes mostly from the token count, not from lost detail.
- Masking the padding in the encoder against black padding: -1.7 points, [-8.5, 5.2].
- Crops did not beat the full frame for any size of the largest person:

| Share of the frame | n | Full 512 | Crop 512 | Full 384 | Crop 384 |
|---|---:|---:|---:|---:|---:|
| Under 10% | 29 | 49.7 | 35.2 | 37.2 | 25.9 |
| 10% to 30% | 67 | 52.4 | 38.1 | 26.1 | 18.7 |
| Over 30% | 54 | 35.7 | 35.7 | 18.0 | 27.2 |

### Held-out check

| Condition | Score | Gain over JSON + LLM | Loss from 512 px |
|---|---:|---|---|
| JSON + LLM | 6.9 | | |
| 512 px, 64 tokens | 50.9 | 44.0 | |
| 384 px, resampled to 64 tokens | 41.0 | 34.1 [26.3, 41.7] | 9.9 [3.3, 16.7] |
| 320 px, resampled | 32.4 | 25.5 [18.3, 32.9] | 18.5 [11.1, 26.3] |
| 256 px, resampled | 22.5 | 15.5 [9.3, 22.3] | 28.5 [20.5, 36.4] |

512 px stays the target. Every later build runs at 512 px with 64 image tokens.

## 2. The encoder on the TPU

### One block, by precision

| Build | 384 px | 512 px | Cosine to float |
|---|---:|---:|---:|
| All INT8 | 84.9 ms | 193 ms | 0.23 |
| mix-a: BF16 norms, attention and residual | 89.6 ms | 201 ms | 0.995 |
| mix-b: mix-a with BF16 GELU | 115 ms | 247 ms | 0.995 to 0.996 |
| All BF16 | 217 ms | 686 ms | 0.99998 |

### 1x1 convolution against MatMul

Block 6 at 512 px, built both ways from the same weights and the same 32 calibration
samples (`encoder/export_block_mm.py`, `encoder/bench_mm.sh`). 20 runs each, two rounds
alternating, nothing else on the TPU; the rounds agreed within 0.2 ms.

| Build | 1x1 conv | MatMul | Conv faster by |
|---|---:|---:|---:|
| All INT8 | 192.5 ms (46 GMAC/s) | 384.9 ms (23 GMAC/s) | 2.0 times |
| mix-a | 200.9 ms | 388.6 ms | 1.93 times |
| All BF16 | 685.4 ms (12.9 GMAC/s) | 598.8 ms (14.8 GMAC/s) | 0.87 times |

- Accuracy is the same both ways: mix-a cosine 0.995, all BF16 0.99998 and 0.99999.
- Over 12 blocks the rewrite saves about 2.3 s per image.
- In BF16 the MatMul build is faster, so the gain comes from INT8 convolution.

### Where the error came from

Block 6, one op moved to BF16 at a time:

| Moved to BF16 | delta-cos | typical-cos | Time |
|---|---:|---:|---:|
| q, k, v | 0.958 | 0.981 | 233 ms |
| o | 0.958 | 0.981 | 211 ms |
| fc1 | 0.961 | 0.981 | 245 ms |
| GELU | 0.960 | 0.982 | 247 ms |
| fc2 | 0.984 | 0.995 | 567 ms |
| All but fc2 | 0.969 | 0.983 | 327 ms |

fc2 causes most of the error: one INT8 scale covers its whole output, and a few channels
reach 50 or more. Dividing each fc2 output channel by its own maximum, and restoring it
with one BF16 depthwise convolution, gives block 6 delta-cos 0.978 and typical-cos 0.995
at 202 ms. Blocks 10 and 11 keep BF16 GELU; block 5 also needs BF16 fc2.

### First and final 12-block builds

Float decoder on the PC; the same CPU scoring path for every row.

| Measure | First build | Final build |
|---|---:|---:|
| Encoder time, 150 images | 2996 ms | 2937 ms (min 2921, max 2962) |
| Token cosine after the final LayerNorm | 0.698 | 0.927 |
| Worst 5% of tokens, same cosine | 0.441 | 0.769 |
| Same answer as float | 37 of 150 | 120 of 150 |
| Score | 21.3 | 51.7 |

- Equalized, with block 5 not yet fixed: 48.7.
- Float minus the final build: -0.8 points, [-5.4, 3.7].
- The raw block cosine is dominated by a few huge values, and it fell in the middle build
  while the score rose. It is no use as a metric; the cosine after the final LayerNorm is.
- Giving all 12 blocks the same two I/O buffers changes no output bit and saves about
  65 MiB.

## 3. The decoder on the TPU, swap build

SmolVLM-256M's text model in the `q4` config of `tpu/smollm`: 20 BF16 layers, 10 MIX2
layers and a MIX2 head, 512 cache slots, 4 tokens per call. Encoder: the final build.

| Encoder | Decoder | Score |
|---|---|---:|
| Float | Float | 50.9 |
| TPU | Float | 51.7 |
| TPU | TPU | 47.1 |
| JSON + LLM | | 6.9 |

- TPU decoder minus float decoder, same features: -4.7 points, [-8.3, -1.3].
- Both on the TPU minus all float: -3.9 points, [-9.1, 1.4].
- Both on the TPU minus JSON + LLM: 40.1 points, [32.3, 48.0].

| Measure | YOLO off | YOLO 15 fps | YOLO 30 fps |
|---|---:|---:|---:|
| Encoder, one image | 2.94 s | 3.12 s | 3.20 s |
| Decoder load | 3.5 s | 3.5 s | 3.8 s |
| Prefill, median | 1.64 s | 2.19 s | 3.32 s |
| Decode | 12.8 tok/s | 9.3 tok/s | 6.4 tok/s |
| Detector frames lost beside the decoder | | 0 | 0 |
| Worst detector latency beside the decoder | | 58 ms | 66 ms |

Two problems on the way:

- Turning on `delta` before the float run lost the residual stream, so the calibration
  data and the prefix were wrong and the answers were nonsense.
- K/V left from one prompt changed the answer to the next: a masked cache slot still gets
  a tiny weight on the TPU. Every program that keeps the decoder loaded empties the used
  slots before a new prompt.

## 4. Memory: why the first builds swapped

320 MiB pool, page cache cleared during each run:

| State | Free CMA |
|---|---:|
| Empty | about 299 MiB |
| Detector and camera | about 270 MiB |
| Encoder alone | about 75 MiB |
| Decoder alone | 59 to 60 MiB |
| Detector and encoder | 74 to 76 MiB |
| Detector and decoder | 48 to 49 MiB |

- Alone, the encoder uses about 224 MiB, the decoder 236 to 241 MiB, the detector and
  camera about 30 MiB. All three come to about 445 to 490 MiB, against a limit near
  396 MiB with the ramoops overlay.
- So the encoder and the decoder took turns, with the detector running. Beside YOLO11n at
  15 fps the encoder took 3.12 s, the decoder 3.7 s to load, the first token about 9.0 s
  as a sum of parts, and the worst detector latency was 665 ms.

## 5. Camera to answer, swap build

`camera/ask.sh`: every step on the board, as separate programs. Detector off, one
question, a dim room, three runs within 0.1 s.

| Step | Where | Time |
|---|---|---:|
| Capture one frame, long exposure | Sensor | 1.53 s |
| Raw to the 512 px canvas | CPU | 0.21 s |
| Patch embedding | CPU | 1.28 s |
| 12 blocks: load about 2.9 s, run 2.93 s | TPU | 5.8 s |
| Final LayerNorm and connector | CPU | 2.27 s |
| Tokenize, in Python | CPU | 1.0 s |
| Decoder: load 3.5 s, prefill 1.6 s, decode 0.16 s | TPU | 5.3 s |
| Ids to text, in Python | CPU | 1.0 s |
| Total | | 18.4 s |

- Checks against the PC: patch embedding within 3e-6, LayerNorm and connector within
  8e-5, the Python tokenizer the same on 454 prompts.
- `camera/watch.py` runs the detector at 5 fps and asks after a person stays for 3
  frames. One forced event, two questions: 20.4 s from the trigger to both answers. The
  detector kept 4.9 fps, lost 2 frames, worst latency 0.59 s.
- At detector threshold 0.4, a jacket scored 0.50 for 3 frames after 19 minutes, and the
  VLM answered "Cooking." and "Person." about nobody. The threshold is now 0.6. At 0.6, 2
  hours and 35999 frames fired nothing: 5.0 fps, no frames lost, worst latency 43 ms.

## 6. The resident build

Goal: the detector and the VLM loaded together, answers within 5 s of a detection.

- **CMA pool 320 to 384 MiB** (`cma=384M`).
- **Patch embedding and the connector on the TPU**, in BF16:

| Step | CPU before | TPU now | Cosine to float |
|---|---:|---:|---:|
| Patch embedding | 1.28 s | 22.7 ms | 0.99998 |
| LayerNorm and connector | 2.27 s | 54 ms | 0.999998 |

- **Attention in groups of heads** (block 6):

| Heads per group | Shared buffer | Time |
|---|---:|---:|
| 12 | 53.5 MB | 201.9 ms |
| 6 | 33.0 MB | 202.3 ms |
| 3 | 22.0 MB | 202.3 ms |
| 1 | 17.3 MB | 202.4 ms |

  With one head per group, the 12-block output is the same bytes. The 12 blocks take 201
  to 205 MiB of CMA instead of 216, and 186 to 188 MiB when the largest block loads first.

- **Block 5 split by channel.** 15 of its 3072 fc1 channels exceed 8 (the largest 30.8,
  the median 3.75). A BF16 branch takes them, and the rest runs in INT8:

| Block 5 | BF16 fc2 | Split |
|---|---:|---:|
| Cosine to float | 0.996331 | 0.999131 |
| Time on the board | 608.8 ms | 204.9 ms |
| Weights, commands, shared buffer | 9.5, 7.8, 20.4 MB | 7.2, 5.2, 17.3 MB |

- **Decoder in INT8 weight products** in all 30 layers, 16 tokens per call, 128 cache
  slots, a head of one row. The five fixed tokens before the image carry activations INT8
  cannot hold, so their K/V comes from a float prefix and calibration starts after them:

| Build | Score | Against the swap build (47.1) | Against float (50.9) |
|---|---:|---|---|
| Decoder only, PC features, no text prefix | 33.6 | -13.5 [-20.5, -6.7] | -17.3 |
| Decoder only, PC features, text prefix | 48.7 | +1.7 [-3.1, 6.5] | -2.2 [-7.2, 2.8] |
| All on the board, no text prefix | 31.3 | -15.8 | -19.7 |
| All on the board, text prefix | 50.4 | +3.3 [-2.2, 8.9] | -0.5 [-6.1, 5.1] |
| Final: text prefix and split block 5 | 48.5 | +1.4 [-3.6, 6.4] | -2.5 [-8.4, 3.2] |

  The split block is closer to float in the simulator yet scores 1.9 lower (9 answers
  better, 15 worse), [-6.7, 2.9]: the two cannot be told apart. It stays, because the
  detector needs its time and memory.

- **One program** (`resident/duosight.c`) loads everything once: the head, the decoder
  layers, then the encoder blocks, largest first, each sharing the I/O buffers at once,
  with the page cache dropped and memory compacted before each model. The 23 MB Python
  tokenizer moved out of the resident process.

Detector off, 150 images, medians: reading the canvas 12 ms, patch 28 ms, 12 blocks
2921 ms (before the split), connector 54 ms, prefill of 86 tokens 483 ms, decode
12.1 tok/s.

### Beside the detector, final build

One forced event each, two questions (`resident/watch_fast.py`):

| Detector | First answer | Both answers | Detector fps | Worst latency |
|---|---:|---:|---:|---:|
| 5 fps | 3.87 s | 4.06 s | 5.0 | 324 ms |
| 15 fps | 3.72 to 3.79 s | 4.06 to 4.25 s | 12.9 | 314 ms |
| 30 fps | 4.16 s | 4.68 s | 28.2 | 312 ms |

- At 5 fps the first answer splits into frame wait 0.25 s, canvas 0.15 s, patch 0.04 s,
  12 blocks 2.65 s, connector 0.07 s, prefill 0.62 s, decode 0.10 s.
- The worst detector latency fell from about 0.7 s to about 0.3 s, because the longest
  VLM call is now a 205 ms block instead of the 609 ms one.
- 5 minutes at 15 fps: 4637 frames, 14.9 fps, no frames lost, 28 dropped, worst latency
  299 ms. CMA: 335 MiB used, 9 MiB free.

### A real person

`--gain 0`, detector at 15 fps; the same person walked in three times:

| Score | Doing | Wearing | First answer | Both | From first sighting |
|---:|---|---|---:|---:|---:|
| 0.86 | Standing. | White. | 4.91 s | 5.20 s | 5.33 s |
| 0.77 | Talking. | Headphones. | 4.04 s | 4.37 s | 4.51 s |
| 0.77 | Looking in window. | Suitcase. | 4.23 s | 4.68 s | 4.82 s |

- The 5.33 s event ran while an extra monitoring process loaded the board; its prefill
  took 1401 ms against 780 to 800 ms in the other two.
- "Doing" was plausible all three times; "wearing" was right once.
- The board is short of RAM and swaps to zram under this load.

## 7. Answers on demand

The encoder needs 2.65 s per new frame. Even at the TPU's peak, the 12 blocks would take
about 0.4 s (an estimate: about 106 GMAC at 512 px against 250 GMAC/s). So no answer about
a frame taken after the question can come within a second. `live.py` answers about the
newest frame the board has encoded instead, and says how old it is.

- `bg` encodes a frame in steps and answers waiting questions between them, about the
  last frame. Then it prefills the image part of the prompt, so a question runs only its
  own tokens. `say` takes a question in words, through the C tokenizer `bpe.h`.
- Keeping every shared prompt token in the cache made the INT8 decoder answer the same
  question about the same frame with "SITTING." and then "Rain.". The cache now keeps 69
  tokens (the prefix and the 64 rows, 4 passes of 16), and the question starts a pass,
  so the passes match a full prefill.
- The C tokenizer gives the same ids as `camera/tok.py` on 175 questions (the 150 scored
  ones and 25 with digits, contractions, runs of spaces, underscores and accents), in
  28 ms for all of them.

Checks, detector off (`resident/bg_check.py`, `resident/bg_score.py`):

| Check | Result |
|---|---|
| Image rows of a frame at once and in steps, 7 questions in between | Same bytes |
| Questions during the steps, about the last frame | 7 of 7 the same answer |
| A frame from the cache against the same frame at once | Same answer |
| All 150 scored questions through `bg` | 150 of 150 the same ids, so 48.5 |
| Question to answer, during the steps | 0.36 to 0.58 s |
| Question to answer, no frame in steps | 0.25 s |

`live.py --bench 40`, detector at 15 fps, an empty room, runs of 95 to 107 s:

| New frames | Detector fps | Dropped | Worst latency | Answer, median | 90% | Max | Frame age, median | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Back to back | 9.6 | 626 | 362 ms | 0.59 s | 0.94 s | 1.25 s | 6.5 s | 9.4 s |
| Every 10 s, as with a person in view | 12.5 | 256 | 335 ms | 0.36 s | 0.66 s | 0.87 s | 8.9 s | 13.7 s |
| Every 60 s, nobody in view | 14.5 | 58 | 328 ms | 0.49 s | 1.09 s | 1.36 s | 30.1 s | 65.1 s |

- Back to back, a frame takes 4.5 s from capture to ready (canvas 118 ms, patch 34 ms,
  blocks 2758 ms, connector 68 ms, image prefill 448 ms, plus questions in between), and
  the detector holds 9.6 fps. So the detector decides: a new frame at once when the
  stable person count changes or a person moves by 15% of the width, otherwise after
  10 s with a person in view or 60 s without one.
- Answer time is mostly decoding beside the detector, 100 to 150 ms per token instead of
  85 ms. The question's prefill, two passes, takes 210 to 300 ms.
- Every (frame, question) pair got one answer only, across all repeats.
- 30 minutes with nobody asking: 31 frames, the detector at 14.5 fps, 0 lost, 842
  dropped, worst latency 333 ms.
- From a PC over Wi-Fi, after a person walked past and the detector started a new frame:
  four questions took 0.35 to 0.56 s on the board and 0.41 to 0.75 s round trip. "What is
  the person doing?" "Standing.", "What is the person wearing?" "White.", "What color is
  the wall?" "White.", "Is it day or night?" "Night." (wrong). The frame was 6.3 to 7.9 s
  old.

## Limits

- All scores come from COCO photos. There is no scored set of frames from the board's
  camera yet, and only three real-person events.
- 150 questions leave intervals of about 7 points either side.
- Answers on demand are about a frame at least 4.5 s old. A frame in steps cannot be
  stopped yet, so a person who just stepped in can wait longer.
- The camera has no automatic exposure. Daylight needs `--gain 0`, a dim room the default
  940.
- The questions assume a person, so a frame without one gets invented answers.
- About 9 MiB of CMA stay free with everything loaded.
