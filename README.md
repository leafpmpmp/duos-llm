# duos-llm

A language model on the [Milk-V Duo S](https://milkv.io/duo-s) with mainline
Linux. [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
runs on the SG2000 TPU at 12.7 tokens per second. The TPU runtime is a small C
program on the ARM core. The project also has a llama.cpp build for each CPU,
with new XTheadVector kernels for the RISC-V core.

- 30 decoder layers and the LM head, one TPU model each.
- 512 or 1024 tokens of context, with a sliding window for long replies.
- Fast prompts: the TPU processes 4 tokens per call.
- Speculative decoding from prompt lookup.
- A system prompt snapshot that loads in about 80 ms.

## Performance

TPU on the ARM core, 512 MB board. PPL is perplexity on WikiText-2 tokens. Lower
is better.

| Model | Prefill | Decode | PPL, 512 tokens | PPL, 1024 tokens |
|---|---|---|---|---|
| `q4` S=512 | 41.1 tok/s | 12.7 tok/s | 19.01 | Not available |
| `q4` S=1024 | 37.1 tok/s | 11.2 tok/s | 19.53 | 16.69 |
| Float reference | | | 18.41 | 15.77 |

S is the number of cache slots. Decode values are without speculation.

The next two results are from an S=256 model:

- Speculation raised decode speed from 10.8 to 21.4 tok/s on a rewrite prompt.
- A system prompt snapshot cut the time to first token from 2.2 s to 0.7 s.

CPU with llama.cpp, one thread. Each cell gives the prompt speed and the
generation speed in tok/s, for 64 prompt tokens and 32 generated tokens.

| Model | ARM A53 | RISC-V C906, scalar | RISC-V C906, XTheadVector |
|---|---|---|---|
| SmolLM2-135M Q8_0 | 7.75 / 5.65 | 0.95 / 0.68 | 4.79 / 2.77 |
| SmolLM2-135M Q4_0 | 8.26 / 6.01 | 1.35 / 0.85 | 4.31 / 2.44 |
| SmolLM2-360M Q4_0 | 2.89 / 2.36 | 0.46 / 0.34 | 1.57 / 1.16 |

The full results are in `bench-arm64.txt`, `bench-135m-riscv.txt` and
`bench-360m-riscv.txt`.

## Vision: SmolVLM

`tpu/smolvlm` runs [SmolVLM-256M](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
on the same TPU, beside a YOLO11n person detector from
[duos-detect](https://github.com/leafpmpmp/duos-detect). One resident program holds the
vision encoder and the text model, and a page on the board takes questions from any
browser on the local network.

- On 150 held-out VQA v2 questions about people, the board scores 48.5. SmolVLM-256M in
  float scores 50.9.
- A question gets its answer in a median of 0.36 to 0.59 s, about the newest frame the
  board has encoded. That frame is at least 4.5 s old.
- The detector keeps 12.5 to 14.5 of its 15 fps.

See [tpu/smolvlm/README.md](tpu/smolvlm/README.md) for the results, the build and the
limits.

## Requirements

Hardware: Milk-V Duo S (SG2000). The TPU path needs the ARM core. The CPU path
runs on either core. The boot switch selects the core.

Software on the board:

- [Armbian](https://github.com/armbian/build) for the Milk-V Duo S (board
  `milkv-duos-arm`, branch `edge`). The image includes the TPU driver
  (`/dev/cvi-tpu0`). Tested with kernel 7.2.10.
- A CMA pool of 228 MiB for the TPU models. In `/boot/armbianEnv.txt`, set
  `extraargs=cma=228M`, then reboot. The stock device tree allows no more
  than this value, because ramoops splits the RAM.

For a larger pool, move ramoops and the boot files:

1. Compile the overlay and copy it to the board:

   ```bash
   dtc -I dts -O dtb -o boot/ramoops-high.dtbo boot/ramoops-high.dts
   scp boot/ramoops-high.dtbo root@BOARD:/boot/overlay-user/
   ```

2. Set these lines in `/boot/armbianEnv.txt`, then reboot:

   ```
   extraargs=cma=320M
   user_overlays=ramoops-high
   initrd_high=0xffffffffffffffff
   fdt_high=0xffffffffffffffff
   ```

The maximum is then about 396 MiB. Tested with 320 MiB.

## Build

You need a Linux PC (x86-64) with Docker, git, curl, unzip and
[uv](https://docs.astral.sh/uv/). The scripts cross-compile in a Debian
container (`docker/Dockerfile`). The TPU compiler image needs about 11 GB.

1. Get llama.cpp, the models, the text data and the Python packages:

   ```bash
   tools/fetch-llama.sh
   tools/fetch-models.sh
   tools/fetch-data.sh
   uv sync --frozen
   ```

2. Build the TPU runtime. It comes from the vendor SDK, with a port to the
   mainline TPU driver. A sparse clone gets only the parts the build needs:

   ```bash
   git clone --depth 1 --filter=blob:none --sparse -b mainline-tpu \
       https://github.com/leafpmpmp/duo-buildroot-sdk-v2 ../duo-sdk
   git -C ../duo-sdk sparse-checkout set cviruntime cvikernel cvibuilder flatbuffers cnpy
   tools/build-cviruntime.sh ../duo-sdk
   ```

3. Build llama.cpp for the ARM core, then the TPU program. The program uses
   the llama.cpp tokenizer:

   ```bash
   ./build-arm64.sh
   tpu/runtime/build.sh
   ```

4. Compile the TPU models. Use `512` or `1024`:

   ```bash
   tpu/q4.sh build 512
   ```

5. Copy the files to the board. The model set needs about 270 MB:

   ```bash
   ssh root@BOARD mkdir -p /root/tpu /root/llm/models
   scp -r build/cviruntime/arm64/install/lib root@BOARD:/root/tpu/
   scp models/SmolLM2-135M-Instruct-Q8_0.gguf root@BOARD:/root/llm/models/
   tpu/q4.sh package 512 | ssh root@BOARD 'mkdir -p /root/llm/tpu && tar xf - -C /root/llm/tpu'
   ```

The model directory on the board is `smollm135s512` for S=512 and
`smollm135s1k` for S=1024.

For the benchmarks, run `tools/build-benches.sh arm64` or
`tools/build-benches.sh riscv64`. The output goes to `build/<arch>-tools`.

## Usage

Run these commands on the board. Stop other TPU programs first.

```bash
cd /root/llm/tpu/smollm135s512
export LD_LIBRARY_PATH=/root/tpu/lib
V=/root/llm/models/SmolLM2-135M-Instruct-Q8_0.gguf
```

Make the system prompt snapshot once. Repeat this step for each new system
prompt:

```bash
./llm-arm64 -d q4 -x prefix-imstart.kv -v $V -S "$(cat system.txt)" --snapshot sys-dev.kv
```

Ask a question:

```bash
./llm-arm64 -d q4 -x sys-dev.kv -v $V -c -S "$(cat system.txt)" -n 400 -p "What is a TPU?"
```

| Option | Function |
|---|---|
| `-d DIR` | Model directory |
| `-v FILE` | GGUF file for the tokenizer |
| `-x FILE` | K/V snapshot to load |
| `-c` | Apply the chat template |
| `-S TEXT` | System message |
| `-p TEXT` | Prompt |
| `-n N` | Maximum number of generated tokens |
| `-t TEMP` | Sampling temperature. Use `-t 0.7` for long text. This turns speculation off. |
| `--draft N` | Draft length. `0` turns speculation off. |
| `--snapshot FILE` | Save the K/V of the system message |
| `--ppl FILE` | Measure perplexity on a token file |

To measure perplexity:

```bash
./llm-arm64 -d q4 -x prefix-imstart.kv --ppl wiki-eval-512-p1.i32
```

## CPU path

Build llama.cpp for one core:

```bash
./build-arm64.sh
ZFH=1 ./build-riscv.sh xtv
```

The output goes to `build/arm64/bin` and `build/riscv64-xtv-zfh/bin`. For a
RISC-V build without vector code, run `./build-riscv.sh scalar`.

Copy `llama-cli` and a GGUF file to the board, then run it with one thread:

```bash
./llama-cli -m SmolLM2-135M-Instruct-Q4_0.gguf -t 1 -st -n 80 --temp 0 -p "What is a CPU?" < /dev/null
```

Use the Q4_0 or Q8_0 files on RISC-V. The Q4_K_M files use mostly scalar code,
because the SmolLM2 tensor sizes do not suit the K-quant kernels.

## How it works

- Each decoder layer is one TPU model, so the runtime can mix precisions per
  layer. The `q4` config uses BF16 for most layers. The other layers use
  MIX2: INT8 for weight products and SiLU, BF16 for the other operations.
- The residual stream stays in fp32 on the CPU. SmolLM2 has activations up to
  about 26000, and INT8 cannot hold them.
- The K/V cache stays in the input buffers of the models. The runtime writes
  only the rows that change.
- Each call takes 4 tokens. Prefill uses all 4. Decode uses 1 token and up to
  3 draft tokens from prompt lookup. The output verifies the drafts.
- When the cache is full, the oldest generated tokens leave it. The prompt
  stays.

## Repository

| Path | Contents |
|---|---|
| `tpu/runtime/` | The TPU program `llm`, and `mkcfg.sh` for model configs |
| `tpu/smollm/` | Model definition, ONNX export, TPU compilation and checks |
| `tpu/smolvlm/` | SmolVLM-256M beside the detector: encoder, decoder, the resident program and the question page |
| `tpu/q4.sh` | Builds and packs the `q4` config |
| `tpu/tpubench.c`, `tpu/cmacopy.c` | TPU and CMA copy benchmarks |
| `tpu/gemv/` | Early test: matrix product speed on the TPU |
| `patches/` | The llama.cpp patch: XTheadVector q4_0 and q8_0 kernels |
| `boot/` | Device tree overlay that moves ramoops |
| `tools/` | Download and build scripts, CPU benchmarks |
| `docker/` | The cross-build image and the TPU compiler image |

`tpu/smollm/export.py` and `tpu/smollm/export_all.py` are the older exports
with 1 token per call. The `q4` config uses `export_t.py`.

## Limits

- `llm` supports only SmolLM2-135M. `tpu/smolvlm` runs SmolVLM-256M.
- `llm` has no interactive mode. Each run answers one prompt.
- Long greedy replies can repeat. Use `-t 0.7`.
- S=1024 models score worse than S=512 models on 512 tokens (19.53 against
  19.01). The cause is not known.
- The TPU runtime needs the CMA pool. Other TPU programs can cause a load
  failure.

## Licence

duos-llm is free software under the MIT License. See [LICENSE](LICENSE).

Third-party parts are not in this repository, except the llama.cpp patch:

- The patch changes [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- The SmolLM2 and SmolVLM weights are under the Apache License 2.0. The compiled
  TPU models come from these weights. Include the licence if you share them.
- `tools/fetch-data.sh` downloads WikiText-2, which is under CC BY-SA.
- The TPU runtime sources are copyright Cvitek and Bitmain and carry no licence
  grant. Build the runtime for your own boards. Do not redistribute it.
