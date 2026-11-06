# Quality test: predictions and decision rule

Fixed on 2026-11-01, before any model ran. Do not edit after the first run.

## Question

Is SmolVLM-256M worth porting to the Duo S? On person questions, does it answer
better than a language model that reads only the detector output, at the 256 px
input the board can afford?

## Setup

- Questions: VQA v2 val, questions that name a person. One question per image.
- Sample: seed 42, 150 "other", 100 "yes/no", 50 "number" answers.
- Images: COCO val2014 with at least one YOLO11n person at conf 0.4.
- Models: float SmolVLM-256M-Instruct and SmolLM2-135M-Instruct on the PC.
- Metric: VQA accuracy, min(matches / 3, 1), with leave-one-out over 10 answers.

## Conditions

| Name | Input |
|---|---|
| json_llm | SmolLM2-135M reads the detections as JSON, no image |
| full256 | VLM, whole image, 256 px, 16 image tokens |
| crop256 | VLM, largest person with 10% margin, 256 px, 16 image tokens |
| full512 | VLM, whole image, 512 px, 64 image tokens |
| crop512 | VLM, largest person with 10% margin, 512 px, 64 image tokens |

## Predictions

1. crop256 beats json_llm on "other" answers by at least 10 points.
2. crop256 beats full256 on "other" answers.
3. crop256 lands within 10 points of full512 on "other" answers.

## Decision rule

Keep the VLM path only if both hold on "other" answers:

- crop256 minus json_llm is at least 10 points, and the paired 95% bootstrap
  interval stays above 0.
- full512 minus crop256 is at most 10 points.

If either fails, the VLM path needs a written reason to continue.

## Amendment after the 14-question smoke run

Made before the full run, and before any full-run score existed.

The smoke run showed that SmolLM2 answers in sentences, such as "The boy is carrying
a long board.", so exact matching scores it 0 even when it is right. That would
inflate the gain of the VLM. Two changes follow, for every condition:

- Answers may run to 24 tokens instead of 12, so that sentences finish.
- The decision rule uses a lenient score: a human answer counts when it appears
  as whole words anywhere in the output. The exact score is reported beside it.

The thresholds and the sample stay as written above.

## Limits

- Float models give an upper bound: the INT8 and BF16 board models can only lose.
- COCO photos are about 640 px, so a crop has far less detail than a RAW crop from
  the 4.6 MP sensor. crop256 is conservative for the raw-crop idea.
- This test says nothing about speed or memory on the board.
