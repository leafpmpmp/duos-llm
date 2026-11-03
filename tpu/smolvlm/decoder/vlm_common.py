"""Parts shared by the decoder scripts. Needs numpy only, so it stays small in memory."""
import glob
import json
import struct

import numpy as np

SNAPSHOT = glob.glob("D:/duosight-eval/hf/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/*")[0]
ASK = "Answer using a single word or phrase."
IMAGE = "<fake_token_around_image><global-img>" + "<image>" * 64 + "<fake_token_around_image>"
IMAGE_ID = 49190
EOS = 49279
MAX_NEW = 24


def prompt(question):
    return f"<|im_start|>User:{IMAGE}{question} {ASK}<end_of_utterance>\nAssistant:"


def load_bf16(names):
    """Read BF16 tensors from the SmolVLM weights as float32."""
    f = open(SNAPSHOT + "/model.safetensors", "rb")
    size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(size))
    out = []
    for name in names:
        info = header[name]
        a, b = info["data_offsets"]
        f.seek(8 + size + a)
        raw = np.frombuffer(f.read(b - a), dtype=np.uint16).astype(np.uint32) << 16
        out.append(raw.view(np.float32).reshape(info["shape"]))
    return out


class Connector:
    """The final LayerNorm and the connector, in numpy: encoder output to image rows."""

    def __init__(self):
        self.w, self.b, self.proj = load_bf16(["model.vision_model.post_layernorm.weight",
                                               "model.vision_model.post_layernorm.bias",
                                               "model.connector.modality_projection.proj.weight"])

    def __call__(self, x):
        """x: [1024, 768] after the last block. Returns [64, 576]."""
        mean = x.mean(-1, keepdims=True)
        h = (x - mean) / np.sqrt(((x - mean) ** 2).mean(-1, keepdims=True) + 1e-6) * self.w + self.b
        h = h.reshape(32, 8, 3072).transpose(1, 0, 2).reshape(8, 8, 12288).transpose(1, 0, 2).reshape(64, 12288)
        return h @ self.proj.T
