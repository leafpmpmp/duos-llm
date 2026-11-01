# SPDX-License-Identifier: MIT
# Make GEMV test graphs (batch 1) with SmolLM2-135M shapes, plus calibration
# inputs. Run in the tpu-mlir container: python3 make_onnx.py
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

rng = np.random.default_rng(0)


def weight(k, n):
    # Typical LLM weight scale: N(0, 0.02)
    return (rng.standard_normal((k, n)) * 0.02).astype(np.float32)


def save(name, nodes, inits, k, n_out):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, k])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, n_out])
    g = helper.make_graph(nodes, name, [x], [y], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    onnx.save(m, f"{name}.onnx")
    os.makedirs(f"calib_{k}", exist_ok=True)
    for i in range(8):
        np.savez(f"calib_{k}/in{i}.npz", x=rng.standard_normal((1, k)).astype(np.float32))
    print(name, sum(int(np.prod(t.dims)) for t in inits) / 1e6, "M params")


def single(name, k, n):
    w = numpy_helper.from_array(weight(k, n), "w")
    save(name, [helper.make_node("MatMul", ["x", "w"], ["y"])], [w], k, n)


def chain(name, layers, d=576, f=1536):
    nodes, inits, cur = [], [], "x"
    for i in range(layers):
        for j, (k, n) in enumerate(((d, f), (f, d))):
            w, mm, out = f"w{i}_{j}", f"mm{i}_{j}", f"r{i}_{j}"
            inits.append(numpy_helper.from_array(weight(k, n) * 3, w))
            last = i == layers - 1 and j == 1
            nodes.append(helper.make_node("MatMul", [cur, w], ["y" if last else mm]))
            if not last:
                nodes.append(helper.make_node("Relu", [mm], [out]))
                cur = out
    save(name, nodes, inits, d, d)


single("g576", 576, 576)
single("lmhead", 576, 49152)
chain("chain30", 30)
