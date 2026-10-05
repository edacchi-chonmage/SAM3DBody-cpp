#!/usr/bin/env python3
"""Write CoreML-friendly copies of the backbone / decoder ONNX models.

Usage:
    python tools/prepare_coreml_models.py [--onnx-dir onnx]
        [--backbone-src backbone_fp16_trt.onnx] [--force] [--check]

Outputs (in --onnx-dir, each with a .data file for external weights):
    backbone_coreml.onnx  from --backbone-src
    decoder_coreml.onnx   from decoder_fp16.onnx

Steps and why:
  1. Neg(x) -> Mul(x, -1).  The CoreML EP (MLProgram) rejects 'Neg' (RoPE
     rotate_half), which splits the graph into dozens of partitions. The -1
     constant uses x's dtype (found with onnx shape inference; the backbone
     has both fp32 and fp16 Neg inputs).
  2. Offline ORT_ENABLE_BASIC optimization with free dimension 'B' = 1 and
     MatMulAddFusion disabled. That fusion creates Gemm nodes whose weights the
     CoreML EP transposes and writes inline into the MIL text (GBs, compile
     never finishes). Weights are saved as external data (>= 1024 bytes).
  3. At run time load the result with ORT_DISABLE_ALL so ORT does not fuse again.
     Use MLComputeUnits=CPUAndGPU (ALL tries the Neural Engine, which fails).

--check: compares original (CPU EP, default opts, B=1, seed 0) with the prepared
model (CPU EP, ORT_DISABLE_ALL) and prints the max abs diff of the first output.
The CoreML partition count is printed from a child process that reads ORT's
GetCapability log line.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

NP = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}
NP_IN = {1: np.float32, 10: np.float16, 7: np.int64, 6: np.int32, 9: np.bool_}


def neg_to_mul(src, dst):
    types = {}
    mi = onnx.shape_inference.infer_shapes(onnx.load(src, load_external_data=False))
    for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output):
        types[v.name] = v.type.tensor_type.elem_type
    m = onnx.load(src)
    g = m.graph
    consts = {}
    n = 0
    for node in g.node:
        if node.op_type == "Neg":
            et = types.get(node.input[0], types.get(node.output[0]))
            if et not in consts:
                name = f"__neg_one_{et}"
                g.initializer.append(numpy_helper.from_array(np.array(-1, dtype=NP[et]), name=name))
                consts[et] = name
            node.op_type = "Mul"
            node.input.append(consts[et])
            n += 1
    print(f"  Neg -> Mul: {n} nodes")
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(dst) + ".data")


def preopt(src, dst):
    so = ort.SessionOptions()
    so.add_free_dimension_override_by_name("B", 1)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = dst
    so.add_session_config_entry("session.optimized_model_external_initializers_file_name",
                                os.path.basename(dst) + ".data")
    so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"],
                         disabled_optimizers=["MatMulAddFusion"])


def rm(path):
    for p in (path, path + ".data"):
        if os.path.exists(p):
            os.remove(p)


def make_inputs(path):
    s = ort.SessionOptions()
    s.add_free_dimension_override_by_name("B", 1)
    s.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, s, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    feed = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        dt = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
              "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(bool)": np.bool_}[i.type]
        if np.issubdtype(dt, np.floating):
            feed[i.name] = rng.standard_normal(shape).astype(dt)
        else:
            feed[i.name] = np.zeros(shape, dtype=dt)
    return feed


def check(orig, prepared):
    feed = make_inputs(prepared)
    so = ort.SessionOptions()
    so.add_free_dimension_override_by_name("B", 1)
    a = ort.InferenceSession(orig, so, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    so2 = ort.SessionOptions()
    so2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    b = ort.InferenceSession(prepared, so2, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).max()
    print(f"  check: max abs diff (first output) = {d:.3e}  (output absmax {np.abs(a.astype(np.float32)).max():.3e})")


CAP_CODE = """
import sys, onnxruntime as ort
so = ort.SessionOptions(); so.log_severity_level = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ort.InferenceSession(sys.argv[1], so, providers=[("CoreMLExecutionProvider",
    {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUOnly"}), "CPUExecutionProvider"])
"""


def partitions(path):
    r = subprocess.run([sys.executable, "-c", CAP_CODE, path], capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if "number of partitions supported by CoreML" in l]
    print("  CoreML:", lines[-1].strip() if lines else f"(partition line not found, rc={r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--backbone-src", default="backbone_fp16_trt.onnx")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = a.onnx_dir
    jobs = [(a.backbone_src, "backbone_coreml.onnx"), ("decoder_fp16.onnx", "decoder_coreml.onnx")]
    for s, o in jobs:
        src, out = os.path.join(d, s), os.path.join(d, o)
        print(f"{s} -> {o}")
        if os.path.exists(out) and not a.force:
            print("  exists, skipping (use --force)")
        else:
            tmp_neg = os.path.join(d, "tmp_neg_" + o)
            tmp_out = os.path.join(d, "tmp_" + o)
            for t in (tmp_neg, tmp_out):
                rm(t)
            try:
                neg_to_mul(src, tmp_neg)
                preopt(tmp_neg, tmp_out)
            finally:
                rm(tmp_neg)
            # the .data name is baked into the model, so rename the model to the
            # final name only after renaming the data file and patching the reference
            m = onnx.load(tmp_out, load_external_data=False)
            for t in m.graph.initializer:
                for e in t.external_data:
                    if e.key == "location":
                        e.value = o + ".data"
            os.replace(tmp_out + ".data", out + ".data")
            onnx.save(m, out)
            rm(tmp_out) if os.path.exists(tmp_out + ".data") else os.remove(tmp_out)
        partitions(out)
        if a.check:
            check(src, out)


if __name__ == "__main__":
    main()
