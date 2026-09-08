import os
import sys
import gc
import psutil
import torch
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

process = psutil.Process(os.getpid())

def get_ram_mb():
    return process.memory_info().rss / (1024 * 1024)

results = []

def log(msg):
    print(msg)
    results.append(msg)

log("============================================================")
log("  URDU SENTIMENT ENGINE - RAM BENCHMARK: PYTORCH VS ONNX  ")
log("============================================================")

ram_start = get_ram_mb()
log(f"1. Baseline Idle Process RAM: {ram_start:.2f} MB")

base_dir = os.path.dirname(os.path.abspath(__file__))
sentiment_dir = os.path.join(base_dir, "models", "sentiment_model")
emotion_dir = os.path.join(base_dir, "models", "emotion_model")
onnx_dir = os.path.join(base_dir, "models", "onnx")
os.makedirs(onnx_dir, exist_ok=True)

# -------------------------------------------------------------------------
# TEST 1: Current Approach (PyTorch FP32 + INT8 Dynamic Quantization)
# -------------------------------------------------------------------------
log("\n--- TEST 1: Current PyTorch + INT8 Dynamic Quantization ---")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ram_before_pt = get_ram_mb()
log(f"RAM before loading PyTorch models: {ram_before_pt:.2f} MB")

log("Loading Sentiment Model (FP32)...")
s_raw = AutoModelForSequenceClassification.from_pretrained(sentiment_dir)
ram_pt_fp32 = get_ram_mb()
log(f"RAM with 1 FP32 Model loaded: {ram_pt_fp32:.2f} MB")

log("Quantizing Sentiment Model (INT8)...")
s_model = torch.quantization.quantize_dynamic(s_raw, {torch.nn.Linear}, dtype=torch.qint8)
s_model.eval()
del s_raw
gc.collect()
ram_pt_quant = get_ram_mb()
log(f"RAM after PyTorch INT8 quantization: {ram_pt_quant:.2f} MB")

log("Predicting with PyTorch INT8...")
tokenizer = AutoTokenizer.from_pretrained(sentiment_dir)
dummy_text = "یہ پروڈکٹ بہت عمدہ ہے"
inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, truncation=True)
with torch.no_grad():
    _ = s_model(**inputs)

ram_pt_infer = get_ram_mb()
log(f"Peak RAM during PyTorch inference: {ram_pt_infer:.2f} MB")

del s_model, tokenizer
gc.collect()
ram_pt_freed = get_ram_mb()
log(f"RAM after PyTorch cleanup: {ram_pt_freed:.2f} MB")

# -------------------------------------------------------------------------
# TEST 2: ONNX Format + ONNX Runtime (CPU Execution Provider)
# -------------------------------------------------------------------------
log("\n--- TEST 2: ONNX Format + ONNX Runtime (CPU Execution Provider) ---")

s_onnx_path = os.path.join(onnx_dir, "sentiment.onnx")

if not os.path.exists(s_onnx_path):
    log("Exporting PyTorch Sentiment Model to ONNX format (Opset 18)...")
    tok = AutoTokenizer.from_pretrained(sentiment_dir)
    mod = AutoModelForSequenceClassification.from_pretrained(sentiment_dir)
    mod.eval()
    dummy_inp = tok(dummy_text, return_tensors="pt")
    
    torch.onnx.export(
        mod,
        (dummy_inp['input_ids'], dummy_inp['attention_mask']),
        s_onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        },
        opset_version=18
    )
    del mod
    gc.collect()
    log(f"ONNX Model Exported! File size: {os.path.getsize(s_onnx_path)/(1024*1024):.2f} MB")

ram_before_onnx = get_ram_mb()
log(f"RAM before loading ONNX session: {ram_before_onnx:.2f} MB")

log("Loading ONNX Runtime Session...")
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

onnx_session = ort.InferenceSession(s_onnx_path, sess_options, providers=['CPUExecutionProvider'])
ram_onnx_loaded = get_ram_mb()
log(f"RAM with ONNX Session loaded: {ram_onnx_loaded:.2f} MB")

log("Predicting with ONNX Runtime...")
tok_onnx = AutoTokenizer.from_pretrained(sentiment_dir)
inp_onnx = tok_onnx(dummy_text, return_tensors="np")

ort_inputs = {
    'input_ids': inp_onnx['input_ids'].astype(np.int64),
    'attention_mask': inp_onnx['attention_mask'].astype(np.int64)
}
logits = onnx_session.run(None, ort_inputs)[0]
ram_onnx_infer = get_ram_mb()
log(f"Peak RAM during ONNX inference: {ram_onnx_infer:.2f} MB")
log(f"ONNX Inference Output Logits: {logits.tolist()}")

del onnx_session
gc.collect()
ram_onnx_freed = get_ram_mb()
log(f"RAM after ONNX cleanup: {ram_onnx_freed:.2f} MB")

out_file = os.path.join(base_dir, "onnx_test_results.txt")
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\nAll benchmark results successfully written to '{out_file}'.")
