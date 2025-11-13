# src/export_onnx_coreml.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path
import torch

MODEL_DIR = "models/roberta_ner"
EXPORT_DIR = "models/onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

# we export a simplified version that accepts input_ids, attention_mask
dummy_input = tokenizer("This is a dummy input for export", return_tensors="pt", padding="max_length", max_length=128)
input_names = ["input_ids", "attention_mask"]
output_names = ["logits"]

torch.onnx.export(model, (dummy_input['input_ids'], dummy_input['attention_mask']),
                  f"{EXPORT_DIR}/roberta_ner.onnx",
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'attention_mask': {0:'batch',1:'seq'}, 'logits': {0:'batch',1:'seq'}},
                  opset_version=13)
print("ONNX saved to", EXPORT_DIR)
