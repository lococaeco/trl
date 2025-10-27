from datasets import load_dataset

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import trl
print(f"TRL import path: {trl.__file__}")

ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
print(ds.column_names)
print(ds[0])