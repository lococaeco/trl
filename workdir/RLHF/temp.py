from transformers import AutoTokenizer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trl

print(trl.__file__)
# tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

# print("EOS token:", tok.eos_token)
# print("EOS token_id:", tok.eos_token_id)
# print("Special tokens:", tok.special_tokens_map)
