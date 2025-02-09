from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time 
import os 

assert torch.cuda.is_available()

torch_device = "cuda"

model_name = "models/vicuna-7b-v1.3"

# tokenize
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How do you fine tune a large language model?"
input_text = (
    f"<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
)

model_inputs = tokenizer(input_text, return_tensors='pt').to(torch_device)
init_len = model_inputs['input_ids'].numel()

WINDOW_SIZE=20
LEVEL = 7

# generate draft tokens
if int(os.environ.get("USE_DRAFT", 0)):
    draft_model_path = "models/TinyLlama-1.1B-Chat-v1.0"
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_path, torch_dtype=torch.float16, device_map=torch_device)

    draft_tokens = draft_model.generate(**model_inputs, max_new_tokens= WINDOW_SIZE + LEVEL - 3, do_sample=False)
    model_inputs['input_ids'] = draft_tokens

# set proxy
if int(os.environ.get("LOAD_LADE", 0)):
    import lade 
    lade.augment_all()
    #For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7
    lade.config_lade(LEVEL=LEVEL, WINDOW_SIZE=WINDOW_SIZE, GUESS_SET_SIZE=20, DEBUG=1, POOL_FROM_PROMPT=True)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=torch_device)
model.tokenizer = tokenizer

torch.cuda.synchronize()
t0g = time.time()

greedy_output = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
torch.cuda.synchronize()
t1g = time.time()

print("Output:\n" + 100 * '-')
print("Greedy output: ", tokenizer.decode(greedy_output[0], skip_special_tokens=False))

print("Greedy Generated Tokens:", (greedy_output.numel() - init_len) ,"Generation Speed: ", (greedy_output.numel() - init_len) / (t1g - t0g), " tokens/s")
#python minimal.py #44 tokens/s
#LOAD_LADE=1 USE_LADE=1 python minimal.py #74 tokens/s, 1.6x throughput without changing output distribution!

