from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from contextlib import contextmanager

@torch.no_grad()
def rollback(past_key_values, end_pos : int):
    past_key_values_trimmed = []
    for kv in past_key_values:
        k, v = kv
        # NOTE() the indexing is specific for bloom. This won't work for other models
        # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
        
        # k, v (batch, head, seq, hidden_dim)
        k = k[:, :, :end_pos, :]
        v = v[:, :, :end_pos, :]
        kv_trimmed = (k, v)
        past_key_values_trimmed.append(kv_trimmed)
    
    return past_key_values_trimmed

def jacobi_decoding(x : torch.Tensor, verified_len, model : torch.nn.Module):
    T = x.shape[1]
    past_key_values = None
    idx = 0

    while verified_len <= T:
        # outputs = model(x)
        if past_key_values:
            last_ids = x[:, verified_len-1:]
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True) 
            idx_next = torch.argmax(outputs.logits.view(-1, v), dim=-1)[None]
        else:
            outputs = model(x)
            v = outputs.logits.shape[2]
            idx_next = torch.argmax(outputs.logits[:, verified_len-1:].view(-1, v), dim=-1)[None]
        mask = x[:, verified_len:] == idx_next[:, :-1]
        accept_len = torch.cumprod(mask, dim=1).sum() + 1 # at least accept one token, only work if batch size = 1
        x = torch.cat((x[:, :verified_len], idx_next), dim=1) # combine verified tokens and unverified tokens
        verified_len += accept_len
        past_key_values = rollback(outputs.past_key_values, verified_len-1) # the last verified token does not have kv cache
        idx += 1
        #print(verified_len)
    return x, idx

def drafter(x : torch.Tensor, model : torch.nn.Module, N : int):
    n = len(x)
    T = len(x) + N

    past_key_values = None
    while n < T:
        # outputs = model(x)
        if past_key_values:
            last_ids = x[:, -1]
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True) 
        else:
            outputs = model(x)
        idx_next = torch.argmax(outputs.logits[:, -1, :], dim=-1)[None]
        past_key_values = outputs.past_key_values # shape (layer_nums, 2, [b, n, s, d]) 其中2表示k和v，n 是 attn_head_nums，nd=h
        x = torch.cat((x, idx_next), dim=1)
        n += 1
    return x

@contextmanager
def timer(wall_times, key):
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    end = time.time()
    wall_times[key].append(end-start)

def main():
    wall_times = {'draft model': [], 'target model': [], 'jacobi decoding': []}

    draft_model_path = 'models/llama-68m'
    target_model_path = 'models/vicuna-7b-v1.3'

    tokenizer = AutoTokenizer.from_pretrained(draft_model_path, padding_side = 'left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_path, device_map='cuda', torch_dtype=torch.float16)
    target_model = AutoModelForCausalLM.from_pretrained(target_model_path, device_map='cuda', torch_dtype=torch.float16)

    tokens = tokenizer(['Hello, I am'], return_tensors='pt').to(draft_model.device)
    with timer(wall_times, 'draft model'):
        draft_tokens = drafter(tokens.input_ids, draft_model, 50).to(target_model.device)
    draft_outputs = tokenizer.batch_decode(draft_tokens, skip_special_tokens=True)
    print(f'draft model:', draft_outputs, 'speed:', (draft_tokens.shape[1]-tokens.input_ids.shape[1])/sum(wall_times['draft model']))

    with timer(wall_times, 'target model'):
        draft_tokens = drafter(tokens.input_ids.to(target_model.device), target_model, 50).to(target_model.device)
    draft_outputs = tokenizer.batch_decode(draft_tokens, skip_special_tokens=True)
    print(f'target model:', draft_outputs, 'speed:', (draft_tokens.shape[1]-tokens.input_ids.shape[1])/sum(wall_times['target model']))
    target_speed = (draft_tokens.shape[1]-tokens.input_ids.shape[1])/sum(wall_times['target model'])

    with timer(wall_times, 'jacobi decoding'):
        draft_tokens = drafter(tokens.input_ids.to(draft_model.device), draft_model, 50).to(target_model.device)
        generate_ids, idx = jacobi_decoding(draft_tokens, tokens.input_ids.shape[1], target_model)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
    jacobi_speed = (generate_ids.shape[1]-tokens.input_ids.shape[1])/(sum(wall_times['jacobi decoding']))

    print(f'jacobi decoding:', outputs, 'speed:', (generate_ids.shape[1]-tokens.input_ids.shape[1])/(sum(wall_times['jacobi decoding'])))
    print(f'compression ratio {(generate_ids.shape[1]-tokens.input_ids.shape[1])/idx}, accelerate: {jacobi_speed/target_speed}')

if __name__ == '__main__':
    main()
