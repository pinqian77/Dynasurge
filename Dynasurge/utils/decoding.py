import time

import torch
from torch.nn.functional import softmax
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Tree.SpecTree import SpecTree
from model.Engine import InferenceEngine
from utils.utils import _make_causal_mask, get_sampling_logits

@torch.inference_mode()
def Autoregressive(target_model : InferenceEngine, dataloader: DataLoader, tokenizer, T, top_p, max_length, args, verbose=False):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    total_time = 0.0

    for data_id, batch in tqdm(enumerate(dataloader), total=num_eval_steps):        
        input_ids = batch['input_ids'][..., :128]
        terminate = True if input_ids[0][-1] == 2 else False            
        position_ids = torch.arange(max_length).to('cuda:0').unsqueeze(0)
        storage_ids = torch.arange(max_length).to('cuda:0')
        attn_mask = _make_causal_mask((max_length, max_length), target_model.dtype, target_model.device)
        torch.cuda.synchronize()
        t1 = time.time()
        inner_decoding_step = 0
        start_length = input_ids.shape[1]

        while inner_decoding_step + 128 < (max_length - start_length) and not terminate: # TODO: should fix Sequioa bug. current max length is max_length - 128
        # while inner_decoding_step < (max_length - start_length) and not terminate:
            if inner_decoding_step == 0:
                logits = target_model.inference(input_ids = input_ids, storage_ids=storage_ids[:start_length],
                                                position_ids = position_ids[..., :start_length], 
                                                attn_mask=attn_mask[:start_length, :start_length][None, None, :, :])[0][-1]
            else:
                logits = target_model.inference(input_ids = input_ids, storage_ids=storage_ids[start_length + inner_decoding_step-1 : start_length + inner_decoding_step],
                                                position_ids = position_ids[..., start_length + inner_decoding_step-1 : start_length + inner_decoding_step], 
                                                attn_mask=attn_mask[start_length + inner_decoding_step-1 : start_length + inner_decoding_step, :start_length + inner_decoding_step][None, None, :, :])[0][-1]
            
            # Adjust logits and sample a new token
            logits = get_sampling_logits(logits=logits, top_p=top_p, T=T)
            p = softmax(logits / T, dim=-1)
            new_token = p.multinomial(num_samples=1).unsqueeze(0)
            
            input_ids = new_token
            num_decoding_steps += 1
            inner_decoding_step += 1

            if input_ids[0][-1] == 2: 
                terminate = True

        torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)
        target_model.clear_kv()
        
        if verbose and num_decoding_steps > 0:
            print(f"data id: {data_id} | time cost: {t2 - t1:.5f}s | final length: {start_length + inner_decoding_step} | decoding step: {inner_decoding_step} | latency per token: {(t2 - t1) / inner_decoding_step:.5f}s")
            
    print(f"[Summary for {args}]")
    print(f"==========> total time: {total_time:.5f}s | total decoding step: {num_decoding_steps} | latency per token: {total_time / num_decoding_steps:.5f}s | latency per request: {total_time / num_eval_steps:.5f}")
    return num_decoding_steps


@torch.inference_mode()
def Sequoia(target_model : InferenceEngine, draft_model: InferenceEngine, dataloader: DataLoader, tokenizer, T, top_p,
            max_length, bfs_verify, residual_graph, grow_map, sampling_callables,
            sample_gather_indices, verbose, args):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to('cuda:0')
    parents_buffer =  torch.zeros(max_length).long().to('cuda:0')
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    
    for data_id, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
        input_ids = batch['input_ids'][..., :128]
        terminate = True if input_ids[0][-1] == 2 else False
        draft_kv_len = 0
        target_kv_len = 0
        attn_mask.fill_(torch.finfo(dtype).min)
        spectree = SpecTree(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                                top_p=top_p,
                                draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length, grow_map=grow_map,
                                attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                parents_buffer = parents_buffer, 
                                position_ids = position_ids,
                                residual_graph = residual_graph,
                                sampling_callables=sampling_callables,
                                sample_gather_indices = sample_gather_indices)
        torch.cuda.synchronize()
        t1 = time.time()

        curr_decoding_steps = 0
        curr_large_model_steps = 0
        start_length = input_ids.shape[1]
        # while input_ids.shape[1] < 256 and not terminate:
        while input_ids.shape[1] < (max_length - start_length) and not terminate:
            # print(f"input_ids shape: {input_ids.shape[1]}, max_length: {max_length}, start_length: {start_length}, max_length - start_length: {max_length - start_length}")
            spectree.construct_grow_map()
            
            if not bfs_verify:
                valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify()
            else:
                valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify_bfs()
            
            curr_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
            curr_large_model_steps += 1
            input_ids = valid_tokens.unsqueeze(0)
            if (input_ids[0][-1] == 2) or (input_ids[0][-1] == 0): 
                terminate = True
        
        num_decoding_steps += curr_decoding_steps
        num_large_model_steps += curr_large_model_steps

        torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)
        draft_model.clear_kv()
        target_model.clear_kv()

        if verbose and num_decoding_steps > 0:
            print(f"data id: {data_id} | time cost: {t2 - t1:.5f}s | final length: {start_length + curr_decoding_steps} | total decoding step: {curr_decoding_steps} | large model step: {curr_large_model_steps} | latency per token: {(t2 - t1) / curr_decoding_steps:.5f}s")
        
    print(f"[Summary for {args}]")
    print(f"==========> total time: {total_time:.5f}s | total decoding step: {num_decoding_steps} | large model step: {num_large_model_steps} | latency per token: {total_time / num_decoding_steps:.5f}s | latency per request: {total_time / num_eval_steps:.5f} | accept rate: {num_decoding_steps / num_large_model_steps:.5f}")
    return num_decoding_steps / num_large_model_steps


@torch.inference_mode()
def Dynasurge(target_model: InferenceEngine, draft_model: InferenceEngine, dataloader: DataLoader, T, top_p, max_length,
              residual_graph, sampling_callables, sample_gather_indices, tokenizer, args, bfs_verify=False, verbose=False):
    from Tree.CreateDynamicTreeMap import create_tree_map
    from Tree.DynamicSpecTree import DynamicSpecTree

    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0


    for data_id, batch in tqdm(enumerate(dataloader), total=num_eval_steps): 
        # input_ids = batch['input_ids'][..., :max_length]
        input_ids = batch['input_ids'][..., :128]

        grow_map = torch.load(args.growmap)
        terminate = True if input_ids[0][-1] == 2 else False
        draft_kv_len = 0
        target_kv_len = 0
        dtype = torch.float16
        attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
        sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
        new_tokens_buffer = None
        parents_buffer = None
        position_ids = torch.zeros(max_length).long().to('cuda:0')
        attn_mask.fill_(torch.finfo(dtype).min)

        spectree = DynamicSpecTree(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                            top_p=top_p,
                            draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                            draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length,
                            grow_map=grow_map,
                            attn_mask=attn_mask, sequence=sequence, new_tokens_buffer=new_tokens_buffer,
                            parents_buffer=parents_buffer,
                            position_ids=position_ids,
                            residual_graph=residual_graph,
                            sampling_callables=sampling_callables,
                            sample_gather_indices=sample_gather_indices, vocab_size=args.vocab)

        #run
        torch.cuda.synchronize()
        t1 = time.time()

        curr_decoding_steps = 0
        curr_large_model_steps = 0
        start_length = input_ids.shape[1]
        # while input_ids.shape[1] < max_length and terminate == False:
        while input_ids.shape[1] < (max_length - start_length) and not terminate:

            grow_map, construction_information = create_tree_map(draft_model, input_ids, args.tree_max_subnodes,
                                                                    args.B,
                                                                    args.M, args.vocab)
            # override
            spectree.override_information(grow_map, construction_information)

            if not bfs_verify:
                valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify()
            else:
                valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify_bfs()

            curr_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
            curr_large_model_steps += 1
            input_ids = valid_tokens.unsqueeze(0)
            if (input_ids[0][-1] == 2) or (input_ids[0][-1] == 0): 
                terminate = True
        
        num_decoding_steps += curr_decoding_steps
        num_large_model_steps += curr_large_model_steps

        torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)
        draft_model.clear_kv()
        target_model.clear_kv()

        if verbose and num_decoding_steps > 0:
            print(f"data id: {data_id} | time cost: {t2 - t1:.5f}s | final length: {start_length + curr_decoding_steps} | total decoding step: {curr_decoding_steps} | large model step: {curr_large_model_steps} | latency per token: {(t2 - t1) / curr_decoding_steps:.5f}s")
    
    print(f"[Summary for {args}]")
    print(f"==========> total time: {total_time:.5f}s | total decoding step: {num_decoding_steps} | large model step: {num_large_model_steps} | latency per token: {total_time / num_decoding_steps:.5f}s | latency per request: {total_time / num_eval_steps:.5f} | accept rate: {num_decoding_steps / num_large_model_steps:.5f}")
    return num_decoding_steps / num_large_model_steps