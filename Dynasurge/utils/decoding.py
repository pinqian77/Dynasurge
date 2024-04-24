import time

import torch
from torch.nn.functional import softmax
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Tree.SpecTree import SpecTree
from model.Engine import InferenceEngine
from utils.utils import _make_causal_mask, get_sampling_logits


@torch.inference_mode()
def Autoregressive(target_model: InferenceEngine, dataloader: DataLoader, T=0.6, top_p=0.9, max_length=512):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    total_time = 0.0

    for _, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
        input_ids = batch['input_ids'][..., :128]
        terminate = True if input_ids[0][-1] == 2 else False
        position_ids = torch.arange(max_length).to('cuda:0').unsqueeze(0)
        storage_ids = torch.arange(max_length).to('cuda:0')
        attn_mask = _make_causal_mask((max_length, max_length), target_model.dtype, target_model.device)
        torch.cuda.synchronize()
        t1 = time.time()
        inner_decoding_step = 0
        start_length = 0
        # Decode tokens until the maximum step is reached or termination is triggered
        while inner_decoding_step < 32 and not terminate:
            # prefill for first token
            if inner_decoding_step == 0:
                start_length = input_ids.shape[1]
                logits = target_model.inference(input_ids=input_ids, storage_ids=storage_ids[:start_length],
                                                position_ids=position_ids[..., :start_length],
                                                attn_mask=attn_mask[:start_length, :start_length][None, None, :, :])[0][
                    -1]
            else:
                logits = target_model.inference(input_ids=input_ids, storage_ids=storage_ids[
                                                                                 start_length + inner_decoding_step - 1: start_length + inner_decoding_step],
                                                position_ids=position_ids[...,
                                                             start_length + inner_decoding_step - 1: start_length + inner_decoding_step],
                                                attn_mask=attn_mask[
                                                          start_length + inner_decoding_step - 1: start_length + inner_decoding_step,
                                                          :start_length + inner_decoding_step][None, None, :, :])[0][-1]

            # Adjust logits based on temperature and top-p filtering, then sample a new token
            logits = get_sampling_logits(logits=logits, top_p=top_p, T=T)
            p = softmax(logits / T, dim=-1)
            new_token = p.multinomial(num_samples=1).unsqueeze(0)

            input_ids = new_token
            num_decoding_steps += 1
            inner_decoding_step += 1

            # Check if the new token is the end-of-sequence token
            if input_ids[0][-1] == 2:
                terminate = True

        torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)
        target_model.clear_kv()

    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, latency per request: {:5f}".format(total_time,
                                                                                                        total_time / num_decoding_steps,
                                                                                                        num_decoding_steps,
                                                                                                        total_time / num_eval_steps))
    return num_decoding_steps


@torch.inference_mode()
def Sequoia(target_model: InferenceEngine, draft_model: InferenceEngine, dataloader: DataLoader, T=0.6, top_p=0.9,
            max_length=512, residual_graph=None, grow_map=None, sampling_callables=None, sample_gather_indices=None):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer = torch.zeros(max_length).long().to('cuda:0')
    parents_buffer = torch.zeros(max_length).long().to('cuda:0')
    position_ids = torch.zeros(max_length).long().to('cuda:0')

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            terminate = True if input_ids[0][-1] == 2 else False
            draft_kv_len = 0
            target_kv_len = 0
            attn_mask.fill_(torch.finfo(dtype).min)
            spectree = SpecTree(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                                top_p=top_p,
                                draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length,
                                grow_map=grow_map,
                                attn_mask=attn_mask, sequence=sequence, new_tokens_buffer=new_tokens_buffer,
                                parents_buffer=parents_buffer,
                                position_ids=position_ids,
                                residual_graph=residual_graph,
                                sampling_callables=sampling_callables,
                                sample_gather_indices=sample_gather_indices)
            torch.cuda.synchronize()
            t1 = time.time()
            while input_ids.shape[1] < 256 and not terminate:
                spectree.construct_grow_map()
                valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify()

                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)
                if (input_ids[0][-1] == 2) or (input_ids[0][-1] == 0):
                    terminate = True

            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            draft_model.clear_kv()
            target_model.clear_kv()
    print(
        "total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}, accept rate: {:.5f}, latency per request: {:5f}".format(
            total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps,
                        num_decoding_steps / num_large_model_steps, total_time / num_eval_steps))
    return num_decoding_steps / num_large_model_steps


@torch.inference_mode()
def Dynasurge(target_model: InferenceEngine, draft_model: InferenceEngine, dataloader: DataLoader, T, top_p, max_length,
              residual_graph, sampling_callables, sample_gather_indices, tokenizer, args):
    from Tree.CreateDynamicTreeMap import create_tree_map
    from Tree.DynamicSpecTree import DynamicSpecTree

    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    data_id = 0

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            data_id = data_id + 1
            input_ids = batch['input_ids'][..., :max_length]

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
            pos = 0
            generated_ids = []
            while input_ids.shape[1] < max_length and terminate == False:

                grow_map, construction_information = create_tree_map(draft_model, input_ids, args.tree_max_subnodes,
                                                                     args.B,
                                                                     args.M, args.vocab)
                # override
                spectree.override_information(grow_map, construction_information)

                valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify()

                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                input_begin_pos = input_ids.shape[1]
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)

                if (input_ids[0][-1] == 2) or (input_ids[0][-1] == 0): terminate = True

                # generated_ids.extend(input_ids[0][input_begin_pos:].tolist())
                #
                # generated_text = (
                #     tokenizer.decode(
                #         generated_ids,
                #         skip_special_tokens=True,
                #         clean_up_tokenization_spaces=True,
                #         spaces_between_special_tokens=False,
                #     )
                #     .strip()
                #     .split(" ")
                # )
                # now = len(generated_text) - 1
                # if now > pos:
                #     print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                #     pos = now
                #
                # print(" ".join(generated_text[pos:]), flush=True)

            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            draft_model.clear_kv()
            target_model.clear_kv()
            if num_large_model_steps > 0:
                print(
                    "Data ID: {} total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}, {}".format(
                        data_id, total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps,
                                             num_decoding_steps / num_large_model_steps), flush=True)
