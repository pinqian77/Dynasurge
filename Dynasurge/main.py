# import debugpy
# try:
#     # Default host: localhost 127.0.0.1, port: 9501
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for Debugger Attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import argparse
import random

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from model.Engine import InferenceEngine
from utils.dataset import get_tokenized_dataset
from utils.decoding import Autoregressive, Sequoia, Dynasurge
from utils.utils import cuda_graph_for_residual, cuda_graph_for_sampling_without_replacement


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17, help='random seed')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')

    parser.add_argument('--draft', type=str, help='draft model', default="JackFram/llama-68m")
    parser.add_argument('--target', type=str, help='target model', default="huggyllama/llama-7b")

    parser.add_argument('--dataset', type=str, default="cnn", help='dataset') # support: (wiki, cnn)
    parser.add_argument('--start', type=int, default=0, help='start')
    parser.add_argument('--end', type=int, default=200, help='end')

    parser.add_argument('--T', type=float, default=0.6, help='temperature')
    parser.add_argument('--P', type=float, default=0.9, help='top p')
    parser.add_argument('--M', type=int, default=384, help='max generation length')
    parser.add_argument('--B', type=int, default=128, help='max draft token budget')

    parser.add_argument('--use_bfs', action='store_true', help='use BFS for tree verification')
    parser.add_argument('--mode', type=str, default="sTree", help='tree mode') # support: (auto, sTree, dTree)

    parser.add_argument('--growmap', type=str, help='growmap path')
    args = parser.parse_args()
    print(args)

    return args

def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    args = parse_arguments()
    setup_seed(args.seed)
    
    ############ load models ############
    if args.target == "huggyllama/llama-7b":
        target_model = InferenceEngine(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, draft=False)
    elif args.target == "JackFram/llama-68m":
        target_model = InferenceEngine(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, draft=False)
    else:
        NotImplementedError("Unsupported target model")
    
    if args.draft == "JackFram/llama-68m" and args.mode != "auto":
        draft_model = InferenceEngine(max_length=args.M, model_name_or_path = args.draft, dtype = torch.float16, draft=True)


    ############ load dataset ############
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    eval_list = list(range(200, 2000))
    # random.shuffle(eval_list)
    tokenized_dataset = get_tokenized_dataset(tokenizer=tokenizer, dataset_name=args.dataset).select(eval_list[args.start :args.end])
    dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=False)

    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)

    ############# simulation setting #############
    if args.mode == "auto":
        Autoregressive(target_model=target_model, dataloader=dataloader, tokenizer=tokenizer, T=args.T, top_p=args.P, max_length=args.M, verbose=args.verbose)

    elif args.mode == "sTree":
        residual_graph = cuda_graph_for_residual()
        
        grow_map = torch.load(args.growmap)
        tree_size = grow_map["size"]
        idx_lists = grow_map["roots"]
        branch_lists = grow_map['branches']
        draft_step = len(grow_map["roots"])
        
        sampling_callables = {}
        sample_gather_indices = {}
        for i in range(draft_step - 1):
            idx_len = len(idx_lists[i])
            num_samples = max(branch_lists[i])
            sampling_callables[i] = cuda_graph_for_sampling_without_replacement(
                max_length=args.M, idx_len=idx_len, num_samples=num_samples,
                temperature=args.T, tree_size=tree_size) 
        for i in range(draft_step - 1):
            ith_gather_list = []
            max_num_samples = max(branch_lists[i])
            for j, branch in enumerate(branch_lists[i]):
                branch_index = torch.arange(branch, device="cuda:0", dtype=torch.long)
                branch_index = branch_index + j * max_num_samples
                ith_gather_list.append(branch_index)
            ith_gather_list = torch.cat(ith_gather_list)
            sample_gather_indices[i] = ith_gather_list

        Sequoia(target_model=target_model, draft_model=draft_model, dataloader=dataloader, tokenizer=tokenizer, T=args.T, top_p=args.P,
                max_length=args.M, bfs_verify=args.use_bfs, residual_graph = residual_graph, grow_map = grow_map, sampling_callables=sampling_callables, 
                sample_gather_indices = sample_gather_indices, verbose=args.verbose)
        
    elif args.mode == "dTree":
        pass