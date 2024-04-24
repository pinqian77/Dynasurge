#!/bin/bash

mkdir -p log
log_file="log/$(date +%Y%m%d%H%M)_results.log"

echo "########## Dataset cnn ##########" >> $log_file
echo "========= Autoregressive ========" >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --mode auto --dataset cnn >> $log_file

echo "========== Static Tree ==========" >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ./growmaps/4x8-tree.pt  --mode sTree --dataset cnn >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --use_bfs --growmap ./growmaps/8x8-tree.pt  --mode sTree --dataset cnn >> $log_file

CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ./growmaps/8x8-tree.pt  --mode sTree --dataset cnn >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --use_bfs --growmap ./growmaps/8x8-tree.pt  --mode sTree --dataset cnn >> $log_file

CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ./growmaps/demo_tree.pt  --mode sTree --dataset cnn >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --use_bfs --growmap ./growmaps/demo_tree.pt  --mode sTree --dataset cnn >> r$log_file



echo "########## Dataset wiki ##########" >> $log_file
echo "========= Autoregressive ========" >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --mode auto --dataset wiki >> $log_file

echo "========== Static Tree ==========" >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ./growmaps/4x8-tree.pt  --mode sTree --dataset wiki >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --use_bfs --growmap ./growmaps/8x8-tree.pt  --mode sTree --dataset wiki >> r$log_file

CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ./growmaps/8x8-tree.pt  --mode sTree --dataset wiki >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --use_bfs --growmap ./growmaps/8x8-tree.pt  --mode sTree --dataset wiki >> r$log_file

CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ./growmaps/demo_tree.pt  --mode sTree --dataset wiki >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --use_bfs --growmap ./growmaps/demo_tree.pt  --mode sTree --dataset wiki >> $log_file