#!/bin/bash

mkdir -p log
log_file="log/$(date +%Y%m%d%H%M)_results.log"

END=200
VERBOSE_MODE=""
# VERBOSE_MODE="--verbose"

# exec > $log_file 2>&1

echo "########################## Dataset cnn #########################" >> $log_file
echo "========= Autoregressive ========" >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --mode auto --dataset cnn $VERBOSE_MODE >> $log_file
echo >> $log_file

echo >> $log_file
echo "========== Static Tree ==========" >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --growmap ./growmaps/4x8-tree.pt  --mode sTree --dataset cnn $VERBOSE_MODE >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --use_bfs --growmap ./growmaps/4x8-tree.pt  --mode sTree --dataset cnn $VERBOSE_MODE >> $log_file
echo >> $log_file

CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --growmap ./growmaps/8x8-tree.pt  --mode sTree --dataset cnn $VERBOSE_MODE >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --use_bfs --growmap ./growmaps/8x8-tree.pt  --mode sTree --dataset cnn $VERBOSE_MODE >> $log_file
echo >> $log_file

CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --growmap ./growmaps/L40-CNN-68m-7b-greedy.pt  --mode sTree --dataset cnn $VERBOSE_MODE >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --use_bfs --growmap ./growmaps/L40-CNN-68m-7b-greedy.pt  --mode sTree --dataset cnn $VERBOSE_MODE >> $log_file
echo >> $log_file

CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --growmap ./growmaps/A100-CNN-68m-7b-greedy.pt  --mode sTree --dataset cnn $VERBOSE_MODE >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --use_bfs --growmap ./growmaps/A100-CNN-68m-7b-greedy.pt  --mode sTree --dataset cnn $VERBOSE_MODE >> $log_file
echo >> $log_file

echo "========== Dynamic Tree ==========" >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --B 16 --growmap ./growmaps/8x8-tree.pt --mode dTree --dataset cnn $VERBOSE_MODE >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --B 16 --use_bfs --growmap ./growmaps/8x8-tree.pt --mode dTree --dataset cnn $VERBOSE_MODE >> $log_file
echo >> $log_file

CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --B 32 --growmap ./growmaps/8x8-tree.pt --mode dTree --dataset cnn $VERBOSE_MODE >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --B 32 --use_bfs --growmap ./growmaps/8x8-tree.pt --mode dTree --dataset cnn $VERBOSE_MODE >> $log_file
echo >> $log_file

CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --B 64 --growmap ./growmaps/8x8-tree.pt --mode dTree --dataset cnn $VERBOSE_MODE >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --B 64 --use_bfs --growmap ./growmaps/8x8-tree.pt --mode dTree --dataset cnn $VERBOSE_MODE >> $log_file
echo >> $log_file

CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --B 128 --growmap ./growmaps/8x8-tree.pt --mode dTree --dataset cnn $VERBOSE_MODE >> $log_file
CUDA_VISIBLE_DEVICES=0 python main.py --draft JackFram/llama-68m --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end ${END} --M 384 --B 128 --use_bfs --growmap ./growmaps/8x8-tree.pt --mode dTree --dataset cnn $VERBOSE_MODE >> $log_file
echo >> $log_file