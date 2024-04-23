
# CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ./Dynasurge/growmaps/8x8-tree.pt  --mode auto --dataset cnn >> resultsv2.log
# CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ./Dynasurge/growmaps/8x8-tree.pt  --mode auto --dataset wiki >> resultsv2.log

CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384  --mode auto --dataset cnn >> results.log
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384  --mode auto --dataset wiki >> results.log

CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ./growmaps/8x8-tree.pt  --mode sTree --dataset cnn >> results.log
CUDA_VISIBLE_DEVICES=0 python main.py --draft  JackFram/llama-68m   --target huggyllama/llama-7b  --T 0.6 --P 1.0  --start 0 --end 2 --M 384 --growmap ./growmaps/8x8-tree.pt  --mode sTree --dataset wiki >> results.log