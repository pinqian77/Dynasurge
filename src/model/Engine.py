import torch
from .Llama_KV import KV_Cache
from .Llama_model import LlamaForCausalLM_FI, LlamaForCausalLM_TG
from typing import List, Optional, Tuple, Union
import gc
import accelerate

class InferenceEngine:
    def __init__(self, 
        max_length:int,
        model_name_or_path :str,
        dtype = torch.float16,
        draft = True,
        device = "cuda:0") -> None:
        
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        if draft:
            self.model = LlamaForCausalLM_FI.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map=device)
        else:
            self.model = LlamaForCausalLM_TG.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map=device)

        self.model.eval()
        self.model_config = self.model.config

        self.kv_cache = KV_Cache(config=self.model_config, max_length=max_length, device=device, dtype=dtype)
    
    @torch.no_grad()
    def inference(self,
            input_ids: torch.LongTensor, 
            storage_ids :torch.LongTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attn_mask: Optional[torch.Tensor] = None):
        
        logits = self.model(input_ids=input_ids, 
                    max_length=self.max_length, storage_ids=storage_ids,
                    attention_mask=attn_mask, position_ids=position_ids,
                    kv_cache=self.kv_cache)

        return logits
    
    def clear_kv(self):
        self.kv_cache.clear()
    
    def set_kv_len(self, kv_len :int):
        self.kv_cache.set_kv_len(kv_len)
    
    def initialize_kv(self, k_cache :torch.Tensor, v_cache :torch.Tensor, kv_len :int):
        self.kv_cache.initialize_kv(k_cache, v_cache, kv_len)
    
    def gather_kv(self, indices: list[int]):
        self.kv_cache.gather_kv(indices)

    def get_kv_cache(self, in_place=False):
        if not in_place:
            return self.kv_cache.k_cache.clone(), self.kv_cache.v_cache.clone()
        else:
            return self.kv_cache.k_cache, self.kv_cache.v_cache








