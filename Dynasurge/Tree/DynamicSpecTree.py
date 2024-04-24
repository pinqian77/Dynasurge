import torch
from torch.nn.functional import softmax
from .Tree import Tree
import time
from model.Engine import InferenceEngine
from utils.utils import get_sampling_logits, ChildrenAccept, get_residual


class DynamicSpecTree(Tree):
    def __init__(self,
                 draft_model_engine: InferenceEngine,
                 target_model_engine: InferenceEngine,
                 prefix: torch.LongTensor,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 draft_kv_len=0,
                 target_kv_len=0,
                 max_length=256,
                 device: str = 'cpu',
                 vocab_size=32000,
                 grow_map=None,
                 attn_mask=None,
                 sequence=None,
                 new_tokens_buffer=None,
                 parents_buffer=None,
                 position_ids=None,
                 residual_graph=None,
                 sampling_callables=None,
                 sample_gather_indices=None) -> None:
        super().__init__(device=device, max_length=max_length)
        assert self.max_length == draft_model_engine.max_length
        self.max_target_seq = draft_model_engine.max_length
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.temperature = temperature
        self.top_p = top_p
        self.residual_graph = residual_graph
        self.grow_map = grow_map
        self.sampling_callables = sampling_callables
        self.sample_gather_indices = sample_gather_indices
        self.draft_step = len(self.grow_map["roots"])
        self.grow_map_roots_gpu = []
        for x in self.grow_map["roots"]:
            self.grow_map_roots_gpu.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.grow_map["Successors"]
        tree_mask: torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 0).type(self.dtype)

        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)
        self.initialize(attn_mask, sequence, new_tokens_buffer, parents_buffer, position_ids, None)
        self.set_prefix(prefix=prefix)
        self.tree_size = self.grow_map["size"]
        self.tree_mask = tree_mask

        self.full_attn_mask[self.max_length - self.tree_size + 1: self.max_length,
        self.max_length - self.tree_size + 1: self.max_length] = tree_mask[1:, 1:]

        total_nodes = len(prefix) + self.tree_size - 1
        self.attn_mask = self.full_attn_mask[self.max_length - total_nodes: 2 * self.max_length - total_nodes,
                         self.max_length - total_nodes: 2 * self.max_length - total_nodes]
        self.ground_truth_len = len(prefix)
        self.r = torch.rand(len(position_ids), dtype=self.dtype).to(self.device)

        self.position_ids[len(prefix): len(prefix) + self.tree_size - 1] = (
                    self.grow_map["depth"][1:].to(self.device) + len(prefix) - 1)
        self.storage_ids = torch.arange(self.max_length).to(self.device)
        self.depth = self.grow_map["depth"][1:].to(self.device)

        self.draft_logits = torch.zeros((self.max_length, vocab_size), dtype=self.dtype).to(self.device)
        if draft_kv_len == 0:
            draft_model_outputs = self.draft_model_engine.inference(input_ids=self.tokens[:self.num_nodes].unsqueeze(0),
                                                                    storage_ids=self.storage_ids[:self.num_nodes],
                                                                    position_ids=self.position_ids[
                                                                                 :self.num_nodes].unsqueeze(0),
                                                                    attn_mask=self.attn_mask[:self.num_nodes][None,
                                                                              None, :, :])
            self.draft_logits[0] = draft_model_outputs[..., -1, :][0]

        else:
            draft_model_outputs = self.draft_model_engine.inference(
                input_ids=self.tokens[draft_kv_len: self.num_nodes].unsqueeze(0),
                storage_ids=self.storage_ids[draft_kv_len: self.num_nodes],
                position_ids=self.position_ids[draft_kv_len: self.num_nodes].unsqueeze(0),
                attn_mask=self.attn_mask[draft_kv_len: self.num_nodes][None, None, :, :])
            self.draft_logits[0] = draft_model_outputs[..., -1, :][0]
        self.draft_kv_len = self.num_nodes

        self.target_kv_len = target_kv_len

        self.rand = torch.empty((self.tree_size, self.draft_logits.shape[1]), dtype=self.dtype).uniform_().to(
            self.device)
        self.seq_to_use = list(range(self.max_length))


    @torch.inference_mode()
    def accept_step(self, parent_id: int):
        logits_id = parent_id - (self.ground_truth_len - 1)
        p = self.target_logits[logits_id]
        draft_logits = self.draft_logits[logits_id]

        children = self.Successors[logits_id]
        if len(children) == 0:
            return (-1, p)

        for pos in children:

            token = self.tokens[pos + (self.ground_truth_len - 1)]
            q = softmax(draft_logits / self.temperature, dim=-1)
            r = self.r[pos + (self.ground_truth_len - 1)]

            if p[token] > r * q[token]:
                return (pos + (self.ground_truth_len - 1), None)
            else:
                p = self.residual_graph(p, q)
                draft_logits[token] = torch.finfo(self.dtype).min
        return (-1, p)

    @torch.inference_mode()
    def verify(self, benchmark=False):
        new_node_num = (self.num_nodes - self.ground_truth_len + 1)
        if self.target_kv_len == 0:
            start_pos = 0
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask = attn_mask[None, None, :, :].type(self.target_model_engine.dtype)
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
            target_model_outputs = self.target_model_engine.inference(
                input_ids=self.tokens[start_pos: end_pos].unsqueeze(0),
                position_ids=self.position_ids[start_pos: end_pos].unsqueeze(0), attn_mask=attn_mask,
                storage_ids=self.storage_ids[start_pos: end_pos])
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
            self.target_logits: torch.FloatTensor = target_model_outputs[0][self.ground_truth_len - 1:]

        else:
            start_pos = self.target_kv_len
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask = attn_mask[None, None, :, :].type(self.target_model_engine.dtype)
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
            target_model_outputs = self.target_model_engine.inference(
                input_ids=self.tokens[start_pos: end_pos].unsqueeze(0),
                position_ids=self.position_ids[start_pos: end_pos].unsqueeze(0), attn_mask=attn_mask,
                storage_ids=self.storage_ids[start_pos: end_pos])
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
            self.target_logits: torch.FloatTensor = target_model_outputs[0][-(new_node_num):]

        assert len(self.target_logits) == (self.num_nodes - self.ground_truth_len + 1)

        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature,
                                                 replicate=False)

        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)

        accept_list = self.seq_to_use[:self.ground_truth_len]

        terminal = False
        while True:
            parent_id = accept_list[-1]
            pos, res = self.accept_step(parent_id=parent_id)
            if pos != -1:
                accept_list.append(pos)
                if self.tokens[pos] == 0 or self.tokens[pos] == 2:
                    terminal = True
                    break
            else:
                residual = res
                break
        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
        accept_length = len(accept_list)
        if not terminal:
            if torch.isnan(residual).any():
                terminal = True
            else:
                self.tokens[accept_length] = residual.multinomial(num_samples=1, replacement=True)

        self.tokens[:accept_length] = self.tokens[accept_list]

        self.draft_model_engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:],
                                                                      self.ground_truth_len)
        self.target_model_engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:],
                                                                       self.ground_truth_len)

        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                self.prepare_for_next_iter(accept_list, self.tokens[:accept_length + 1])
                return self.tokens[
                       :accept_length + 1], accept_length, accept_length, t2 - t1, t3 - t2, t4 - t3, terminal
            self.prepare_for_next_iter(accept_list, self.tokens[:accept_length + 1])
            return self.tokens[:accept_length + 1], accept_length, accept_length, terminal
        else:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                return self.tokens[:accept_length], accept_length, accept_length, t2 - t1, t3 - t2, t4 - t3, terminal
            return self.tokens[:accept_length], accept_length, accept_length, terminal

    def verbose(self):
        super().verbose()


    def prepare_for_next_iter(self, accept_list: list[int], valid_tokens: torch.LongTensor):
        if len(accept_list) + 1 > self.max_target_seq:
            return
        self.position_ids[:len(accept_list)] = self.position_ids[accept_list]
        self.position_ids[len(accept_list)] = len(accept_list)
        self.position_ids[len(valid_tokens): len(valid_tokens) + self.tree_size - 1] = (
                    self.depth + len(valid_tokens) - 1)[:min(self.tree_size - 1, self.max_target_seq - len(accept_list)-1)]
        self.ground_truth_len = len(valid_tokens)
        self.num_nodes = len(valid_tokens)

        total_nodes = len(valid_tokens) + self.tree_size - 1
        self.attn_mask = self.full_attn_mask[self.max_length - total_nodes: 2 * self.max_length - total_nodes,
                         self.max_length - total_nodes: 2 * self.max_length - total_nodes]

        # draft_model_outputs = self.draft_model_engine.inference(
        #     input_ids=self.tokens[len(accept_list): self.num_nodes].unsqueeze(0),
        #     storage_ids=self.storage_ids[len(accept_list): self.num_nodes],
        #     position_ids=self.position_ids[len(accept_list): self.num_nodes].unsqueeze(0),
        #     attn_mask=self.attn_mask[len(accept_list): self.num_nodes][None, None, :, :])

        # self.draft_logits[0] = draft_model_outputs[..., -1, :][0]
        self.draft_kv_len = self.num_nodes
        self.target_kv_len = len(accept_list)


    def override_information(self, grow_map,construction_information):
        self.num_nodes = construction_information['num_nodes']
        self.ground_truth_len = construction_information['ground_truth_len']
        self.attn_mask = construction_information['full_attention_mask']
        self.tokens = construction_information['tokens']
        self.position_ids = construction_information['position_ids']
        self.storage_ids = construction_information['storage_ids']
        self.draft_logits = construction_information['draft_logits']

        self.grow_map = grow_map
        self.grow_map_roots_gpu = []
        for x in self.grow_map["roots"]:
            self.grow_map_roots_gpu.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.grow_map["Successors"]
        tree_mask: torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 0).type(self.dtype)

        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)
        self.tree_size = self.grow_map["size"]
        self.depth = self.grow_map["depth"][1:].to(self.device)






