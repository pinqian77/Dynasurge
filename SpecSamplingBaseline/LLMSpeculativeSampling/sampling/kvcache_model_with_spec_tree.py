import torch
from typing import Optional

from sampling.utils import norm_logits, sample, sample_with_replacement
from transformers.models.bloom.modeling_bloom import BloomForCausalLM


gamma_to_subnode_nums = {}
for i in range(1, 100):
    if i <= 1:
        gamma_to_subnode_nums[i] = 1
    elif i <= 6:
        gamma_to_subnode_nums[i] = 2
    elif i <= 39:
        gamma_to_subnode_nums[i] = 3
    else:
        gamma_to_subnode_nums[i] = 4


class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class Tree:
    def __init__(self):
        self.nodes = {}

    def get_node(self, value):
        if value not in self.nodes:
            self.nodes[value] = Node(value)
        return self.nodes[value]

    def add_edge(self, parent, child):
        parent_node = self.get_node(parent)
        child_node = self.get_node(child)
        parent_node.add_child(child_node)


def build_tree(edges):
    tree = Tree()
    for parent, child in edges:
        tree.add_edge(parent, child)
    return tree


def _debug_show_kvcache(past_key_values):
    if past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break


class KVCacheModel_tree():
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids: torch.Tensor, use_debug=True):

        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k,
                                                          self._top_p)
            self._past_key_values = outputs.past_key_values
            last_token_logits = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            k, v = self._past_key_values[-1]
            cached_len = k.shape[2]

            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)

            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)

            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)

            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)

            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)

            last_token_logits = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values

        return last_token_logits, self._past_key_values

    def _generate_with_kvcache(self, prefix: torch.Tensor,
                               gamma: int,
                               max_subnode_num: int,
                               use_debug=False,
                               ):
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        current_node_num = 1
        node_id_map = {}
        remaining_tokens = gamma

        prefix_dict = {}
        prefix_dict['text'] = prefix
        prefix_dict['last_token'] = prefix[-1]
        prefix_dict['last_token_id'] = 0
        tree_edges = []
        waiting_for_process = [prefix_dict]

        while waiting_for_process and remaining_tokens > 0:
            x = waiting_for_process.pop(0)
            current_draw_tokens_num = max_subnode_num if max_subnode_num < remaining_tokens else remaining_tokens
            if self._past_key_values:
                self.rollback(x['text'].shape[1]-1)
            q, kv_cache = self._forward_with_kvcache(x['text'], use_debug)
            next_tokens = sample_with_replacement(q, current_draw_tokens_num)
            next_tokens_no_repeat = set(next_tokens[0].tolist())
            remaining_tokens -= len(next_tokens_no_repeat)

            for tok in next_tokens_no_repeat:
                node_id_map[current_node_num] = tok
                new_token_dict = {}
                new_token_dict['text'] = torch.cat((x['text'], torch.tensor([[tok]], device=x['text'].device)), dim=1)
                new_token_dict['last_token'] = tok
                new_token_dict['last_token_id'] = current_node_num
                waiting_for_process.append(new_token_dict)
                tree_edges.append((x['last_token_id'], current_node_num))
                current_node_num += 1

        tree = build_tree(tree_edges)
        return tree, node_id_map

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int, model_type: str) -> torch.Tensor:
        if model_type == 'approx':
            max_subnode_num = gamma_to_subnode_nums[gamma]
            tree, node_id_map = self._generate_with_kvcache(input, gamma, max_subnode_num)
            # flatten the tree
            input_length = input.shape[1]
            flatten_append_node_length = len(node_id_map)
            flatten_append_text = torch.tensor([node_id_map[i] for i in node_id_map]).unsqueeze(0)
            output_text= torch.cat((input, flatten_append_text), dim=1)
            print(1)
            causal_mask = torch.triu(torch.ones((flatten_append_node_length, flatten_append_node_length), dtype=torch.bool), 1)


            # check all generated tokens



        elif model_type == 'target':
            output = self._generate_with_kvcache(input, gamma)
        return output

    @torch.no_grad()
    def rollback(self, end_pos: int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)

            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)

        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]
