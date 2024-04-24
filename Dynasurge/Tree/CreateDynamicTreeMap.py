import torch
import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"

def sample_with_replacement(probs: torch.Tensor, num_samples: int = 1):
    probs = probs.softmax(dim=-1)
    idx_next = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    return idx_next


def show_tensor(x):
    for i in x.tolist():
        print(i)


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


def update_grow_map(grow_map: dict, current_layer: int):
    pass


def create_tree_map(draft_model, input_ids, max_subnodes_num, max_token_for_tree, max_length, vocab_size):
    device = input_ids.device
    input_ids = input_ids.squeeze()

    current_node_num = len(input_ids)

    # prevent over generate
    max_token_for_tree = min(max_token_for_tree, max_length - current_node_num)

    prefix_len = len(input_ids)
    node_id_map = {}
    remaining_tokens = max_token_for_tree

    dtype = torch.float16

    tokens = torch.zeros(max_length, dtype=torch.long, device=device)
    tokens[:current_node_num] = input_ids
    storage_ids = torch.arange(max_length).to(device)

    position_ids = torch.zeros(max_length, dtype=torch.long, device=device)
    position_ids[:current_node_num] = torch.arange(current_node_num)

    tree_attention_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, device=device, dtype=dtype, )
    rows, cols = torch.tril_indices(current_node_num, current_node_num, offset=0, device=device)
    tree_attention_mask[rows, cols] = 0

    draft_logits = torch.zeros((max_length, vocab_size), dtype=dtype).to(device)

    draft_model_outputs = draft_model.inference(tokens[:current_node_num].unsqueeze(0),
                                                storage_ids=storage_ids[:current_node_num],
                                                attn_mask=tree_attention_mask[:current_node_num][None, None, :, :],
                                                position_ids=position_ids[:current_node_num].unsqueeze(0))
    draft_logits[0] = draft_model_outputs[..., -1, :][0]

    grow_map = {
        'roots': [[0]],
        'branches': [],
        'Successors': [],
        'full_tokens_mask': tree_attention_mask,
        'depth': [0],
        'mask': torch.zeros((max_token_for_tree + 1, max_token_for_tree + 1), dtype=dtype, device=device),
        'size': max_token_for_tree + 1,
        'id_map': {},
    }
    grow_map['mask'][0, 0] = 1
    node_idx = 0
    idx_list = [0]

    draft_kv_len = current_node_num
    current_layer_id = 0

    branches = []

    while len(grow_map['depth']) < max_token_for_tree + 1:  # process each layer
        total_branch = 0
        current_layer_id += 1

        new_layer_roots = []
        current_layer_branches = []
        generated_tokens = []
        new_index_list = []
        for father_idx in idx_list:  # process each node(index)
            if remaining_tokens <= 0:
                current_layer_branches.append(0)
                grow_map['Successors'].append([])
                continue
            new_tokens = sample_with_replacement(draft_logits[father_idx], min(max_subnodes_num, remaining_tokens))
            new_tokens_no_repeat = set(new_tokens.tolist())
            current_node_successors = []
            current_node_branches = 0
            for tok in new_tokens_no_repeat:
                node_idx += 1
                remaining_tokens -= 1
                new_index_list.append(node_idx)
                new_layer_roots.append(node_idx)
                current_node_branches += 1
                current_node_successors.append(node_idx)
                generated_tokens.append(tok)
                grow_map['depth'].append(current_layer_id)
                total_branch += 1
                branches.append((father_idx, node_idx))
                grow_map['full_tokens_mask'][node_idx + prefix_len - 1] = grow_map['full_tokens_mask'][
                    father_idx + prefix_len - 1]
                grow_map['full_tokens_mask'][node_idx + prefix_len - 1, node_idx + prefix_len - 1] = 0
                grow_map['mask'][node_idx] = grow_map['mask'][father_idx]
                grow_map['mask'][node_idx, node_idx] = 1
                grow_map['id_map'][node_idx] = tok

                position_ids[node_idx + prefix_len - 1] = current_layer_id + prefix_len

            current_layer_branches.append(current_node_branches)
            grow_map['Successors'].append(current_node_successors)

        idx_list = new_index_list
        grow_map['branches'].append(current_layer_branches)
        grow_map['roots'].append(new_layer_roots)

        tokens[current_node_num:current_node_num + len(generated_tokens)] = torch.tensor(generated_tokens,
                                                                                         device=device)
        current_node_num += len(generated_tokens)

        draft_model_outputs = draft_model.inference(tokens[draft_kv_len:current_node_num].unsqueeze(0),
                                                    storage_ids=storage_ids[draft_kv_len:current_node_num],
                                                    attn_mask=grow_map['full_tokens_mask'][
                                                              current_node_num - total_branch:current_node_num][None,
                                                              None, :, :],
                                                    position_ids=position_ids[
                                                                 current_node_num - total_branch:current_node_num].unsqueeze(
                                                        0))
        draft_logits[current_node_num - total_branch - prefix_len + 1:current_node_num - prefix_len + 1] = \
            draft_model_outputs[0][-total_branch:]

        draft_kv_len = current_node_num

    grow_map['branches'].append([0] * len(grow_map['roots'][-1]))
    grow_map['Successors'].extend([[]] * len(grow_map['roots'][-1]))
    grow_map['depth'] = torch.tensor(grow_map['depth'], dtype=torch.long, device=device)

    construction_information = {
        'tokens': tokens,
        'position_ids': position_ids,
        'storage_ids': storage_ids,
        'draft_logits': draft_logits,
        'branches': branches,
        'full_attention_mask': grow_map['full_tokens_mask'],
        'num_nodes': current_node_num,
        'ground_truth_len': prefix_len,
    }

    assert len(grow_map['depth']) == max_token_for_tree + 1
    assert sum([len(i) for i in grow_map['roots']]) == max_token_for_tree + 1
    assert sum([len(i) for i in grow_map['branches']]) == max_token_for_tree + 1
    assert len(grow_map['Successors']) == max_token_for_tree + 1

    return grow_map, construction_information
