import dataclasses
from typing import List
import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        input_gate: nn.Module,
        task_gate: nn.Module,
        moe_args: MoeArgs,
        use_task_gate: bool = True,
        lb_weight: float = 1.0,
        alpha_init: float = 0.5,
        alpha_trainable: bool = True,
    ):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.input_gate = input_gate
        self.task_gate = task_gate
        self.args = moe_args
        self.use_task_gate = use_task_gate
        self.lb_weight = lb_weight

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha.requires_grad = bool(alpha_trainable)

    def forward(self, inputs: torch.Tensor, task_param, return_routing: bool = False):
        """
        inputs: [L, B, D]
        task_param: [D]
        """
        input_gate_logits = self.input_gate(inputs)  # [L, B, E]

        if self.use_task_gate:
            task_gate_logits = self.task_gate(task_param)  # [E]
            gate_logits = (1 - self.alpha) * input_gate_logits + self.alpha * task_gate_logits
        else:
            gate_logits = input_gate_logits

        gate_logits = torch.nan_to_num(gate_logits, nan=0.0, posinf=0.0, neginf=0.0)

        # Top-k routing
        topk_logits, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok, dim=-1
        )  # [L, B, K], [L, B, K]

        # Load balancing loss
        weights_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float32).to(inputs.dtype)
        average_weight = torch.mean(weights_softmax, dim=[0, 1])  # [E]

        indices_topk = F.one_hot(
            selected_experts, num_classes=self.args.num_experts
        ).sum(dim=2)  # [L, B, E]
        average_count = torch.mean(indices_topk.float(), dim=[0, 1]).to(inputs.dtype)  # [E]

        l_aux = torch.mean(average_weight * average_count) * self.args.num_experts
        l_aux = l_aux * self.lb_weight

        # softmax over selected experts only
        topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32).to(inputs.dtype)  # [L, B, K]

        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            idx_l, idx_b, idx_k = torch.where(selected_experts == i)
            if idx_l.numel() > 0:
                results[idx_l, idx_b] += topk_weights[idx_l, idx_b, idx_k, None] * expert(inputs[idx_l, idx_b])

        if return_routing:
            routing_info = {
                "gate_logits": gate_logits.detach().cpu(),           # [L, B, E]
                "selected_experts": selected_experts.detach().cpu(), # [L, B, K]
                "topk_weights": topk_weights.detach().cpu(),         # [L, B, K]
                "average_weight": average_weight.detach().cpu(),     # [E]
                "average_count": average_count.detach().cpu(),       # [E]
                "alpha": self.alpha.detach().cpu(),                  # scalar
            }
            return results, l_aux.float(), routing_info

        return results, l_aux.float()