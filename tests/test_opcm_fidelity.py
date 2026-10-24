"""
Fidelity test: OPCMMerger.merge_state_dict vs. a verbatim transcription of the
official OPCMForCLIP.run / merge_linear_weights / merge_other_parameters
(external/opcm_author_repo/fusion_bench/method/opcm/opcm.py).

The official module cannot be imported here (needs lightning/fusion_bench), so the
reference below copies its arithmetic line by line, operating on real nn.Modules,
with only logging/IO/fabric removed.
"""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from mm_flex_merge.merging import OPCMMerger


# ---------------------------------------------------------------------------
# Verbatim reference (official logic)
# ---------------------------------------------------------------------------

def _svd(w, full_matrices=True):
    u, s, vh = torch.linalg.svd(w, full_matrices=full_matrices)
    return u, s, vh.T


def _is_leaf_module(module):
    return len(list(module.children())) == 0


def _get_task_vector_norm(model, pretrained_model):
    sd, psd = model.state_dict(), pretrained_model.state_dict()
    return torch.linalg.norm(torch.cat([(sd[k] - psd[k]).flatten() for k in sd]))


class _OfficialOPCM:
    def __init__(self, alpha):
        self.alpha = alpha

    def merge_linear_weights(self, merged_W, pretrained_W, task_W, alpha):
        previous_merged_tv = merged_W - pretrained_W
        task_tv = task_W - pretrained_W
        u, s, v = _svd(previous_merged_tv)
        split_rank = (s.cumsum(dim=0) / s.sum() > alpha).float().argmax().item()
        projected_task_tv = u.T @ task_tv @ v
        projected_task_tv.diag().fill_(0)
        projected_task_tv[:split_rank, :split_rank] = 0
        cleaned_task_tv = u @ projected_task_tv @ v.T
        return pretrained_W + (self.previous_lambda_t * previous_merged_tv + cleaned_task_tv) / self.lambda_t

    def merge_other_parameters(self, merged_W, pretrained_W, task_W):
        previous_merged_tv = merged_W - pretrained_W
        task_tv = task_W - pretrained_W
        return pretrained_W + (self.previous_lambda_t * previous_merged_tv + task_tv) / self.lambda_t

    @torch.no_grad()
    def run(self, pretrained_model, task_models):
        merged_model = copy.deepcopy(task_models[0])
        self.avg_task_vector_norm = _get_task_vector_norm(merged_model, pretrained_model)
        self.all_task_vector_norm = [self.avg_task_vector_norm]
        self.previous_lambda_t = 1
        for task_model in task_models[1:]:
            self.all_task_vector_norm.append(_get_task_vector_norm(task_model, pretrained_model))
            self.avg_task_vector_norm = np.mean(self.all_task_vector_norm)
            self.lambda_t = 1
            for module_name, module in list(merged_model.named_modules()):
                if not _is_leaf_module(module):
                    continue
                if isinstance(module, nn.Linear):
                    module.weight.data = self.merge_linear_weights(
                        module.weight,
                        pretrained_model.get_submodule(module_name).weight,
                        task_model.get_submodule(module_name).weight,
                        alpha=self.alpha,
                    )
                    if module.bias is not None:
                        module.bias.data = self.merge_other_parameters(
                            module.bias,
                            pretrained_model.get_submodule(module_name).bias,
                            task_model.get_submodule(module_name).bias,
                        )
                else:
                    for param_name, param in module.named_parameters():
                        param.data = self.merge_other_parameters(
                            param,
                            pretrained_model.get_submodule(module_name).get_parameter(param_name),
                            task_model.get_submodule(module_name).get_parameter(param_name),
                        )
            task_vector_norm = _get_task_vector_norm(merged_model, pretrained_model)
            self.lambda_t *= task_vector_norm / self.avg_task_vector_norm
            for param_name, param in merged_model.named_parameters():
                param.data = pretrained_model.get_parameter(param_name) + (
                    param - pretrained_model.get_parameter(param_name)
                ) * (self.avg_task_vector_norm / task_vector_norm)
            self.previous_lambda_t = self.lambda_t
        return merged_model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_model():
    return nn.Sequential(nn.Linear(12, 8), nn.LayerNorm(8), nn.Linear(8, 5)).double()


def _make_pool(num_tasks, seed):
    torch.manual_seed(seed)
    pretrained = _make_model()
    tasks = []
    for _ in range(num_tasks):
        m = copy.deepcopy(pretrained)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.1 * torch.randn_like(p))
        tasks.append(m)
    return pretrained, tasks


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("num_tasks", [2, 4])
def test_layerwise_opcm_matches_official(alpha, num_tasks):
    pretrained, tasks = _make_pool(num_tasks, seed=7)
    expected = _OfficialOPCM(alpha).run(pretrained, tasks).state_dict()

    merged = OPCMMerger(alpha=alpha).merge_state_dict(
        pretrained.state_dict(), [t.state_dict() for t in tasks]
    )

    assert set(merged) == set(expected)
    for k in expected:
        torch.testing.assert_close(merged[k], expected[k], rtol=1e-9, atol=1e-10)


def test_official_diag_fill_is_noop():
    """Documents why the port does not zero the diagonal."""
    p = torch.ones(3, 4)
    p.diag().fill_(0)
    assert torch.all(p.diagonal() == 1)


def test_merged_task_vector_norm_equals_mean_task_norm():
    pretrained, tasks = _make_pool(3, seed=11)
    psd = pretrained.state_dict()
    merged = OPCMMerger(alpha=0.5).merge_state_dict(psd, [t.state_dict() for t in tasks])

    def norm(sd):
        return torch.linalg.norm(torch.cat([(sd[k] - psd[k]).flatten() for k in psd]))

    mean_norm = torch.stack([norm(t.state_dict()) for t in tasks]).mean()
    torch.testing.assert_close(norm(merged), mean_norm)
