import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from untangle.wrappers.swag_wrapper import SWAGWrapper


def test_swag_recomputes_batch_norm_for_single_batch_loader() -> None:
    wrapper = SWAGWrapper.__new__(SWAGWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    wrapper._sampled_params_swag = torch.empty(0)
    wrapper._modules_and_names = []
    batch_norm = wrapper.model[0]
    batch_norm.register_buffer("running_means_swag", torch.zeros(0, 2))
    batch_norm.register_buffer("running_vars_swag", torch.zeros(0, 2))
    batch_norm.register_buffer(
        "num_batches_tracked_swag", torch.zeros(0, dtype=torch.long)
    )

    loader = DataLoader(
        TensorDataset(torch.tensor([[2.0, 4.0], [4.0, 8.0]]), torch.zeros(2)),
        batch_size=2,
    )
    wrapper._set_and_store_bn_stats(loader, fraction=0.1, channels_last=False)

    assert batch_norm.num_batches_tracked.item() == 1
    assert batch_norm.running_means_swag.shape == (1, 2)
