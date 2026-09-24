import torch

from untangle.wrappers.duq_wrapper import DUQHead


def test_duq_logits_remain_finite_when_rbf_values_underflow() -> None:
    head = DUQHead(
        num_classes=2,
        num_features=1,
        rbf_length_scale=0.1,
        ema_momentum=0.999,
        num_hidden_features=1,
    )
    head.eval()

    with torch.no_grad():
        head._weight.fill_(1)
        head._ema_num_samples_per_class.fill_(1)
        head._ema_embedding_sums_per_class.copy_(torch.tensor([[0.0], [1.0]]))

    output = head(torch.tensor([[100.0]]))

    assert torch.isfinite(output["logit"]).all()
    assert torch.equal(output["duq_value"], torch.ones(1))
