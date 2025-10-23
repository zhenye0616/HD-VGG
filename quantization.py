import torch


def fake_quantize_tensor(tensor, num_bits=5):
    """Simulate quantization and dequantization on a tensor (activation or weight)."""
    qmin, qmax = 0, 2 ** num_bits - 1
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)
    q_tensor = torch.round(tensor / scale + zero_point)
    dq_tensor = (q_tensor - zero_point) * scale
    return dq_tensor


def apply_fake_quantization(model, num_bits=5):
    """Apply fake quantization to model weights."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(fake_quantize_tensor(param, num_bits))
    return model


class QuantizedWrapper(torch.nn.Module):
    """
    Wraps a model so that all inputs/outputs are fake-quantized during forward pass.
    This simulates activation quantization.
    """
    def __init__(self, model, num_bits=5):
        super().__init__()
        self.model = model
        self.num_bits = num_bits

    def forward(self, x):
        # Quantize input activation
        x = fake_quantize_tensor(x, self.num_bits)
        out = self.model(x)
        # Optionally quantize output as well
        out = fake_quantize_tensor(out, self.num_bits)
        return out