import torch
import torch.nn as nn


def _compute_quant_params(tensor, num_bits=5):
    """
    Compute per-tensor quantization parameters matching fake quant logic.
    Returns scale (quantization step) and zero-point.
    """
    qmin, qmax = 0, 2 ** num_bits - 1
    min_val, max_val = tensor.detach().min(), tensor.detach().max()
    dynamic_range = max_val - min_val
    scale = dynamic_range / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)
    return scale, zero_point, dynamic_range


def quantization_step(tensor, num_bits=5):
    """
    Derive the quantization step Δ for a tensor under uniform fake quantization.
    """
    scale, _, dynamic_range = _compute_quant_params(tensor, num_bits)
    if dynamic_range.abs().item() < 1e-12:
        # Return an all-zero tensor to signal a degenerate scale.
        return torch.zeros_like(scale)
    return scale


def fake_quantize_tensor(tensor, num_bits=5):
    """Simulate quantization and dequantization on a tensor (activation or weight)."""
    scale, zero_point, dynamic_range = _compute_quant_params(tensor, num_bits)
    if dynamic_range.abs().item() < 1e-12:
        return tensor
    scale = scale.clamp(min=1e-8)
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


class ActivationFakeQuant(nn.Module):
    """Apply fake quantization to an activation tensor."""

    def __init__(self, num_bits=5):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, x):
        return fake_quantize_tensor(x, self.num_bits)


def _is_already_wrapped(module):
    if isinstance(module, ActivationFakeQuant):
        return True
    if isinstance(module, nn.Sequential):
        return any(isinstance(sub, ActivationFakeQuant) for sub in module)
    return False


def insert_activation_fake_quant(module, num_bits=5, target_types=None):
    """
    Recursively wrap target layers so their outputs pass through fake quantization.
    """
    if target_types is None:
        target_types = (nn.ReLU,)

    for name, child in list(module.named_children()):
        insert_activation_fake_quant(child, num_bits, target_types)
        if isinstance(child, target_types) and not _is_already_wrapped(child):
            wrapped = nn.Sequential(child, ActivationFakeQuant(num_bits))
            setattr(module, name, wrapped)


def ensure_activation_fake_quant(model, num_bits=5, target_types=None):
    """
    Ensure per-layer activation fake quantization is applied once to the model.
    """
    if getattr(model, "_activation_fake_quant_enabled", False):
        return
    insert_activation_fake_quant(model, num_bits=num_bits, target_types=target_types)
    setattr(model, "_activation_fake_quant_enabled", True)


class GaussianActivationNoise(nn.Module):
    """
    Inject zero-mean Gaussian noise whose std matches the quantization step Δ.
    """

    def __init__(
        self,
        num_bits=5,
        sigma_factor=1.0,
        enabled=True,
        apply_during_eval=False,
        clamp=False,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.sigma_factor = sigma_factor
        self.enabled = enabled
        self.apply_during_eval = apply_during_eval
        self.clamp = clamp

    def forward(self, x):
        if not self.enabled:
            return x
        if not self.training and not self.apply_during_eval:
            return x
        if x.numel() == 0:
            return x

        delta = quantization_step(x, self.num_bits)
        # Avoid injecting noise when the tensor is effectively constant.
        if delta.abs().item() < 1e-8:
            return x

        std = delta * self.sigma_factor
        noise = torch.randn_like(x) * std
        noisy = x + noise

        if self.clamp:
            scale, zero_point, _ = _compute_quant_params(x, self.num_bits)
            qmin, qmax = 0, 2 ** self.num_bits - 1
            if scale.abs().item() > 1e-8:
                min_val = (qmin - zero_point) * scale
                max_val = (qmax - zero_point) * scale
                noisy = torch.clamp(noisy, min=min_val, max=max_val)
        return noisy


class QuantizedWrapper(torch.nn.Module):
    """
    Wraps a model so that all inputs/outputs are fake-quantized during forward pass.
    This simulates activation quantization.
    """
    def __init__(self, model, num_bits=5):
        super().__init__()
        self.model = model
        self.num_bits = num_bits
        ensure_activation_fake_quant(self.model, num_bits=self.num_bits)

    def forward(self, x):
        # Quantize input activation
        x = fake_quantize_tensor(x, self.num_bits)
        out = self.model(x)
        # Optionally quantize output as well
        out = fake_quantize_tensor(out, self.num_bits)
        return out

    def __getattr__(self, name):
        """
        Delegate missing attributes (e.g., hd_head, features) to the wrapped model so
        downstream code can interact with it transparently.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
