import torch
import pytest
from starmap4astron.models.encoder import LightcurveEncoder1D, ResBlock1D

#Run python -m pytest -q


def test_encoder_output_shape():
    # smaller sizes to speed up
    enc = LightcurveEncoder1D(
        latent_channels=16,
        grid_size=(4, 4),
        in_channels=1,
        base_channels=8
    )
    B, L = 3, 256
    x = torch.randn(B, 1, L)
    z = enc(x)
    assert z.shape == (B, 16, 4, 4), \
        f"Expected (3,16,4,4), got {z.shape}"


def test_encoder_zero_input():
    enc = LightcurveEncoder1D(
        latent_channels=8,
        grid_size=(2, 3),
        in_channels=1,
        base_channels=4
    )
    B, L = 5, 128
    x = torch.zeros(B, 1, L)
    z = enc(x)
    # With zero input and default biases, output should be exactly zero
    assert torch.allclose(z, torch.zeros_like(z)), "Zero input must give all-zero latent"


def test_encoder_grad_flow():
    enc = LightcurveEncoder1D()
    B, L = 2, 512
    x = torch.randn(B, 1, L, requires_grad=True)
    z = enc(x)
    loss = z.sum()
    loss.backward()
    # ensure gradients backpropagate into the input
    assert x.grad is not None, "No gradient to input"
    assert torch.any(x.grad != 0), "Gradient is all zero"


def test_resblock1d_shape():
    block = ResBlock1D(channels=3, dilation=2)
    B, C, L = 4, 3, 64
    x = torch.randn(B, C, L)
    y = block(x)
    # shape must be unchanged
    assert y.shape == x.shape, f"ResBlock changed shape: {y.shape} vs {x.shape}"
