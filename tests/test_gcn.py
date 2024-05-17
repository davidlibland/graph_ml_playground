"""Tests the GCN model."""
import pytest
import torch

from graph_ml_playground.gcn import GCNLayer

torch.manual_seed(13)


@pytest.mark.parametrize("input_dim, output_dim, edge_dim, n, n_edges", [
    (3, 4, 5, 7, 4), (4, 5, 6, 3, 2), (1, 2, 3, 4, 0)
])
def test_gcn_layer(input_dim, output_dim, edge_dim, n, n_edges):
    """Tests the gcn layer"""
    gcn_layer = GCNLayer(input_dim, output_dim, edge_dim)
    edge_features = torch.randn(n_edges, edge_dim)
    node_features = torch.randn(n, input_dim)
    edge_indices = torch.randint(0, n, (n_edges, 2))
    output = gcn_layer(edge_features, node_features, edge_indices)
    assert output.shape == (n, output_dim)

    # Check that gradients are non-trivial:
    for param in gcn_layer.parameters():
        assert param.grad is None
    scalar_output = output.sum()
    scalar_output.backward()
    for name, param in gcn_layer.named_parameters():
        if n_edges == 0:
            assert param.grad is not None
            assert torch.norm(param.grad) == 0, f"{name} has a gradient"
        else:
            assert param.grad is not None
            assert torch.norm(param.grad) > 0, f"{name} has zero gradient"
