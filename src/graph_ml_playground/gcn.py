"""A simple graph convolutional network implementation."""
from typing import Any

import torch
import torch.nn as nn

import lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


class GCNLayer(nn.Module):
    """A simple graph convolutional layer."""
    def __init__(self, input_dim, output_dim, edge_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.edge_linear = nn.Linear(edge_dim + output_dim, output_dim)
        self.out_dim = output_dim

    def forward(self, edge_features, node_features, edge_indices):
        """
        Forward pass through the layer.

        Args:
            edge_features: The edge features, of shape (n_edges, in_features).
            node_features: The node features, of shape (n_nodes, in_features).
            edge_indices: The edge indices, of shape (n_edges, 2).
        """
        # First apply a linear layer to the node features:
        node_features = self.linear(node_features)
        # Next layer-norm them:
        node_features = torch.nn.functional.layer_norm(node_features, normalized_shape=node_features.shape)
        # Now we apply the non-linearity:
        node_features = torch.nn.functional.gelu(node_features)

        # Now these node features need to be pulled onto each edge:
        gather_indices = edge_indices[:, 0].unsqueeze(1).expand(-1, node_features.shape[1])
        gathered_features = torch.gather(node_features, 0, gather_indices)
        # gathered_features = node_features[edge_indices[:, 0]]

        # Now we concatenate the edge features with the gathered features:
        edge_features = torch.cat([edge_features, gathered_features], dim=1)
        # Now apply a linear layer to these edge features:
        edge_features = self.edge_linear(edge_features)
        # Next layer-norm them:
        edge_features = torch.nn.functional.layer_norm(edge_features, normalized_shape=edge_features.shape)
        # Now we apply the non-linearity:
        edge_features = torch.nn.functional.gelu(edge_features)

        # Finally, we scatter the edge features onto each node:
        scatter_indices = edge_indices[:, 1].unsqueeze(1).expand(-1, node_features.shape[1])

        output = torch.zeros(node_features.shape[0], self.out_dim, device=node_features.device, dtype=node_features.dtype, requires_grad=False)
        output = torch.scatter_reduce(output, 0, scatter_indices, edge_features, "mean", include_self=False)
        return output


class GCNClasifier(pl.LightningModule):
    def __init__(self, node_dim, edge_dim, hidden_dim, depth, learning_rate=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.ModuleList()
        self.input_layer = GCNLayer(node_dim, hidden_dim, edge_dim)
        for _ in range(1, depth-1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, edge_dim))
        self.output_layer = GCNLayer(hidden_dim, 1, edge_dim)
        self.metrics = nn.ModuleDict({
            "accuracy": BinaryAccuracy(),
            "auc": BinaryAUROC()
        })

    def forward(self, batch):
        node_features, edge_index, edge_features = batch.x, batch.edge_index, batch.edge_attr
        x = self.input_layer(edge_features, node_features, edge_index.T)
        for layer in self.layers:
            x = x + layer(edge_features, x, edge_index.T)
        x = self.output_layer(edge_features, x, edge_index.T)
        y_out = torch.zeros(batch.num_graphs, 1, device=x.device, dtype=x.dtype, requires_grad=False)
        y_out = torch.scatter_reduce(y_out, 0, batch.batch.unsqueeze(1), x, "mean", include_self=False)
        return y_out

    def training_step(self, batch, batch_idx):
        y_out = self.forward(batch)
        y = batch.y
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_out.flatten(), y.to(y_out)).mean()
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_after_backward(self) -> None:
        for param in self.parameters():
            if torch.isnan(param.grad).any():
                raise ValueError("NaNs in gradients")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        y = batch.y
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.flatten(), y.to(logits)).mean()
        self.log("cross_entropy", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for name, metric in self.metrics.items():
            metric(logits.flatten(), y)
            self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

