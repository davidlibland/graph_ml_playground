"""Train on the mutag dataset"""
import torch
import torch_geometric
import torch_geometric.loader as geom_loader
import lightning as pl

from graph_ml_playground.gcn import GCNClasifier


DATASET_PATH = "data"

tu_dataset = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name="MUTAG")
pl.seed_everything(13)

torch.manual_seed(42)
tu_dataset.shuffle()
train_dataset = tu_dataset[:150]
test_dataset = tu_dataset[150:]

graph_train_loader = geom_loader.DataLoader(train_dataset, batch_size=64, shuffle=True)
graph_val_loader = geom_loader.DataLoader(test_dataset, batch_size=64) # Additional loader if you want to change to a larger dataset
graph_test_loader = geom_loader.DataLoader(test_dataset, batch_size=64)


def train():
    """Train the model"""
    model = GCNClasifier(
        node_dim=tu_dataset.num_node_features,
        edge_dim=tu_dataset.num_edge_features,
        hidden_dim=32,
        depth=2)

    trainer = pl.Trainer(max_epochs=1000, check_val_every_n_epoch=50)

    trainer.fit(model, graph_train_loader, graph_val_loader)



if __name__ == "__main__":
    train()
