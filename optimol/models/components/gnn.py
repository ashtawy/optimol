import logging

import torch
import torch.nn as nn
from optimol.models.components.conv_layers import (
    MLP,
    ESAGEConv,
    GaussianSmearing,
    RadiusInteractionGraph,
    get_activation_module,
)
from torch.nn import BatchNorm1d
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


class ConvLayers(nn.Module):
    def __init__(
        self,
        n_atom_types,
        n_atom_embeddings,
        n_edge_features,
        atom_embedding_sizes,
        activation="relu",
        atom_update_aggregation="add",
        batch_norm=False,
        dropout=0.0,
        readout_pooling="add",
        readout_mlp_layer_sizes=[],
        apply_output_activation=True,
        skip_connections=False,
    ):
        super(ConvLayers, self).__init__()
        self.batch_norm = batch_norm
        self.layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])
        self.activation = get_activation_module(activation)
        layer_in_dim = n_atom_embeddings
        self.skip_connections = skip_connections
        self.dropout = dropout

        self.interaction_graph = RadiusInteractionGraph(7.0, 32)
        self.rbf = GaussianSmearing(0.0, 7.0, n_edge_features)
        # self.atom_encoder = nn.Embedding(n_atom_types, n_atom_embeddings)
        for layer_out_dim in atom_embedding_sizes:
            self.layers.append(
                ESAGEConv(
                    in_channels=layer_in_dim,
                    out_channels=layer_out_dim,
                    edge_dim=n_edge_features,
                    aggr=atom_update_aggregation,
                )
            )
            layer_in_dim = layer_out_dim

            if self.batch_norm:
                self.bn_layers.append(BatchNorm1d(layer_out_dim))
                # self.bn_layers.append(LayerNorm(layer_out_dim))
            layer_in_dim = layer_out_dim

        pooling_mapper = {
            "add": global_add_pool,
            "max": global_max_pool,
            "mean": global_mean_pool,
        }
        readout_pooling = (
            [readout_pooling] if isinstance(readout_pooling, str) else readout_pooling
        )
        self.readout_pooling = []
        for rp in readout_pooling:
            if rp in pooling_mapper:
                self.readout_pooling.append(pooling_mapper[rp])
            else:
                raise ValueError(
                    f"Unsupported Graph Readout Pooling type: {rp} in {readout_pooling}. Use: {list(pooling_mapper.keys())}"
                )
        if len(readout_mlp_layer_sizes) > 0:
            mixing_hidden_layer_sizes = readout_mlp_layer_sizes[:-1]
            n_output_features = readout_mlp_layer_sizes[-1]
            self.mlp = MLP(
                n_input_features=layer_out_dim * len(readout_pooling),
                hidden_layer_sizes=mixing_hidden_layer_sizes,
                n_output_features=n_output_features,
                activation=activation,
                apply_output_activation=apply_output_activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )
            self.n_outputs = n_output_features
        else:
            self.mlp = None
            self.n_outputs = layer_out_dim * len(readout_pooling)

    def forward(self, batch):
        # x = self.atom_encoder(batch.x)
        x = batch.x
        if hasattr(batch, "pos") and batch.pos is not None:
            edge_index, edge_attr = self.interaction_graph(batch.pos, batch.batch)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_attr = self.rbf(edge_attr)
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        else:
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr.to(torch.float32)
        for layer_index, conv_layer in enumerate(self.layers):
            x_in = x
            x = conv_layer(x, edge_index, edge_attr)
            x = self.activation(x)

            if self.skip_connections and x_in.shape[-1] == x.shape[-1]:
                x = x_in + x
            if self.batch_norm:
                x = self.bn_layers[layer_index](x)
            if self.dropout > 0:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = torch.cat([rp(x, batch.batch) for rp in self.readout_pooling], axis=1)
        if self.mlp is not None:
            x = self.mlp(x).squeeze(-1)
        return x


class GNN(nn.Module):
    def __init__(
        self,
        n_global_features,
        n_atom_types,
        n_atom_embeddings,
        n_edge_features,
        gnn_embedding,
        global_embedding,
        output_net_configs,
        n_tasks,
    ):
        super(GNN, self).__init__()
        self.mol_emb_size = 0
        self.encoders = nn.ModuleList([])
        self.encoder_types = []

        if n_atom_types > 0:
            conv_layer = ConvLayers(
                n_atom_types=n_atom_types,
                n_atom_embeddings=n_atom_embeddings,
                n_edge_features=n_edge_features,
                atom_embedding_sizes=gnn_embedding["atom_embedding_sizes"],
                activation=gnn_embedding["activation"],
                atom_update_aggregation=gnn_embedding["atom_update_aggregation"],
                batch_norm=gnn_embedding["batch_norm"],
                dropout=gnn_embedding["dropout"],
                readout_pooling=gnn_embedding["readout_pooling"],
                readout_mlp_layer_sizes=gnn_embedding["readout_mlp_layer_sizes"],
                skip_connections=gnn_embedding.get("skip_connections", False),
                apply_output_activation=True,
            )
            self.encoders.append(conv_layer)
            self.encoder_types.append("mol_graph_features")
            self.mol_emb_size += conv_layer.n_outputs

        if n_global_features > 0:
            if global_embedding is None:
                mlp = None
                self.mol_emb_size += n_global_features
            else:
                mlp = MLP(
                    n_input_features=n_global_features,
                    hidden_layer_sizes=global_embedding["hidden_layer_sizes"],
                    n_output_features=global_embedding["n_output_features"],
                    activation=global_embedding["activation"],
                    apply_output_activation=global_embedding["apply_output_activation"],
                    batch_norm=global_embedding["batch_norm"],
                    dropout=global_embedding["dropout"],
                )
                self.mol_emb_size += mlp.n_outputs
            self.encoders.append(mlp)
            self.encoder_types.append("mol_global_features")

        self.output_net = MLP(
            n_input_features=self.mol_emb_size,
            hidden_layer_sizes=output_net_configs["hidden_layer_sizes"],
            n_output_features=n_tasks,
            activation=output_net_configs["activation"],
            apply_output_activation=output_net_configs["apply_output_activation"],
            batch_norm=output_net_configs["batch_norm"],
            dropout=output_net_configs["dropout"],
        )

    def forward(self, batch):
        input_embeddings = []
        for i, encoder in enumerate(self.encoders):
            if self.encoder_types[i] == "mol_graph_features":
                input_embeddings.append(encoder(batch))
            elif self.encoder_types[i] == "mol_global_features" and encoder is None:
                x_i = (
                    batch if isinstance(batch, torch.Tensor) else batch.global_features
                )
                input_embeddings.append(x_i)
            elif self.encoder_types[i] == "mol_global_features" and encoder is not None:
                x_i = (
                    batch if isinstance(batch, torch.Tensor) else batch.global_features
                )
                input_embeddings.append(encoder(x_i))
        x = torch.cat(input_embeddings, axis=1)
        output = self.output_net(x) if self.output_net else x
        output = output.reshape([-1, 1]) if output.ndim == 1 else output
        return output


if __name__ == "__main__":
    _ = GNN()
