import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, LGConv, SAGEConv


class GNNEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        gat_heads: int = 2,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.dropout = dropout

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.convs = nn.ModuleList()
        dims = [in_channels] + [hidden_channels] * (num_layers - 1)

        for i in range(num_layers - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            self.convs.append(self._make_conv(in_dim, out_dim, gat_heads, is_last=False))

        self.final_conv = self._make_conv(hidden_channels, out_channels, gat_heads, is_last=True)

    def _make_conv(self, in_dim: int, out_dim: int, gat_heads: int, is_last: bool):
        if self.model_name == "gcn":
            return GCNConv(in_dim, out_dim)
        if self.model_name == "graphsage":
            return SAGEConv(in_dim, out_dim)
        if self.model_name == "gat":
            heads = 1 if is_last else gat_heads
            concat = not is_last
            out_channels = out_dim if not concat else max(1, out_dim // heads)
            return GATConv(in_dim, out_channels, heads=heads, concat=concat)
        raise ValueError(f"Unsupported model_name: {self.model_name}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_conv(x, edge_index)
        return x


class LightGCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1 for LightGCN")

        # LightGCN keeps message passing linear; a projection aligns feature dims.
        self.input_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            embeddings.append(x)

        return torch.stack(embeddings, dim=0).mean(dim=0)


class LinkPredictor(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    @staticmethod
    def decode(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
