from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
from typing import Dict, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import subgraph


@dataclass
class DatasetBundle:
    name: str
    data: Data
    train_data: Data
    val_data: Data
    test_data: Data


def _pad_features(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    if x.size(-1) == target_dim:
        return x
    pad = torch.zeros((x.size(0), target_dim - x.size(-1)), dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _hetero_movielens_to_homo(hetero_data) -> Data:
    if "user" not in hetero_data.node_types or "movie" not in hetero_data.node_types:
        raise ValueError("MovieLens hetero dataset must contain user and movie node types")

    user_x = hetero_data["user"].x
    movie_x = hetero_data["movie"].x

    if user_x is None:
        user_x = torch.ones((hetero_data["user"].num_nodes, 1), dtype=torch.float)
    if movie_x is None:
        movie_x = torch.ones((hetero_data["movie"].num_nodes, 1), dtype=torch.float)

    dim = max(user_x.size(-1), movie_x.size(-1))
    user_x = _pad_features(user_x.float(), dim)
    movie_x = _pad_features(movie_x.float(), dim)

    x = torch.cat([user_x, movie_x], dim=0)

    if ("user", "rates", "movie") in hetero_data.edge_types:
        edge_index = hetero_data[("user", "rates", "movie")].edge_index
    else:
        edge_type = hetero_data.edge_types[0]
        edge_index = hetero_data[edge_type].edge_index

    user_count = hetero_data["user"].num_nodes
    src = edge_index[0]
    dst = edge_index[1] + user_count

    bipartite = torch.stack([src, dst], dim=0)
    rev = torch.stack([dst, src], dim=0)
    edge_index = torch.cat([bipartite, rev], dim=1)

    return Data(x=x, edge_index=edge_index)


def load_dataset(name: str, root: str = "./data") -> Data:
    name = name.lower()

    if name == "amazon_computers":
        from torch_geometric.datasets import Amazon

        return Amazon(root=f"{root}/Amazon", name="Computers")[0]

    if name == "yelp":
        from torch_geometric.datasets import Yelp

        yelp_root = f"{root}/Yelp"
        try:
            data = Yelp(root=yelp_root)[0]
        except Exception as exc:
            # Recovery path for partially downloaded/corrupted Yelp cache files.
            msg = str(exc).lower()
            if "failed to read all data" in msg or "not fully written" in msg:
                raw_dir = os.path.join(yelp_root, "raw")
                proc_dir = os.path.join(yelp_root, "processed")
                for fname in ("feats.npy", "adj_full.npz", "class_map.json", "role.json"):
                    fpath = os.path.join(raw_dir, fname)
                    if os.path.exists(fpath):
                        os.remove(fpath)
                if os.path.isdir(proc_dir):
                    shutil.rmtree(proc_dir)
                data = Yelp(root=yelp_root)[0]
            else:
                raise

        return Data(x=data.x.float(), edge_index=data.edge_index)

    if name == "movielens_1m":
        data = None
        errors = []

        try:
            from torch_geometric.datasets import MovieLens1M  # type: ignore

            maybe = MovieLens1M(root=f"{root}/MovieLens1M")
            data = maybe[0]
        except Exception as exc:
            errors.append(f"MovieLens1M failed: {exc}")

        if data is None:
            try:
                from torch_geometric.datasets import MovieLens  # type: ignore

                maybe = MovieLens(root=f"{root}/MovieLens", name="1m")
                data = maybe[0]
            except Exception as exc:
                errors.append(f"MovieLens(name='1m') failed: {exc}")

        if data is None:
            detail = " | ".join(errors)
            raise RuntimeError(
                "Could not load MovieLens-1M from PyG in this environment. "
                "Please check your torch_geometric version. Details: "
                f"{detail}"
            )

        if hasattr(data, "node_types"):
            return _hetero_movielens_to_homo(data)

        return Data(x=data.x.float(), edge_index=data.edge_index)

    raise ValueError(f"Unsupported dataset name: {name}")


def split_for_link_prediction(
    data: Data,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    is_undirected: bool = True,
) -> Tuple[Data, Data, Data]:
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=is_undirected,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    return transform(data)


def downsample_nodes(data: Data, max_nodes: int, seed: int) -> Data:
    if data.num_nodes <= max_nodes:
        return data

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(data.num_nodes, generator=g)[:max_nodes]
    keep_nodes = perm.sort().values

    edge_index, _ = subgraph(keep_nodes, data.edge_index, relabel_nodes=True)
    x = data.x[keep_nodes] if data.x is not None else None

    return Data(x=x, edge_index=edge_index)


def load_all_bundles(
    dataset_names: list[str],
    root: str = "./data",
    quick: bool = False,
    seed: int = 42,
) -> Dict[str, DatasetBundle]:
    bundles: Dict[str, DatasetBundle] = {}

    quick_caps = {
        "movielens_1m": 12000,
        "yelp": 50000,
        "amazon_computers": 14000,
    }

    for name in dataset_names:
        data = load_dataset(name, root=root)
        if quick:
            cap = quick_caps.get(name, data.num_nodes)
            data = downsample_nodes(data, max_nodes=cap, seed=seed)
        train_data, val_data, test_data = split_for_link_prediction(data)
        bundles[name] = DatasetBundle(
            name=name,
            data=data,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )
    return bundles
