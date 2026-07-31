import torch
import numpy as np

def create_graph_structure(use_shortcut=False, fully_connected=False):
    """
    センサーの座標からedge_indexとedge_weightを生成する関数
    fully_connected=True の場合は完全グラフ（自己ループ含む）を生成します。
    """
    # センサー座標 (x, y) ※0-indexedで管理
    coords = np.array([
        [27, 218], # 0: センサ1 (母指)
        [23, 183], # 1: センサ2 (第一中足骨頭)
        [46, 182], # 2: センサ3 (第三中足骨頭)
        [74, 176], # 3: センサ4 (第四中足骨頭)
        [74, 111], # 4: センサ5 (外側アーチ)
        [64,  52], # 5: センサ6 (外側踵)
        [38,  44], # 6: センサ7 (内側踵)
        [50,  19]  # 7: センサ8 (中央踵)
    ], dtype=np.float32)

    # 座標の正規化 (0〜1)
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    norm_coords = (coords - coords_min) / (coords_max - coords_min)
    norm_coords = torch.tensor(norm_coords, dtype=torch.float32)

    if fully_connected:
        bidirectional_edges = []
        for i in range(8):
            for j in range(8):
                bidirectional_edges.append((i, j))
    else:
        # 基本エッジ
        base_edges = [
            (0, 1), (1, 2), (2, 3), (5, 6), (6, 7), (3, 4), (4, 5), (1, 6)
        ]
        # ショートカットエッジ (アブレーション用)
        shortcut_edges = [
            (1, 3), (0, 6), (3, 5)
        ]
        edges = base_edges.copy()
        if use_shortcut:
            edges.extend(shortcut_edges)
            
        # 無向グラフにするため双方向にエッジを張る
        bidirectional_edges = []
        for u, v in edges:
            bidirectional_edges.append((u, v))
            bidirectional_edges.append((v, u))
            
    # edge_index の作成: 形状は (2, Num_Edges)
    edge_index = torch.tensor(bidirectional_edges, dtype=torch.long).t().contiguous()
    
    # エッジの重みをユークリッド距離の逆数として計算
    edge_weight = []
    for u, v in bidirectional_edges:
        if u == v:
            edge_weight.append(1.0) # 自己ループの重み
        else:
            dist = np.linalg.norm(coords[u] - coords[v])
            weight = 1.0 / (dist + 1e-6) # ゼロ割り算防止
            edge_weight.append(weight)
            
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    
    return norm_coords, edge_index, edge_weight
