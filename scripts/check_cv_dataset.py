import pickle
import numpy as np
from pathlib import Path

# 被験者リスト
EXPECTED_PARTICIPANTS = {
    'oba', 'ono', 'pon', 'kuno', 'john', 'konan',
    'obara', 'fukuzawa', 'kiuchi', 'yanaze', 'adachi', 'iwasaki'
}

def load_pkl(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def check_overlap(train_ids, val_ids, test_ids, id_map):
    # IDから被験者名に逆変換
    inv_map = {v: k for k, v in id_map.items()}
    train_subs = set(inv_map[i] for i in np.unique(train_ids))
    val_subs   = set(inv_map[i] for i in np.unique(val_ids))
    test_subs  = set(inv_map[i] for i in np.unique(test_ids))

    intersect_tv = train_subs.intersection(val_subs)
    intersect_tt = train_subs.intersection(test_subs)
    intersect_vt = val_subs.intersection(test_subs)

    has_leak = False
    if intersect_tv:
        print(f"    [ERROR] train ∩ val = {intersect_tv}")
        has_leak = True
    if intersect_tt:
        print(f"    [ERROR] train ∩ test = {intersect_tt}")
        has_leak = True
    if intersect_vt:
        print(f"    [ERROR] val ∩ test = {intersect_vt}")
        has_leak = True

    if not has_leak:
        print("    [OK] No subject leak between splits.")
        print(f"    Train: {sorted(list(train_subs))}")
        print(f"    Val  : {sorted(list(val_subs))}")
        print(f"    Test : {sorted(list(test_subs))}")

    return train_subs, val_subs, test_subs

def check_nan_inf(tensor, name):
    has_nan = np.isnan(tensor).any()
    has_inf = np.isinf(tensor).any()
    if has_nan or has_inf:
        print(f"    [WARNING] {name} has NaN: {has_nan}, Inf: {has_inf}")
    else:
        print(f"    [OK] {name} has no NaN/Inf.")

def print_stats(y, name):
    mean_y = np.mean(y)
    std_y  = np.std(y)
    min_y  = np.min(y)
    max_y  = np.max(y)
    print(f"    {name} Stats: Mean={mean_y:.4f}, Std={std_y:.4f}, Min={min_y:.4f}, Max={max_y:.4f}")

def extract_xy(ensemble):
    """
    片脚/両脚モデル用にXとyをスライスする想定
    ここでは両脚モデル (X_bilateral) の 28ch とターゲット 12ch (角度9+GRF3) を想定して取り出します
    ※ 実際の深層学習モデルに合わせてDataset側でスライスしてください
    """
    # X_bi: ipsi 圧力8+IMU6, contra 圧力8+IMU6
    X = np.concatenate([ensemble[:, :, :14], ensemble[:, :, 26:40]], axis=-1)
    # y: ipsi 角度9+GRF3
    y = ensemble[:, :, 14:26]
    return X, y

def main():
    base_dir = Path("data/processed/cv")
    if not base_dir.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        return

    fold_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("fold")])
    if len(fold_dirs) != 6:
        print(f"[WARNING] Expected 6 folds, found {len(fold_dirs)} folds.")

    # Fold間整合性チェック用の変数
    global_test_subs = []
    
    print("="*60)
    print(" CV Dataset Verification Check")
    print("="*60)

    for i, fold_dir in enumerate(fold_dirs, 1):
        print(f"\n[{fold_dir.name.upper()}]")
        
        splits = {'train': None, 'val': None, 'test': None}
        missing = []
        for s in splits.keys():
            pkl_path = fold_dir / f"{s}.pkl"
            if pkl_path.exists():
                splits[s] = load_pkl(pkl_path)
            else:
                missing.append(s)

        if missing:
            print(f"  [ERROR] Missing files: {missing}")
            continue
            
        print("  1. Dictionary Keys:")
        print(f"    {list(splits['train'].keys())}")

        id_map = splits['train']['id_map']
        inv_map = {v: k for k, v in id_map.items()}

        # -----------------------------------------------------
        # サンプル数、Shape、NaN/Inf確認
        # -----------------------------------------------------
        print("\n  2 & 4 & 6. Tensor Shape, X/y structure, and NaN/Inf Check:")
        
        split_subject_ids = {}

        for s_name, data_dict in splits.items():
            ensemble = data_dict['ensemble']
            subj_ids = data_dict['subject_ids']
            split_subject_ids[s_name] = subj_ids
            
            X, y = extract_xy(ensemble)
            
            print(f"    {s_name.upper()} Original ensemble: {ensemble.shape} (N, T, F) -> X: {X.shape}, y: {y.shape}")
            
            check_nan_inf(X, f"X_{s_name}")
            check_nan_inf(y, f"y_{s_name}")
            
        # -----------------------------------------------------
        # 被験者ごとのストライド内訳
        # -----------------------------------------------------
        print("\n  4. Stride Counts per Participant:")
        for s_name in ['train', 'val', 'test']:
            subs_array = split_subject_ids[s_name]
            unique_ids, counts = np.unique(subs_array, return_counts=True)
            dist_str = ", ".join([f"{inv_map[uid]}: {c}" for uid, c in zip(unique_ids, counts)])
            print(f"    {s_name.upper()} (Total: {len(subs_array)}): {dist_str}")

        # -----------------------------------------------------
        # 被験者リークの確認
        # -----------------------------------------------------
        print("\n  3. Subject Leak Check:")
        t_subs, v_subs, te_subs = check_overlap(
            split_subject_ids['train'],
            split_subject_ids['val'],
            split_subject_ids['test'],
            id_map
        )
        global_test_subs.extend(list(te_subs))

        # -----------------------------------------------------
        # ラベル分布確認
        # -----------------------------------------------------
        print("\n  5. Target Label (y) Distribution Distribution Check:")
        for s_name, data_dict in splits.items():
            # y は 回帰なので Mean, Std, Min, Max
            _, y = extract_xy(data_dict['ensemble'])
            print_stats(y, f"y_{s_name.upper()}")

        print("-" * 60)

    # -----------------------------------------------------
    # Fold間整合性チェック
    # -----------------------------------------------------
    print("\n[GLOBAL CONSISTENCY CHECK]")
    
    # 1. 各被験者が test に1回ずつ入っているか
    test_counter = {sub: global_test_subs.count(sub) for sub in EXPECTED_PARTICIPANTS}
    
    bad_counts = {sub: count for sub, count in test_counter.items() if count != 1}
    if not bad_counts:
        print("  [OK] 全被験者がテストセットにちょうど1回ずつ割り当てられています。")
    else:
        print(f"  [ERROR] テスト回数が1回ではない被験者がいます: {bad_counts}")
        
    # 2. 全被験者が使われているか
    used_subs = set(global_test_subs)
    unused = EXPECTED_PARTICIPANTS - used_subs
    if not unused:
        print("  [OK] 定義された12名の被験者全員がシステムで利用されています。")
    else:
        print(f"  [ERROR] 未使用の被験者がいます: {unused}")

if __name__ == "__main__":
    main()
