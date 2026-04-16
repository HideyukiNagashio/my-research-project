import pickle
import numpy as np
from pathlib import Path

from src.preprocessing.config import ANGLE_SCALE
from src.preprocessing.pipeline import process_all_data_raw
from src.preprocessing.normalization import compute_global_stats, apply_global_normalization

# =====================================================================
# 1. Foldの手動定義 (6-fold Group Cross Validation)
# 12名の被験者を Test=2, Val=2, Train=8 で分割
# =====================================================================
FOLDS = [
    {
        "name": "fold1",
        "test": ['oba', 'ono'],
        "val": ['pon', 'kuno'],
        "train": ['john', 'konan', 'obara', 'fukuzawa', 'kiuchi', 'yanaze', 'adachi', 'iwasaki']
    },
    {
        "name": "fold2",
        "test": ['pon', 'kuno'],
        "val": ['john', 'konan'],
        "train": ['obara', 'fukuzawa', 'kiuchi', 'yanaze', 'adachi', 'iwasaki', 'oba', 'ono']
    },
    {
        "name": "fold3",
        "test": ['john', 'konan'],
        "val": ['obara', 'fukuzawa'],
        "train": ['kiuchi', 'yanaze', 'adachi', 'iwasaki', 'oba', 'ono', 'pon', 'kuno']
    },
    {
        "name": "fold4",
        "test": ['obara', 'fukuzawa'],
        "val": ['kiuchi', 'yanaze'],
        "train": ['adachi', 'iwasaki', 'oba', 'ono', 'pon', 'kuno', 'john', 'konan']
    },
    {
        "name": "fold5",
        "test": ['kiuchi', 'yanaze'],
        "val": ['adachi', 'iwasaki'],
        "train": ['oba', 'ono', 'pon', 'kuno', 'john', 'konan', 'obara', 'fukuzawa']
    },
    {
        "name": "fold6",
        "test": ['adachi', 'iwasaki'],
        "val": ['oba', 'ono'],
        "train": ['pon', 'kuno', 'john', 'konan', 'obara', 'fukuzawa', 'kiuchi', 'yanaze']
    }
]

# =====================================================================
# Helper 関数
# =====================================================================
def load_cached_raw_data(data_dir: Path):
    """
    Phase 1で出力されたローカルの raw_strides ファイル群を読み込む。
    毎回全処理を行うと時間がかかるため、構築済みのデータをロードして再利用する。
    """
    if not data_dir.exists():
        return []
        
    all_results = []
    files = list(data_dir.glob("*_raw.npz"))
    
    print(f"Loading cached raw data from {data_dir} ...")
    for path in files:
        data = np.load(path, allow_pickle=True)
        durations = data['durations'] if 'durations' in data else None
        
        all_results.append({
            'participant': str(data['participant']),
            'condition': str(data['condition']),
            'mass': float(data['mass']),
            'ensemble': data['ensemble'],
            'columns': data['columns'].tolist(),
            'durations': durations
        })
    return all_results

def save_split_data(output_path: Path, results_list, stats, p_map, c_map):
    """
    指定されたサブセット(train/val/test)の結果リストに正規化を適用し،
    結合して .pkl 形式で保存するヘルパー関数。
    """
    if not results_list:
        print(f"    [SKIP] Empty results list for {output_path.name}")
        return
        
    all_ens, all_ids, all_conds, all_durs = [], [], [], []
    columns = results_list[0]['columns']
    
    # tqdmで適用経過を表示したい場合は外して使えるが、ここは高速なので割愛
    for r in results_list:
        # 正規化を適用 (apply_global_normalization を利用)
        norm = apply_global_normalization(r['ensemble'], stats)
        n = norm.shape[0]
        
        all_ens.append(norm)
        all_ids.append(np.full(n, p_map[r['participant']]))
        all_conds.append(np.full(n, c_map[r['condition']]))
        
        if r['durations'] is not None:
            all_durs.append(r['durations'])
            
    # 全被験者のデータを1つに結合
    combined = np.concatenate(all_ens, axis=0)
    
    save_data = {
        'ensemble': combined,
        'subject_ids': np.concatenate(all_ids, axis=0),
        'condition_ids': np.concatenate(all_conds, axis=0),
        'columns': columns,
        'id_map': p_map,
        'condition_map': c_map,
        'angle_scale': ANGLE_SCALE
    }
    
    if all_durs:
        save_data['durations'] = np.concatenate(all_durs, axis=0)

    # Pickleとして保存
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)


def main():
    print("\n" + "="*55)
    print("  Start Manual Cross Validation Processing")
    print("="*55)
    
    raw_dir = Path('data/interim/raw_strides')
    
    # -------------------------------------------------------------
    # 2. 全データの取得
    #    すでに抽出＆同期処理済みの .npz ファイルがあればロード。
    #    なければ pipeline 側の関数を叩いて生成する。
    # -------------------------------------------------------------
    all_results = load_cached_raw_data(raw_dir)
    
    if not all_results:
        print("キャッシュされた raw データがないため、新規に処理を実行します...")
        try:
            # 既存の Phase 1 関数を利用
            all_results = process_all_data_raw(output_dir=str(raw_dir), n_workers=None)
        except Exception as e:
            print(f"Error processing raw data: {e}")
            return

    if not all_results:
        print("有効なデータが取得できませんでした。終了します。")
        return

    # IDマッピングの作成
    unique_p = sorted(set(r['participant'] for r in all_results))
    p_map    = {name: i for i, name in enumerate(unique_p)}
    c_map    = {'h': 0, 'm': 1, 'l': 2}
    
    base_out_dir = Path('data/processed/cv')
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # 3. Fold ごとの正規化と保存
    # -------------------------------------------------------------
    for dict_fold in FOLDS:
        fold_name = dict_fold["name"]
        
        # ログ表示の要件
        print(f"\n{fold_name.capitalize()}")
        print(f"Train subjects: {dict_fold['train']}")
        print(f"Val subjects:   {dict_fold['val']}")
        print(f"Test subjects:  {dict_fold['test']}")
        
        fold_dir = base_out_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # 被験者ごとにデータを振り分け抽出
        train_results = [r for r in all_results if r['participant'] in dict_fold['train']]
        val_results   = [r for r in all_results if r['participant'] in dict_fold['val']]
        test_results  = [r for r in all_results if r['participant'] in dict_fold['test']]
        
        if not train_results:
            print(f"  [ERROR] {fold_name} の train データがありません。スキップします。")
            continue
            
        try:
            # ① trainデータのみで統計量計算
            stats_path = fold_dir / 'normalization_stats.npz'
            stats = compute_global_stats(train_results, stats_path=str(stats_path))
            
            # ② 各 Split に同一の統計量を適用して保存
            print(f"  Applying normalization and saving split items...")
            
            save_split_data(fold_dir / 'train.pkl', train_results, stats, p_map, c_map)
            save_split_data(fold_dir / 'val.pkl',   val_results,   stats, p_map, c_map)
            save_split_data(fold_dir / 'test.pkl',  test_results,  stats, p_map, c_map)
            
            print(f"  Saved to {fold_dir} (train.pkl, val.pkl, test.pkl)")
            
        except Exception as e:
            print(f"  [ERROR] {fold_name} の処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*55)
    print("  完了 (CV Processing Complete)")
    print("="*55)


if __name__ == "__main__":
    main()
