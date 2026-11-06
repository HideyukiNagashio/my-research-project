"""
遊脚期+立脚期 Transformer 実装の検証スクリプト．

  Part A (データ不要)   : ダミーデータで forward を検証
      1. 各条件の入力形状   2. Encoder 出力形状   3. 立脚期抽出位置   4. 出力形状 (200x3)
      6. NaN/Inf            + Positional Encoding の整合・既存 transformer との等価性
  Part B (CV データ必要): 作成済みデータセットの検証
      5. 入力と教師データの時間的対応   7. Group CV 分割が既存と同一   + 既存データとの一致
      (--legacy_dir の既存 cv_grf と比較)

実行: PYTHONPATH=. python scripts/verify_swing_stance.py [--skip_data] [--folds 1]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from src.models import get_model
from src.models.transformer import PositionalEncoding, TimeSeriesTransformer
from src.models.transformer_stance import IndexedPositionalEncoding
from src.training.dataset_swing import (GaitSwingStanceDataset, get_window_spec, CONDITIONS, build_condition_ensemble)

PASS, FAIL = "[PASS]", "[FAIL]"
n_fail = 0


def check(cond, msg):
    global n_fail
    print(f"  {PASS if cond else FAIL} {msg}")
    if not cond:
        n_fail += 1


def build_model(spec, **kw):
    return get_model('transformer_stance', input_dim=14, output_dim=3, d_model=128, nhead=4, num_layers=3,
                     dim_feedforward=256, dropout_prob=0.1, seq_len=spec.seq_len, stance_start=spec.stance_start,
                     stance_len=spec.stance_len, pos_start=spec.pos_start, **kw)


# ----------------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------------
def part_a(n_swing):
    print("=" * 70 + f"\n Part A: dummy forward test (n_swing={n_swing})\n" + "=" * 70)
    torch.manual_seed(0)
    B = 4
    expected = {'HS-TO': (200, 0), 'TO-TO': (n_swing + 200, n_swing), 'HS-HS': (200 + n_swing, 0)}
    params = {}
    pe_rows = {}
    for cond in CONDITIONS:
        spec = get_window_spec(cond, n_swing=n_swing)
        T, s0 = expected[cond]
        print(f"\n[{cond}] {spec}")
        model = build_model(spec).eval()
        params[cond] = sum(p.numel() for p in model.parameters())
        x = torch.randn(B, T, 14)
        with torch.no_grad():
            enc = model.encode(x)
            out = model(x)
            manual = model.fc(enc[:, s0:s0 + 200])
        check(x.shape == (B, T, 14), f"1. input shape = {tuple(x.shape)} (expected {B}x{T}x14)")
        check(enc.shape == (B, T, 128), f"2. encoder output shape = {tuple(enc.shape)} (expected {B}x{T}x128)")
        check(spec.stance_start == s0 and spec.stance_end == s0 + 200,
              f"3. stance extraction index = [{spec.stance_start}:{spec.stance_end}] (expected [{s0}:{s0 + 200}])")
        check(torch.equal(out, manual), "3. forward output == fc(encoder_out[stance slice]) (exact)")
        check(out.shape == (B, 200, 3), f"4. output shape = {tuple(out.shape)} (expected {B}x200x3)")
        check(bool(torch.isfinite(enc).all() and torch.isfinite(out).all()), "6. no NaN/Inf in encoder/output")

        # 抽出位置の独立確認: 立脚期以外の入力を変えると，Self-Attention 経由で出力は変わるが，
        # 立脚期入力のみを変えた場合と比べて「どの出力 index が影響を受けるか」ではなく，
        # 抽出ずれを検出するため，Encoder 出力の one-hot 差分で slice の先頭/末尾を確認する
        enc_fake = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(1, T, 128).contiguous()
        sl = enc_fake[:, spec.stance_start:spec.stance_end, 0]
        check(int(sl[0, 0]) == s0 and int(sl[0, -1]) == s0 + 199,
              f"3. slice picks encoder positions {int(sl[0, 0])}..{int(sl[0, -1])} (expected {s0}..{s0 + 199})")

        pos = model.pos_encoder.positions.long()
        print(f"     PE positions: {int(pos[0])}..{int(pos[-1])}  (stance: {int(pos[spec.stance_start])}.."
              f"{int(pos[spec.stance_end - 1])})")
        check(int(pos[spec.stance_start]) == 0 and int(pos[spec.stance_end - 1]) == 199,
              "   stance positions are 0..199")
        pe_rows[cond] = model.pos_encoder.pe[0, spec.stance_start:spec.stance_end]

    print("\n[Cross-condition checks]")
    check(len(set(params.values())) == 1, f"parameter count identical across conditions: {params}")
    legacy = TimeSeriesTransformer(14, 128, 4, 3, 256, 3, 0.1)
    check(sum(p.numel() for p in legacy.parameters()) == params['HS-TO'],
          "parameter count identical to existing TimeSeriesTransformer")
    check(all(torch.equal(pe_rows[c], pe_rows['HS-TO']) for c in CONDITIONS),
          "stance-token positional encodings are identical across HS-TO / TO-TO / HS-HS")

    # PE が既存実装と一致 (0..199)
    old_pe = PositionalEncoding(128).pe[0, :200]
    new_pe = IndexedPositionalEncoding(128, torch.arange(200)).pe[0]
    check(torch.equal(old_pe, new_pe), "PE(0..199) is bit-identical to existing PositionalEncoding")
    neg = IndexedPositionalEncoding(128, torch.arange(-5, 5)).pe[0]
    check(torch.isfinite(neg).all() and not torch.equal(neg[:5], neg[5:]) and
          torch.allclose(neg[5], old_pe[0]), "negative positions are well-defined and distinguishable")

    # HS-TO は既存 transformer と数値的に等価（重みを共有）
    spec = get_window_spec('HS-TO')
    new = build_model(spec).eval()
    legacy.eval()
    sd = {k: v for k, v in legacy.state_dict().items() if not k.startswith('pos_encoder')}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    check(not unexpected and not missing, f"state_dict keys compatible with existing transformer "
                                          f"(missing={missing}, unexpected={unexpected})")
    x = torch.randn(3, 200, 14)
    with torch.no_grad():
        d = (new(x) - legacy(x)).abs().max().item()
    check(d < 1e-6, f"HS-TO output == existing TimeSeriesTransformer output (max abs diff {d:.2e})")

    # 立脚期以外（遊脚期）の入力が出力に影響する = Self-Attention で遊脚期情報が使われている
    spec = get_window_spec('TO-TO', n_swing=n_swing)
    m = build_model(spec).eval()
    x = torch.randn(2, spec.seq_len, 14)
    x2 = x.clone(); x2[:, :n_swing] += 1.0
    with torch.no_grad():
        d = (m(x) - m(x2)).abs().max().item()
    check(d > 1e-6, f"TO-TO: output depends on swing-phase input (max abs diff {d:.3e})")
    spec = get_window_spec('HS-HS', n_swing=n_swing)
    m = build_model(spec).eval()
    x = torch.randn(2, spec.seq_len, 14)
    x2 = x.clone(); x2[:, 200:] += 1.0
    with torch.no_grad():
        d = (m(x) - m(x2)).abs().max().item()
    check(d > 1e-6, f"HS-HS: output depends on swing-phase input (max abs diff {d:.3e})")

    # 1 step の学習 (backward) が通る
    spec = get_window_spec('TO-TO', n_swing=n_swing)
    m = build_model(spec).train()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    loss = torch.nn.functional.mse_loss(m(torch.randn(8, spec.seq_len, 14)), torch.randn(8, 200, 3))
    loss.backward(); opt.step()
    check(bool(torch.isfinite(loss)), f"TO-TO: one training step OK (loss {loss.item():.4f})")


# ----------------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------------
def reconstruct_to_200(swing, stance, a, b, c, kind):
    """
    正規化済み (遊脚N点 / 立脚200点) を実サンプル位置に戻し，既存の 1.0_pre/post_swing と同じ
    「区間全体を 200 点に線形正規化」した系列へ変換する（整合性の独立検証用）．
    swing, stance: (n, 14).  kind: 'TO-TO' (a=swing len, b=stance len) / 'HS-HS' (b=stance len, c=swing len)
    """
    nsw = swing.shape[0]
    if kind == 'TO-TO':
        L = a + b
        t_sw = np.linspace(0, a - 1, nsw)
        t_st = a + np.linspace(0, b - 1, stance.shape[0])
        seg = [(np.arange(a), t_sw, swing), (a + np.arange(b), t_st, stance)]
    else:
        L = b + c
        t_st = np.linspace(0, b - 1, stance.shape[0])
        t_sw = b + np.linspace(0, c - 1, nsw)
        seg = [(np.arange(b), t_st, stance), (b + np.arange(c), t_sw, swing)]
    rec = np.zeros((L, stance.shape[1]))
    for idx, tsrc, ysrc in seg:
        for ch in range(stance.shape[1]):
            rec[idx, ch] = np.interp(idx, tsrc, ysrc[:, ch])
    t200 = np.linspace(0, L - 1, 200)
    return np.stack([np.interp(t200, np.arange(L), rec[:, ch]) for ch in range(rec.shape[1])], axis=1)


def part_b(args):
    from scripts.run_manual_cv import GRF_FOLDS
    print("\n" + "=" * 70 + f"\n Part B: dataset check ({args.data_dir} vs {args.legacy_dir})\n" + "=" * 70)
    data_dir, legacy_dir = Path(args.data_dir), Path(args.legacy_dir)
    for fold in args.folds:
        fd = GRF_FOLDS[fold - 1]
        fname = f'fold{fold}'
        print(f"\n[{fname}] test={fd['test']} val={fd['val']}")
        ids = {}
        for split in ['train', 'val', 'test']:
            with open(data_dir / fname / f'{split}.pkl', 'rb') as f:
                new = pickle.load(f)
            inv = {v: k for k, v in new['id_map'].items()}
            subs = sorted(set(inv[i] for i in np.unique(new['subject_ids'])))
            ids[split] = set(subs)
            # 7. Group CV 分割
            check(subs == sorted(fd[split]), f"7. {split} subjects == GRF_FOLDS ({subs})")

            # ストライドと標準化の一致: 既存 pkl と比較（test, val のみ．train は巨大なため --check_train 指定時）
            if split == 'train' and not args.check_train:
                continue
            with open(legacy_dir / fname / f'{split}.pkl', 'rb') as f:
                old = pickle.load(f)
            check(np.array_equal(new['subject_ids'], old['subject_ids']) and
                  np.array_equal(new['condition_ids'], old['condition_ids']),
                  f"   {split}: stride order / subject / condition ids identical to existing data (N={len(old['subject_ids'])})")
            check(np.array_equal(new['ensemble_X_0.5'], old['ensemble_X_0.5'].astype(np.float32)) and
                  np.array_equal(new['ensemble_Y_0.5'], old['ensemble_Y_0.5'].astype(np.float32)),
                  f"   {split}: stance X_0.5 / Y_0.5 identical to existing data (float32)")

            n_swing = new['ensemble_S_pre_swing'].shape[1]
            for cond, key in [('TO-TO', 'ensemble_X_1.0_pre_swing'), ('HS-HS', 'ensemble_X_1.0_post_swing')]:
                X = build_condition_ensemble(new, cond)[:, :, :14].astype(np.float64)
                legacy = old[key][:, :, :14]
                sl = new['seg_lengths']
                spec = get_window_spec(cond, n_swing=n_swing)
                errs = []
                sel = np.arange(0, len(X), max(1, len(X) // 200))
                for n in sel:
                    a, b, c = sl[n]
                    sw = X[n, :n_swing] if cond == 'TO-TO' else X[n, 200:]
                    st = X[n, spec.stance_start:spec.stance_end]
                    errs.append(np.abs(reconstruct_to_200(sw, st, a, b, c, cond) - legacy[n]).mean())
                scale = np.abs(legacy[sel]).mean()
                check(np.mean(errs) < 0.02 * max(scale, 1e-6),
                      f"5. {split} {cond}: re-expressed on the legacy 200-pt axis, mean|diff|={np.mean(errs):.4f} "
                      f"(mean|signal|={scale:.3f}, {len(sel)} strides) -> swing/stance temporal alignment")

            # 立脚期の GRF（教師）が入力の立脚期 pressure と同一ストライドであること: HS 直後に Fz が立ち上がる
            fz = new['ensemble_Y_0.5'][:, :, 25]
            check(np.abs(fz[:, 0]).mean() < 0.15 and np.abs(fz[:, -1]).mean() < 0.15,
                  f"   {split}: target Fz is ~0.05BW at both ends of the stance window "
                  f"(start {fz[:, 0].mean():.3f}, end {fz[:, -1].mean():.3f} BW)")
            del old

        check(not (ids['train'] & ids['val']) and not (ids['train'] & ids['test']) and not (ids['val'] & ids['test']),
              "7. no subject leak between splits")

        # Dataset 経由の形状・対応・NaN/Inf
        ref = None
        for cond in CONDITIONS:
            ds = GaitSwingStanceDataset(str(data_dir / fname / 'test.pkl'), cond, n_swing=args.n_swing)
            T = ds.spec.seq_len
            check(tuple(ds.X.shape[1:]) == (T, 14) and tuple(ds.y.shape[1:]) == (200, 3),
                  f"1/4. {cond}: X {tuple(ds.X.shape)}, y {tuple(ds.y.shape)}")
            check(bool(torch.isfinite(ds.X).all() and torch.isfinite(ds.y).all()), f"6. {cond}: no NaN/Inf in X/y")
            stance = ds.X[:, ds.spec.stance_start:ds.spec.stance_end]
            if ref is None:
                ref = (stance, ds.y)
            check(torch.equal(stance, ref[0]) and torch.equal(ds.y, ref[1]),
                  f"5. {cond}: stance window of X and y identical to HS-TO (same stride, same time axis)")
    # 正規化統計が既存と一致
    for fold in args.folds:
        a = np.load(data_dir / f'fold{fold}' / 'normalization_stats.npz', allow_pickle=True)
        b = np.load(legacy_dir / f'fold{fold}' / 'normalization_stats.npz', allow_pickle=True)
        check(float(a['pressure_global_max']) == float(b['pressure_global_max']) and
              np.array_equal(a['imu_std'], b['imu_std']), f"fold{fold}: normalization stats identical to existing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_swing', type=int, default=140)
    ap.add_argument('--data_dir', type=str, default=None)
    ap.add_argument('--legacy_dir', type=str, default='data/processed/cv_grf')
    ap.add_argument('--folds', type=int, nargs='+', default=[1])
    ap.add_argument('--skip_data', action='store_true')
    ap.add_argument('--check_train', action='store_true')
    args = ap.parse_args()
    args.data_dir = args.data_dir or f'data/processed/cv_grf_swing{args.n_swing}'

    part_a(args.n_swing)
    if not args.skip_data:
        part_b(args)
    print("\n" + ("ALL CHECKS PASSED" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED"))
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()

