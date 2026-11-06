#!/usr/bin/env python
"""Fresh inference with ST-HVG-Replogle fewshot/k562 on the frozen k562 split.

Mirrors `state tx predict` (state v0.11.2) but as an explicit, auditable runner:
- builds PerturbationDataModule from the *training* config kwargs + local toml
- loads checkpoint weights into StateTransitionPerturbationModel

split=test : per-cell preds/reals for the frozen k562 test split (unseen pairs)
             -> results npz (preds, reals, pert, celltype)
split=train: streams over ALL training cells (train_dataloader(test=True)) and
             accumulates per-(celltype, pert) running sums of preds and reals
             -> compact aggregates npz (used for L4 seen-pair scoring AND for
             the fixed model-independent baselines B1/B2)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import yaml
import lightning.pytorch as pl

sys.path.insert(0, "code/state/src")
from cell_load.data_modules import PerturbationDataModule
from state.tx.models.state_transition import StateTransitionPerturbationModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_k562_config.yaml")
    ap.add_argument("--toml", default="configs/local_k562.toml")
    ap.add_argument("--checkpoint", default="weights/sthvg_fewshot_k562_final.ckpt")
    ap.add_argument("--split", choices=["test", "train"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-batches", type=int, default=0, help=">0: debug cap")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_kwargs = dict(cfg["data"]["kwargs"])
    data_kwargs["toml_config_path"] = args.toml
    dm = PerturbationDataModule(batch_size=cfg["training"]["batch_size"], **data_kwargs)
    dm.setup(stage="test")

    pl.seed_everything(cfg["training"]["train_seed"])

    var_dims = dm.get_var_dims()
    mk = cfg["model"]["kwargs"]
    init = {
        "input_dim": var_dims["input_dim"], "hidden_dim": mk["hidden_dim"],
        "gene_dim": var_dims["gene_dim"], "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"], "pert_dim": var_dims["pert_dim"],
        **mk,
    }
    model = StateTransitionPerturbationModel.load_from_checkpoint(
        args.checkpoint, weights_only=False, map_location="cpu", **init)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    print("model on", dev, "| var_dims:", var_dims, flush=True)

    loader = dm.test_dataloader() if args.split == "test" else dm.train_dataloader(test=True)
    n_tot = loader.batch_sampler.tot_num
    print("cells to score:", n_tot, flush=True)
    od = var_dims["output_dim"]
    t0 = time.time()

    if args.split == "test":
        preds = np.empty((n_tot, od), np.float32)
        reals = np.empty((n_tot, od), np.float32)
        perts, ctypes = [], []
        i0 = 0
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if args.max_batches and bi >= args.max_batches:
                    break
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model.predict_step(batch, bi, padded=False)
                p = out["preds"].float().cpu().numpy()
                r = out["pert_cell_emb"].float().cpu().numpy()
                n = p.shape[0]
                preds[i0:i0+n] = p
                reals[i0:i0+n] = r
                pn, cn = out["pert_name"], out["celltype_name"]
                perts += [str(x) for x in (pn if isinstance(pn, (list, tuple)) else [pn]*n)]
                ctypes += [str(x) for x in (cn if isinstance(cn, (list, tuple)) else [cn]*n)]
                i0 += n
                if bi % 200 == 0:
                    print(f"batch {bi} cells {i0}/{n_tot} {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(args.out, preds=preds[:i0], reals=reals[:i0],
                            pert=np.array(perts), celltype=np.array(ctypes))
        meta = dict(checkpoint=args.checkpoint, split=args.split, n_cells=int(i0),
                    output_dim=int(od))
    else:
        # aggregated mode: per (celltype, pert) sums of preds and reals
        agg = {}  # (ct, pert) -> [count, pred_sum(od), real_sum(od)]
        i0 = 0
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if args.max_batches and bi >= args.max_batches:
                    break
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model.predict_step(batch, bi, padded=False)
                p = out["preds"].float().cpu().numpy().astype(np.float64)
                r = out["pert_cell_emb"].float().cpu().numpy().astype(np.float64)
                n = p.shape[0]
                pn, cn = out["pert_name"], out["celltype_name"]
                pns = [str(x) for x in (pn if isinstance(pn, (list, tuple)) else [pn]*n)]
                cns = [str(x) for x in (cn if isinstance(cn, (list, tuple)) else [cn]*n)]
                for i in range(n):
                    key = (cns[i], pns[i])
                    e = agg.get(key)
                    if e is None:
                        e = agg[key] = [0, np.zeros(od), np.zeros(od)]
                    e[0] += 1
                    e[1] += p[i]
                    e[2] += r[i]
                i0 += n
                if bi % 500 == 0:
                    print(f"batch {bi} cells {i0}/{n_tot} groups {len(agg)} {time.time()-t0:.0f}s", flush=True)
        keys = sorted(agg.keys())
        counts = np.array([agg[k][0] for k in keys], np.int64)
        pred_means = np.stack([agg[k][1] / agg[k][0] for k in keys]).astype(np.float32)
        real_means = np.stack([agg[k][2] / agg[k][0] for k in keys]).astype(np.float32)
        np.savez_compressed(args.out,
                            celltype=np.array([k[0] for k in keys]),
                            pert=np.array([k[1] for k in keys]),
                            counts=counts, pred_means=pred_means, real_means=real_means)
        meta = dict(checkpoint=args.checkpoint, split=args.split, n_cells=int(i0),
                    n_groups=len(keys), output_dim=int(od))
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("saved", args.out, meta, flush=True)

if __name__ == "__main__":
    main()
