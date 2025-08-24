#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bake static overlay tiles (no captions inside image) and write a manifest.json.
Output tree avoids any folder named "15" (layer index) by using "objs15/" explicitly:

  static_tiles/<dataset>/objs15/<triplet>/
      base.webp
      shape/LXX_pair_YY.webp
      color/LXX_pair_YY.webp
      both/LXX_pair_YY.webp
  static_tiles/manifest.json

Run (example):
  python bake_static_overlays.py \
    --in_root "C:/.../visualize_scene_description" \
    --out_root "C:/.../visualize_scene_description/static_tiles" \
    --datasets row just_number just_scan simple \
    --layers 0 27 \
    --modes shape color both
"""

import os, re, json, math, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageFilter

# -------- parsing & matching --------
def normalize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "", (s or "").lower())

def parse_pairs_from_generated_text(raw: str) -> List[Tuple[str, str]]:
    if not raw:
        return []
    try:
        i0 = raw.index("["); i1 = raw.rindex("]") + 1
        arr = json.loads(raw[i0:i1])
        out = []
        for obj in arr:
            if isinstance(obj, dict) and "shape" in obj and "color" in obj:
                out.append((str(obj["shape"]), str(obj["color"])))
        if out:
            return out
    except Exception:
        pass
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r"\{[^}]*\}", raw, flags=re.DOTALL):
        s = m.group(0)
        m1 = re.search(r'"shape"\s*:\s*"([^"]+)"', s)
        m2 = re.search(r'"color"\s*:\s*"([^"]+)"', s)
        if m1 and m2:
            out.append((m1.group(1), m2.group(1)))
    return out

def ordered_tokens_and_layers(per_token: Dict[str, Dict[str, Dict[str, float]]]
                              ) -> Tuple[List[str], List[Dict[str, Dict[str, float]]]]:
    items = sorted(per_token.items(), key=lambda kv: kv[0])  # "0001:<tok>"
    toks, pls = [], []
    for k, patch_layers in items:
        toks.append(k.split(":", 1)[1] if ":" in k else k)
        pls.append(patch_layers)
    return toks, pls

def match_value_to_tokens(value: str, tok_norm: List[str], start_pos: int, max_span: int = 3
                          ) -> Tuple[List[int], int]:
    target = normalize_token(value)
    N = len(tok_norm)
    for i in range(start_pos, N):
        for L in range(min(max_span, N - i), 0, -1):
            chunk = tok_norm[i:i+L]
            if any(c == "" for c in chunk): continue
            if target == "-".join(chunk) or target == "".join(chunk):
                return list(range(i, i+L)), i+L
    return [], start_pos

# -------- geometry & vectors --------
def infer_grid(n_patches: int, img_w: int, img_h: int) -> Tuple[int, int]:
    target_ratio = img_w / max(1e-6, img_h)
    best = None
    for gh in range(1, n_patches + 1):
        if n_patches % gh != 0: continue
        gw = n_patches // gh
        err = abs((gw / gh) - target_ratio)
        if best is None or err < best[0]:
            best = (err, gh, gw)
    if best is None:
        gw = int(round(math.sqrt(n_patches)))
        gh = max(1, n_patches // max(1, gw))
        return gh, max(1, n_patches // gh)
    _, gh, gw = best
    return gh, gw

def vector_for_token_layer(patch_layers: Dict[str, Dict[str, float]],
                           layer_key: str, n_patches: int) -> np.ndarray:
    v = np.full(n_patches, np.nan, dtype=float)
    for p in range(n_patches):
        m = patch_layers.get(f"patch{p}")
        if not m: continue
        val = m.get(layer_key)
        if val is None: continue
        v[p] = float(val)
    return v

def nanmean_vectors(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors: return None
    stack = np.vstack(vectors)
    with np.errstate(invalid="ignore"):
        v = np.nanmean(stack, axis=0)
    if np.all(np.isnan(v)): return None
    return v

# -------- color mapping --------
def hsl_to_rgb(h: float, s: float, l: float):
    h/=360.0; s/=100.0; l/=100.0
    a = s * min(l, 1 - l)
    def f(n):
        k = (n + h * 12) % 12
        c = l - a * max(-1, min(k - 3, 9 - k, 1))
        return int(round(255 * c))
    return f(0), f(8), f(4)

def heatmap_rgba(grid: np.ndarray, alpha: float, vmax: Optional[float]) -> Image.Image:
    gh, gw = grid.shape
    finite = grid[np.isfinite(grid)]
    if vmax is None:
        vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    vmax = max(1e-6, float(vmax))
    rgba = Image.new("RGBA", (gw, gh))
    px = rgba.load()
    a255 = int(round(alpha * 255))
    for j in range(gh):
        for i in range(gw):
            v = grid[j, i]
            x = 0.0 if not math.isfinite(v) else max(0.0, min(1.0, v / vmax))
            hue = (1.0 - x) * 240.0
            r, g, b = hsl_to_rgb(hue, 100.0, 50.0)
            px[i, j] = (r, g, b, a255)
    return rgba

# -------- bake one (NO caption drawn) --------
def bake_one(base_img_path: Path, json_path: Path, out_dir: Path,
             layer_idx: int, mode: str, alpha: float=0.45, blur_px: float=2.0
             ) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    gen_text  = data.get("generated_text", "")
    per_token = data.get("per_token", {})
    if not per_token: return []

    pairs = parse_pairs_from_generated_text(gen_text)
    if not pairs: return []

    tokens, tpl = ordered_tokens_and_layers(per_token)
    tok_norm = [normalize_token(t) for t in tokens]

    # patches
    all_patch_ids = set()
    for m in tpl[0].keys():
        if m.startswith("patch"):
            try: all_patch_ids.add(int(m.replace("patch", "")))
            except: pass
    P = (max(all_patch_ids) + 1) if all_patch_ids else 0
    if P <= 0: return []

    base = Image.open(base_img_path).convert("RGB")
    W, H = base.size
    GH, GW = infer_grid(P, W, H)
    layer_key = f"layer{layer_idx + 1}"

    cache: Dict[int, np.ndarray] = {}
    def get_vec(ti: int) -> np.ndarray:
        if ti not in cache:
            cache[ti] = vector_for_token_layer(tpl[ti], layer_key, P)
        return cache[ti]

    out_dir.mkdir(parents=True, exist_ok=True)
    metas: List[dict] = []
    pos = 0
    idx = 0
    for (shape_val, color_val) in pairs:
        idx += 1
        ms, pos_s = match_value_to_tokens(shape_val, tok_norm, pos, 3)
        mc, pos   = match_value_to_tokens(color_val, tok_norm, pos_s, 3)
        vecs: List[np.ndarray] = []
        if mode in ("shape", "both") and ms: vecs.extend(get_vec(ti) for ti in ms)
        if mode in ("color", "both") and mc: vecs.extend(get_vec(ti) for ti in mc)
        if not vecs: continue

        v = nanmean_vectors(vecs)
        if v is None: continue

        grid = v.reshape(GH, GW)
        hm = heatmap_rgba(grid, alpha=alpha, vmax=None)
        hm_big = hm.resize((W, H), resample=Image.BILINEAR).filter(
            ImageFilter.GaussianBlur(radius=blur_px)
        )

        comp = base.copy()
        comp.paste(hm_big, (0, 0), hm_big)  # overlay only; NO caption text

        out_path = out_dir / f"L{layer_idx:02d}_pair_{idx:02d}.webp"
        try:
            comp.save(out_path, format="WEBP", quality=90, method=6)
            metas.append({
                "mode": mode, "layer": layer_idx, "pair": idx,
                "file": out_path, "shape": shape_val, "color": color_val
            })
        except Exception as e:
            print(f"[WARN] {out_path} -> {e}")
    return metas

# -------- I/O helpers --------
def discover_triplets(ds_dir_15: Path) -> List[str]:
    out = []
    if not ds_dir_15.is_dir(): return out
    for p in sorted(ds_dir_15.iterdir(), key=lambda x: (len(x.name), x.name)):
        if p.is_dir():
            try: int(p.name); out.append(p.name)
            except: pass
    return out

def io_paths(in_root: Path, dataset: str, triplet: str) -> Tuple[Path, Path]:
    d = in_root / dataset / "15" / triplet
    return d / f"nObjects=15_triplet={triplet}_0.png", d / f"attn_patches_nObjects=15_triplet={triplet}_0.json"

# -------- main --------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in_root", type=Path, required=True, help="root of datasets (row/ just_number/ ...)")
    ap.add_argument("--out_root", type=Path, default=Path("static_tiles"))
    ap.add_argument("--datasets", nargs="*", default=["row","just_number","just_scan","simple"])
    ap.add_argument("--layers", nargs="*", type=int, default=list(range(28)))
    ap.add_argument("--modes", nargs="*", default=["shape","color","both"], choices=["shape","color","both"])
    ap.add_argument("--triplets", nargs="*", default=None)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--blur", type=float, default=2.0)
    ap.add_argument("--objects_dir_name", type=str, default="objs15",
                    help="explicit folder name for the '15 objects' level in output to avoid collision with layer 15")
    args = ap.parse_args()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {"entries": {}, "note": "paths are relative to manifest.json directory"}
    total_tiles = 0

    for ds in args.datasets:
        ds15 = args.in_root / ds / "15"
        trips = args.triplets if args.triplets else discover_triplets(ds15)
        if not trips:
            print(f"[SKIP] {ds}: no triplets found")
            continue
        manifest["entries"].setdefault(ds, {})
        for trip in trips:
            base_img_path, json_path = io_paths(args.in_root, ds, trip)
            if not base_img_path.is_file() or not json_path.is_file():
                print(f"[SKIP] {ds}/{trip}: missing base or json")
                continue

            trip_out = out_root / ds / args.objects_dir_name / trip
            trip_out.mkdir(parents=True, exist_ok=True)

            # copy base
            base_out = trip_out / "base.webp"
            if not base_out.exists():
                try:
                    Image.open(base_img_path).convert("RGB").save(base_out, format="WEBP", quality=92, method=6)
                except Exception as e:
                    print(f"[WARN] base copy {base_out} -> {e}")

            all_meta: List[dict] = []
            for mode in args.modes:
                mode_dir = trip_out / mode
                for L in args.layers:
                    tiles = bake_one(base_img_path, json_path, mode_dir, L, mode,
                                     alpha=args.alpha, blur_px=args.blur)
                    for m in tiles: m["file"] = Path(m["file"]).resolve().relative_to(out_root).as_posix()
                    all_meta.extend(tiles)
                    total_tiles += len(tiles)
                    print(f"[OK] {ds}/{trip} mode={mode} L={L:02d} -> {len(tiles)} tiles")

            manifest["entries"][ds][trip] = {
                "base": base_out.resolve().relative_to(out_root).as_posix(),
                "tiles": sorted(all_meta, key=lambda m: (m["mode"], m["layer"], m["pair"]))
            }

    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Wrote manifest -> {manifest_path}")
    print(f"Done. total tiles made={total_tiles}")

if __name__ == "__main__":
    main()
