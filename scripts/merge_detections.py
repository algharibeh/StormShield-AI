"""Merge detector output JSONs from several blobs into one nuScenes-format file.

Each input has {"meta": {...}, "results": {sample_token: [boxes]}}. The output
is the union of the results dicts (later files win on duplicate tokens, which
are reported) with the meta of the first file.

Usage:
  python scripts/merge_detections.py --out detections/centerpoint_b1b10.json \
      detections/centerpoint.json detections/centerpoint_blob10.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    merged = {}
    meta = None
    for path in args.inputs:
        with open(path) as f:
            d = json.load(f)
        if meta is None:
            meta = d.get("meta", {})
        elif d.get("meta", {}) != meta:
            print(f"warning: meta differs in {path}: {d.get('meta')} vs {meta}")
        dup = set(merged) & set(d["results"])
        if dup:
            print(f"warning: {len(dup)} duplicate sample tokens in {path}; later file wins")
        merged.update(d["results"])
        print(f"{path}: {len(d['results'])} samples")
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": merged}, f)
    print(f"wrote {args.out}: {len(merged)} samples")


if __name__ == "__main__":
    main()
