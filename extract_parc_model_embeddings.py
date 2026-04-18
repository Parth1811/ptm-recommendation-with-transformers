"""Extract model parameter embeddings from PARC probe pkl files.

Reads all pkl files in parc/cache/probes/fixed_budget_500/, extracts the
unique model_param_embedding (512-dim, from autoencoder) for each
(architecture, source_dataset) pair, and saves them as NPZ files compatible
with the model_embedding_loader.

Usage:
    python extract_parc_model_embeddings.py
    python extract_parc_model_embeddings.py --output-dir artifacts/extracted/model_combined_embeddings
    python extract_parc_model_embeddings.py --dry-run
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def extract_embeddings(
    probe_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, np.ndarray]:
    """Extract unique model embeddings from PARC probe pkl files.

    Args:
        probe_dir: Directory containing .pkl probe files
        output_dir: Where to save the NPZ embedding files
        dry_run: If True, just report what would be done

    Returns:
        Dict mapping model_name -> embedding array
    """
    pkls = sorted(probe_dir.glob("*.pkl"))
    if not pkls:
        print(f"No pkl files found in {probe_dir}")
        return {}

    print(f"Scanning {len(pkls)} pkl files in {probe_dir}")

    # Extract unique embeddings per model
    models: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}

    for pkl_path in pkls:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        emb = data.get("model_param_embedding")
        if emb is None:
            continue

        arch = data["architecture"]
        source = data["source_dataset"]
        model_name = f"{arch}_{source}"

        if model_name not in models:
            models[model_name] = emb
            sources[model_name] = str(pkl_path)

    print(f"Found {len(models)} unique model embeddings:")
    for name in sorted(models.keys()):
        print(f"  {name}: shape={models[name].shape}, dtype={models[name].dtype}")

    if dry_run:
        print("\nDry run — no files written.")
        return models

    # Save each embedding as NPZ
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for name, emb in sorted(models.items()):
        out_path = output_dir / f"{name}_embedding.npz"
        if out_path.exists():
            # Check if it's the same
            existing = np.load(out_path)["embedding"]
            if np.allclose(existing, emb):
                skipped += 1
                continue

        np.savez(out_path, embedding=emb, source=sources[name])
        written += 1

    print(f"\nSaved {written} new embeddings to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} (already exist with same values)")

    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract model embeddings from PARC probe pkl files"
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=Path("parc/cache/probes/fixed_budget_500"),
        help="Directory containing PARC probe pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/extracted/model_combined_embeddings"),
        help="Output directory for NPZ embedding files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be extracted, don't write files",
    )
    args = parser.parse_args()

    if not args.probe_dir.exists():
        print(f"Error: probe directory not found: {args.probe_dir}")
        sys.exit(1)

    extract_embeddings(args.probe_dir, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
