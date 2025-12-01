"""Generate latent embeddings for extracted model parameters using a trained AutoEncoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from beautilog import logger

from config import ConfigParser, ModelAutoEncoderEvalConfig
from dataloader import ModelParameterDataset
from model import ModelAutoEncoder

_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "fp64": torch.float64,
}

_NUMPY_DTYPE_MAP: dict[str, np.dtype] = {
    "float32": np.float32,
    "float": np.float32,
    "fp32": np.float32,
    "float64": np.float64,
    "double": np.float64,
    "fp64": np.float64,
}


def _resolve_torch_dtype(name: str) -> torch.dtype:
    dtype = _TORCH_DTYPE_MAP.get(name.lower())
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype '{name}'.")
    return dtype


def _resolve_numpy_dtype(name: str) -> np.dtype:
    dtype = _NUMPY_DTYPE_MAP.get(name.lower())
    if dtype is None:
        raise ValueError(f"Unsupported numpy dtype '{name}'.")
    return dtype


def _resolve_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(config: ModelAutoEncoderEvalConfig, device: torch.device) -> ModelAutoEncoder:
    weights_path = Path(config.weights_path).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"AutoEncoder weights not found: {weights_path}")

    model = ModelAutoEncoder(device=device)
    state_dict = torch.load(weights_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded AutoEncoder weights from %s", weights_path)
    return model


def _select_files(dataset: ModelParameterDataset, substring: str) -> list[tuple[int, Path]]:
    if not substring:
        return list(enumerate(dataset.files))
    matches = [(idx, path) for idx, path in enumerate(dataset.files) if substring in path.name]
    if not matches:
        logger.warning("No files matching substring '%s' were found; processing all files instead.", substring)
        return list(enumerate(dataset.files))
    return matches


def _save_embedding(
    embedding: np.ndarray,
    *,
    source_path: Path,
    root_dir: Path,
    output_dir: Path,
    save_dtype: np.dtype,
) -> None:
    relative = source_path.relative_to(root_dir)
    destination_dir = output_dir / relative.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / f"{relative.stem}_embedding.npz"

    np.savez(
        destination_path,
        embedding=embedding.astype(save_dtype, copy=False),
        source=np.asarray(str(source_path), dtype=object),
    )
    logger.debug("Saved embedding to %s", destination_path)


def main() -> None:
    torch.set_grad_enabled(False)
    ConfigParser.load()
    eval_config = ConfigParser.get(ModelAutoEncoderEvalConfig)

    device = _resolve_device(eval_config.device)
    model = _load_model(eval_config, device)
    model.to(device)

    parameter_root = Path(eval_config.parameter_root).expanduser()
    if not parameter_root.exists():
        raise FileNotFoundError(f"Parameter directory not found: {parameter_root}")

    input_dtype = _resolve_torch_dtype(eval_config.input_dtype)
    dataset = ModelParameterDataset(
        root_dir=parameter_root,
        dtype=input_dtype,
        flatten=eval_config.flatten,
        normalize=eval_config.normalize_inputs,
    )

    save_dtype = _resolve_numpy_dtype(eval_config.save_dtype)
    output_root = Path(eval_config.output_directory).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    batch_size = max(1, int(eval_config.batch_size))
    file_entries = _select_files(dataset, eval_config.file_substring)
    total_files = len(file_entries)
    if total_files == 0:
        logger.warning("No parameter archives found in %s; nothing to do.", parameter_root)
        return

    logger.info(
        "Generating embeddings for %d archives from %s (batch_size=%d, device=%s).",
        total_files,
        parameter_root,
        batch_size,
        device,
    )

    for start in range(0, total_files, batch_size):
        batch_entries = file_entries[start : start + batch_size]
        batch_tensors = [dataset[idx] for idx, _ in batch_entries]
        batch = torch.stack(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            encoded, _ = model(batch)

        embeddings = encoded.detach().cpu().numpy()

        for embedding, (_, source_path) in zip(embeddings, batch_entries):
            _save_embedding(
                embedding,
                source_path=source_path,
                root_dir=parameter_root,
                output_dir=output_root,
                save_dtype=save_dtype,
            )

    logger.info("Completed embedding generation for %d archive(s).", total_files)


if __name__ == "__main__":
    main()
