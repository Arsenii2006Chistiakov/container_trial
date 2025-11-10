#!/usr/bin/env python3
"""Video embedding API built on top of the Vivit pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from PIL import Image
from pydantic import BaseModel, Field
from torchcodec.decoders import VideoDecoder
from transformers import AutoModel, VivitImageProcessor
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan


logger = logging.getLogger("video_embedding_api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(asctime)s] %(name)s: %(message)s",
    )
decoder_logger = logging.getLogger("video_embedding_api.decoder")
embeddings_logger = logging.getLogger("video_embedding_api.embeddings")


DEFAULT_CONFIG_SWEEP: List[Tuple[int, int]] = [(8, 3), (7, 2), (6, 2), (5, 1)]
DEFAULT_MIN_CLUSTER_SIZE = 9
DEFAULT_MIN_TARGET_PROP = 0.75
DEFAULT_CREATOR_THRESHOLD = 0.5


class BatchShapes(BaseModel):
    """Summary of tensor shapes for a processed batch."""

    videos: int = Field(..., description="Number of videos contained in the batch")
    pixel_values_shape: List[int] = Field(
        ..., description="Shape of the pixel tensor passed to the model"
    )
    embedding_shape: List[int] = Field(
        ..., description="Shape of the embedding tensor returned by the model"
    )


class ProcessResponse(BaseModel):
    """Payload returned to the client after processing the incoming links."""

    total_links: int = Field(..., description="Number of GCS links provided by the client")
    processed: int = Field(..., description="Number of videos successfully processed")
    skipped: int = Field(..., description="Number of videos skipped or failed")
    embeddings_count_total: int = Field(
        0, description="Total number of embeddings generated for clustering"
    )
    cluster_member_count: Optional[int] = Field(
        default=None, description="Number of members in the selected cluster, if any"
    )
    cluster_embedding_dim: Optional[int] = Field(
        default=None, description="Embedding dimensionality for cluster members, if any"
    )
    batches: List[BatchShapes] = Field(
        default_factory=list, description="Shape summary for every processed batch"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Human-readable messages for skipped or failed videos",
    )
    cluster: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Clustering summary including selected video links if successful",
    )


class ProcessRequest(BaseModel):
    """Incoming API payload with GCS video links."""

    gcs_links: List[str] = Field(
        ..., min_items=1, description="Collection of gs:// links to process"
    )


def _chunked(items: List[Dict], chunk_size: int) -> Iterable[List[Dict]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def _parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    if not gcs_uri:
        raise ValueError("Empty GCS URI provided")
    prefix = "gs://"
    if not gcs_uri.startswith(prefix):
        raise ValueError(f"Unsupported GCS URI format: {gcs_uri}")
    remainder = gcs_uri[len(prefix) :]
    parts = remainder.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Malformed GCS URI: {gcs_uri}")
    return parts[0], parts[1]


class GCSDownloader:
    """Minimal helper that streams GCS objects to a local cache."""

    def __init__(self) -> None:
        self._client = storage.Client()

    def download(self, gcs_uri: str, destination_root: Path) -> Path:
        bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
        destination_root.mkdir(parents=True, exist_ok=True)

        video_dir = destination_root / uuid.uuid4().hex
        video_dir.mkdir(parents=True, exist_ok=True)
        local_path = video_dir / Path(blob_path).name

        logger.debug("Downloading %s -> %s", gcs_uri, local_path)
        bucket = self._client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(str(local_path))

        return local_path


class VideoEmbeddingPipeline:
    """Encapsulates downloading, preprocessing, and inference for video links."""

    def __init__(
        self,
        cache_root: Optional[Path] = None,
        batch_size: int = 8,
        gpu_id: int = 0,
        *,
        decoder_workers: int = 6,
        negative_embeddings_path: Optional[Path] = None,
        config_sweep: Optional[Iterable[Tuple[int, int]]] = None,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        min_target_proportion: float = DEFAULT_MIN_TARGET_PROP,
        creator_diversity_threshold: float = DEFAULT_CREATOR_THRESHOLD,
    ) -> None:
        self.cache_root = Path(cache_root) if cache_root else Path(tempfile.gettempdir()) / "video_embedding_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self._lock = threading.Lock()

        self.device = self._resolve_device(gpu_id)
        self.decoder_device = self._initial_decoder_device()
        self.decoder_workers = max(1, int(decoder_workers))

        self.processor_config = {
            "do_resize": True,
            "size": {"height": 224, "width": 224},
            "do_center_crop": True,
            "crop_size": {"height": 224, "width": 224},
            "do_rescale": True,
            "rescale_factor": 0.00784313725490196,
            "offset": True,
            "do_normalize": True,
            "return_tensors": "pt",
        }

        self.processor = VivitImageProcessor(**self.processor_config)
        self.model: Optional[AutoModel] = None
        self.gcs_downloader = GCSDownloader()
        self.negative_embeddings_path = (
            Path(negative_embeddings_path)
            if negative_embeddings_path is not None
            else Path("negative_embeddings")
        )
        self.config_sweep = list(config_sweep or DEFAULT_CONFIG_SWEEP)
        self.min_cluster_size = min_cluster_size
        self.min_target_proportion = min_target_proportion
        self.creator_diversity_threshold = creator_diversity_threshold

    def process_links(self, gcs_links: List[str]) -> Dict:
        if not gcs_links:
            raise ValueError("At least one GCS link is required")

        with self._lock:
            download_dir = Path(
                tempfile.mkdtemp(prefix="gcs_batch_", dir=str(self.cache_root))
            )
            try:
                downloaded_meta: List[Dict] = []
                download_errors: List[Dict[str, str]] = []

                for link in gcs_links:
                    try:
                        local_path = self.gcs_downloader.download(link, download_dir)
                        downloaded_meta.append({"gcs_uri": link, "path": local_path})
                    except Exception as exc:  # noqa: BLE001 - surface full context to caller
                        logger.exception("Failed to download %s", link)
                        download_errors.append({"gcs_uri": link, "error": str(exc)})

                batches: List[Dict] = []
                skipped_records: List[Dict[str, str]] = []
                total_processed = 0
                clustering_records: List[Dict[str, Any]] = []

                if downloaded_meta:
                    for chunk in _chunked(downloaded_meta, self.batch_size):
                        pixel_values, processed_meta, skipped = self._prepare_pixel_batch(chunk)
                        skipped_records.extend(skipped)

                        if pixel_values is None or not processed_meta:
                            continue

                        try:
                            embeddings = self._run_inference(pixel_values)
                        except Exception as exc:  # noqa: BLE001
                            logger.exception("Model inference failed")
                            skipped_records.extend(
                                {
                                    "gcs_uri": meta["gcs_uri"],
                                    "reason": f"inference failed: {exc}",
                                }
                                for meta in processed_meta
                            )
                            continue

                        batches.append(
                            {
                                "videos": len(processed_meta),
                                "pixel_values_shape": list(
                                    int(dim) for dim in pixel_values.shape
                                ),
                                "embedding_shape": list(
                                    int(dim) for dim in embeddings.shape
                                ),
                            }
                        )

                        embeddings_cpu = embeddings.detach().cpu()
                        for meta, embedding_tensor in zip(processed_meta, embeddings_cpu):
                            gcs_uri = meta["gcs_uri"]
                            clustering_records.append(
                                {
                                    "gcs_uri": gcs_uri,
                                    "url": gcs_uri,
                                    "video_id": Path(gcs_uri).stem if gcs_uri else None,
                                    "embedding": embedding_tensor.reshape(-1).tolist(),
                                }
                            )

                        total_processed += len(processed_meta)

                        del pixel_values
                        del embeddings
                        del embeddings_cpu
                        self._maybe_empty_cache()

                errors = [
                    f"{item['gcs_uri']}: {item['error']}" for item in download_errors
                ] + [
                    f"{item['gcs_uri']}: {item['reason']}" for item in skipped_records
                ]

                cluster_summary: Optional[Dict[str, Any]] = None
                if clustering_records:
                    cluster_result = cluster_videos_with_negatives(
                        clustering_records,
                        self.negative_embeddings_path,
                        config_sweep=self.config_sweep,
                        min_cluster_size=self.min_cluster_size,
                        min_target_proportion=self.min_target_proportion,
                        creator_diversity_threshold=self.creator_diversity_threshold,
                    )

                    if cluster_result:
                        # Include embeddings and shapes for members, plus counts
                        members_full = cluster_result["members"]
                        cluster_summary = {
                            "cluster_label": cluster_result["cluster_label"],
                            "config": cluster_result["config"],
                            "members": members_full,
                            "member_count": len(members_full),
                            "embedding_dim": (
                                int(members_full[0]["embedding_shape"][0])
                                if members_full and members_full[0].get("embedding_shape")
                                else None
                            ),
                        }

                response_payload: Dict[str, Any] = {
                    "total_links": len(gcs_links),
                    "processed": total_processed,
                    "skipped": len(download_errors) + len(skipped_records),
                    "embeddings_count_total": len(clustering_records),
                    "batches": batches,
                    "errors": errors,
                    "cluster": cluster_summary,
                }
                if cluster_summary:
                    response_payload["cluster_member_count"] = cluster_summary.get("member_count")
                    response_payload["cluster_embedding_dim"] = cluster_summary.get("embedding_dim")
                return response_payload
            finally:
                shutil.rmtree(download_dir, ignore_errors=True)
                self._maybe_empty_cache()

    def _prepare_pixel_batch(
        self, meta_batch: List[Dict]
    ) -> Tuple[Optional[torch.Tensor], List[Dict], List[Dict[str, str]]]:
        all_frames: List[Image.Image] = []
        processed_meta: List[Dict] = []
        skipped: List[Dict[str, str]] = []

        # Decode videos concurrently
        with ThreadPoolExecutor(max_workers=self.decoder_workers) as executor:
            future_to_meta = {
                executor.submit(self._sample_frames, Path(meta["path"])): meta
                for meta in meta_batch
            }
            for future in as_completed(future_to_meta):
                meta = future_to_meta[future]
                gcs_uri = meta["gcs_uri"]
                try:
                    frames = future.result()
                    if len(frames) != 32:
                        skipped.append(
                            {
                                "gcs_uri": gcs_uri,
                                "reason": f"insufficient frames ({len(frames)})",
                            }
                        )
                        continue
                    all_frames.extend(frames)
                    processed_meta.append(meta)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to decode %s: %s", gcs_uri, exc)
                    skipped.append({"gcs_uri": gcs_uri, "reason": f"decode failed: {exc}"})

        if not processed_meta:
            return None, processed_meta, skipped

        processed = self.processor(images=all_frames, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)
        pixel_values = pixel_values.reshape(len(processed_meta), 32, 3, 224, 224).contiguous()
        del processed

        return pixel_values, processed_meta, skipped

    def _sample_frames(self, video_path: Path) -> List[Image.Image]:
        decoder = self._open_decoder(video_path)
        used_device = getattr(decoder, "device", self.decoder_device)
        num_workers = getattr(decoder, "num_workers", None)
        frame_count = 0
        for _ in decoder:
            frame_count += 1
        del decoder

        if frame_count < 32:
            decoder_logger.info(
                "path=%s device=%s workers=%s frames=%d",
                str(video_path),
                used_device,
                str(num_workers),
                frame_count,
            )
            return []

        target_indices = [(j * (frame_count - 1)) // 31 for j in range(32)]
        decoder = self._open_decoder(video_path)

        frames: List[Image.Image] = []
        target_ptr = 0
        for idx, frame in enumerate(decoder):
            if target_ptr >= 32:
                break
            if idx == target_indices[target_ptr] and frame is not None:
                frame_image = Image.fromarray(
                    frame.permute(1, 2, 0).cpu().numpy().astype("uint8")
                )
                frames.append(frame_image)
                target_ptr += 1
        del decoder

        if len(frames) != 32:
            decoder_logger.info(
                "path=%s device=%s workers=%s frames=%d",
                str(video_path),
                used_device,
                str(num_workers),
                frame_count,
            )
            return []

        decoder_logger.info(
            "path=%s device=%s workers=%s frames=%d",
            str(video_path),
            used_device,
            str(num_workers),
            frame_count,
        )
        return frames

    def _run_inference(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self._ensure_model()
        with torch.no_grad():
            outputs = self.model(pixel_values)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def _ensure_model(self) -> None:
        if self.model is None:
            logger.info("Loading Vivit model onto %s", self.device)
            model = AutoModel.from_pretrained("google/vivit-b-16x2-kinetics400")
            self.model = model.to(self.device).eval()

    def _resolve_device(self, gpu_id: int) -> torch.device:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            target = gpu_id if 0 <= gpu_id < count else 0
            if gpu_id != target:
                logger.warning(
                    "Requested GPU %s unavailable; using GPU %s instead", gpu_id, target
                )
            torch.cuda.set_device(target)
            return torch.device(f"cuda:{target}")

        logger.warning("CUDA unavailable; falling back to CPU")
        return torch.device("cpu")

    def _maybe_empty_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _initial_decoder_device(self) -> str:
        if self.device.type == "cuda":
            return f"cuda:{self.device.index}"
        return "cpu"

    def _open_decoder(self, video_path: Path) -> VideoDecoder:
        try:
            return VideoDecoder(str(video_path), device=self.decoder_device)
        except RuntimeError as exc:
            message = str(exc)
            if (
                self.decoder_device != "cpu"
                and "Unsupported device" in message
            ):
                logger.warning(
                    "Decoder backend cannot use %s; falling back to CPU",
                    self.decoder_device,
                )
                self.decoder_device = "cpu"
                return VideoDecoder(str(video_path), device="cpu")
            raise


def _load_negative_embeddings(path: Optional[Path], limit: Optional[int]) -> np.ndarray:
    if path is None:
        return np.empty((0, 0), dtype=np.float32)

    path = Path(path)
    if not path.exists():
        logger.warning("Negative embeddings path not found: %s", path)
        return np.empty((0, 0), dtype=np.float32)

    def _load_file(file_path: Path) -> Optional[np.ndarray]:
        try:
            suffix = file_path.suffix.lower()
            if suffix == ".npy":
                array = np.load(file_path)
            elif suffix == ".npz":
                npz = np.load(file_path)
                first_key = next(iter(npz.files))
                array = npz[first_key]
            elif suffix == ".json":
                with file_path.open("r", encoding="utf-8") as handle:
                    array = np.asarray(json.load(handle))
            else:
                return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load negatives from %s: %s", file_path, exc)
            return None

        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim > 2:
            array = array.reshape(array.shape[0], -1)

        if array.ndim != 2:
            logger.warning(
                "Skipping negatives file with unsupported shape %s: %s",
                array.shape,
                file_path,
            )
            return None
        return array

    arrays: List[np.ndarray] = []
    if path.is_dir():
        expected_dim: Optional[int] = None
        for child in sorted(path.iterdir()):
            if not child.is_file():
                continue
            array = _load_file(child)
            if array is None or array.size == 0:
                continue
            if expected_dim is None:
                expected_dim = array.shape[1]
            if array.shape[1] != expected_dim:
                logger.warning(
                    "Skipping %s due to dimension mismatch (expected %d, got %d)",
                    child,
                    expected_dim,
                    array.shape[1],
                )
                continue
            arrays.append(array)

        if not arrays:
            logger.warning("No negative embeddings loaded from directory %s", path)
            return np.empty((0, 0), dtype=np.float32)

        data = np.vstack(arrays)
    else:
        array = _load_file(path)
        if array is None:
            return np.empty((0, 0), dtype=np.float32)
        data = array

    if limit is not None and limit > 0 and data.shape[0] > limit:
        indices = np.random.choice(data.shape[0], limit, replace=False)
        data = data[indices]
        logger.info("Limited negative embeddings to %d rows", limit)

    return data.astype(np.float32, copy=False)


def _select_cluster_from_embeddings(
    emb_target: np.ndarray,
    emb_negative: np.ndarray,
    *,
    config_sweep: Optional[Iterable[Tuple[int, int]]] = None,
    min_cluster_size: int,
    min_target_prop: float,
) -> Optional[Dict[str, Any]]:
    if emb_target.ndim != 2:
        raise ValueError("Target embeddings must be a 2D array")

    sweep = list(config_sweep or DEFAULT_CONFIG_SWEEP)
    if not sweep:
        raise ValueError("config_sweep must provide at least one parameter set")

    combined = (
        np.vstack([emb_target, emb_negative])
        if emb_negative.size
        else emb_target
    )

    scaler = StandardScaler()
    features = scaler.fit_transform(combined)
    target_count = emb_target.shape[0]

    for min_cluster, min_samples in sweep:
        clusterer = hdbscan.HDBSCAN(
            metric="euclidean",
            min_cluster_size=min_cluster,
            min_samples=min_samples,
        )
        labels = clusterer.fit_predict(features)
        if labels.size == 0:
            continue

        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue

            cluster_indices = np.where(labels == cluster_id)[0]
            target_indices = cluster_indices[cluster_indices < target_count]
            if target_indices.size == 0:
                continue

            cluster_size = cluster_indices.size
            target_ratio = target_indices.size / cluster_size

            if cluster_size < min_cluster_size:
                continue
            if target_ratio < min_target_prop:
                    continue

            return {
                "cluster_id": int(cluster_id),
                "mcs": int(min_cluster),
                "ms": int(min_samples),
                "indices": target_indices.tolist(),
                "cluster_size": int(cluster_size),
                "target_ratio": float(target_ratio),
            }

    return None


def _check_creator_diversity(records: List[Dict[str, Any]], threshold: float) -> bool:
    if threshold <= 0 or not records:
        return True

    creator_counts: Dict[str, int] = {}
    for item in records:
        url = item.get("url") or ""
        match = re.search(r"@([^/]+)", url)
        if match:
            creator = match.group(1)
            creator_counts[creator] = creator_counts.get(creator, 0) + 1

    if not creator_counts:
        return True

    total = len(records)
    top_creator, top_count = max(creator_counts.items(), key=lambda kv: kv[1])
    proportion = top_count / total

    logger.info(
        "Creator dominance check: top=@%s %d/%d (%.1f%%)",
        top_creator,
        top_count,
        total,
        proportion * 100,
    )

    return proportion <= threshold


def cluster_videos_with_negatives(
    video_records: List[Dict[str, Any]],
    negative_embeddings_path: Optional[Path],
    *,
    config_sweep: Optional[Iterable[Tuple[int, int]]] = None,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_target_proportion: float = DEFAULT_MIN_TARGET_PROP,
    creator_diversity_threshold: float = DEFAULT_CREATOR_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    if not video_records:
        raise ValueError("video_records must not be empty")

    processed_records: List[Dict[str, Any]] = []
    target_vectors: List[np.ndarray] = []

    for index, record in enumerate(video_records):
        embedding = record.get("embedding")
        if embedding is None:
            logger.warning("Skipping record %d due to missing embedding", index)
            continue

        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] != 1:
            logger.warning(
                "Skipping record %d due to incompatible embedding shape %s",
                index,
                arr.shape,
            )
            continue

        flattened = arr.reshape(-1)
        target_vectors.append(flattened)

        gcs_uri = record.get("gcs_uri")
        video_id = record.get("video_id")
        if video_id is None and gcs_uri:
            video_id = Path(gcs_uri).stem
        if video_id is None:
            video_id = f"video_{index}"

        processed_records.append(
            {
                "gcs_uri": gcs_uri,
                "url": record.get("url"),
                "video_id": video_id,
                "embedding": flattened,
                "metadata": record.get("metadata", {}),
            }
        )

    if not processed_records:
        raise ValueError("No valid video embeddings supplied for clustering")

    emb_target = np.vstack(target_vectors)
    negatives = _load_negative_embeddings(
        Path(negative_embeddings_path) if negative_embeddings_path else None,
        limit=emb_target.shape[0],
    )

    logger.info(
        "Clustering %d videos with %d negative samples",
        emb_target.shape[0],
        negatives.shape[0] if negatives.size else 0,
    )

    choice = _select_cluster_from_embeddings(
        emb_target,
        negatives,
        config_sweep=config_sweep,
        min_cluster_size=min_cluster_size,
        min_target_prop=min_target_proportion,
    )

    if not choice:
        logger.info("No cluster satisfied the configured thresholds")
        return None

    members = [processed_records[i] for i in choice["indices"]]

    if not _check_creator_diversity(members, creator_diversity_threshold):
        logger.warning("Cluster rejected because creator diversity threshold was not met")
        return None

    cluster_label = f"hdbscan-{choice['mcs']}-{choice['ms']}-c{choice['cluster_id']}"

    serialized_members: List[Dict[str, Any]] = []
    for member in members:
        embedding_array = np.asarray(member["embedding"], dtype=np.float32)
        serialized_members.append(
            {
                "gcs_uri": member.get("gcs_uri"),
                "url": member.get("url"),
                "video_id": member.get("video_id"),
                "embedding": embedding_array.tolist(),
                "embedding_shape": list(embedding_array.shape),
                "metadata": member.get("metadata", {}),
            }
        )

    return {
        "cluster_label": cluster_label,
        "config": {
            "min_cluster_size": choice["mcs"],
            "min_samples": choice["ms"],
            "cluster_id": choice["cluster_id"],
            "cluster_size": choice["cluster_size"],
            "target_ratio": choice["target_ratio"],
        },
        "members": serialized_members,
    }


app = FastAPI(title="Video Embedding Pipeline", version="1.0.0")
_pipeline = VideoEmbeddingPipeline()


@app.post("/process", response_model=ProcessResponse)
async def process_videos(payload: ProcessRequest) -> ProcessResponse:
    try:
        summary = await asyncio.to_thread(_pipeline.process_links, payload.gcs_links)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return ProcessResponse(**summary)


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    