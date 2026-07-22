import json
import os
import struct

import numpy as np


MAGIC = b"FURSEE_VECTOR_DB_V1\n"
METADATA_LENGTH_STRUCT = "<Q"


def missing_feature_db_message(folder):
    return (
        f"Feature database not found in {folder}. Run `python db.py cold` first. "
        "To add new photos later, run `python db.py append --append-folder PATH`."
    )


def resolve_feature_db_path(folder, preferred_name="features.fvdb", legacy_name="features.json"):
    preferred_path = os.path.join(folder, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path

    legacy_path = os.path.join(folder, legacy_name)
    if os.path.exists(legacy_path):
        return legacy_path

    return None


def require_feature_db(folder, preferred_name="features.fvdb", legacy_name="features.json"):
    db_path = resolve_feature_db_path(folder, preferred_name=preferred_name, legacy_name=legacy_name)
    if db_path is None:
        raise FileNotFoundError(missing_feature_db_message(folder))
    return db_path


class VectorDatabase:
    def __init__(self, keys=None, vectors=None, metric="cosine", metadata=None):
        self.keys = list(keys or [])
        if vectors is None:
            self.vectors = np.empty((0, 0), dtype=np.float32)
        else:
            self.vectors = np.asarray(vectors, dtype=np.float32)
            if self.vectors.ndim == 1:
                self.vectors = self.vectors.reshape(1, -1)
        self.metric = metric
        self.metadata = self._normalize_metadata(metadata)
        self._validate()

    @staticmethod
    def default_metadata_for_key(key):
        return {
            "is_augmented": False,
            "source_key": key,
            "augment_index": None,
        }

    def _normalize_metadata(self, metadata):
        if metadata is None:
            return [self.default_metadata_for_key(key) for key in self.keys]

        normalized = []
        for key, item in zip(self.keys, metadata):
            if item is None:
                normalized.append(self.default_metadata_for_key(key))
                continue
            meta = dict(item)
            meta.setdefault("is_augmented", False)
            meta.setdefault("source_key", key)
            meta.setdefault("augment_index", None)
            normalized.append(meta)
        return normalized

    def _validate(self):
        if self.vectors.ndim != 2:
            raise ValueError("Vector matrix must be two-dimensional.")
        if len(self.keys) != self.vectors.shape[0]:
            raise ValueError("The number of keys must match the number of vectors.")
        if len(self.metadata) != len(self.keys):
            raise ValueError("The number of metadata entries must match the number of keys.")

    @property
    def dimension(self):
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            magic = f.read(len(MAGIC))
            if magic != MAGIC:
                raise ValueError(f"Invalid vector database file: {path}")
            metadata_length_bytes = f.read(struct.calcsize(METADATA_LENGTH_STRUCT))
            metadata_length = struct.unpack(METADATA_LENGTH_STRUCT, metadata_length_bytes)[0]
            metadata = json.loads(f.read(metadata_length).decode("utf-8"))
            raw_vectors = f.read()

        count = int(metadata.get("count", 0))
        dimension = int(metadata.get("dimension", 0))
        keys = metadata.get("keys", [])
        item_metadata = metadata.get("metadata")
        if count == 0 or dimension == 0:
            vectors = np.empty((count, dimension), dtype=np.float32)
        else:
            vectors = np.frombuffer(raw_vectors, dtype=np.float32).copy().reshape(count, dimension)
        return cls(keys=keys, vectors=vectors, metric=metadata.get("metric", "cosine"), metadata=item_metadata)

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        keys = [item["key"] for item in items]
        vectors = [item.get("vector", item.get("value")) for item in items]
        metadata = [item.get("metadata") for item in items]
        return cls(keys=keys, vectors=vectors, metric="cosine", metadata=metadata)

    @classmethod
    def load_auto(cls, folder, preferred_name="features.fvdb", legacy_name="features.json"):
        db_path = resolve_feature_db_path(folder, preferred_name=preferred_name, legacy_name=legacy_name)
        if db_path is None:
            raise FileNotFoundError(missing_feature_db_message(folder))

        if db_path.endswith(".json"):
            return cls.from_json(db_path)
        return cls.load(db_path)

    def save(self, path):
        self._validate()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        vectors = np.ascontiguousarray(self.vectors, dtype=np.float32)
        metadata = {
            "version": 1,
            "metric": self.metric,
            "dtype": "float32",
            "dimension": int(vectors.shape[1]) if vectors.size else 0,
            "count": int(vectors.shape[0]),
            "keys": self.keys,
            "metadata": self.metadata,
        }
        metadata_bytes = json.dumps(metadata, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack(METADATA_LENGTH_STRUCT, len(metadata_bytes)))
            f.write(metadata_bytes)
            f.write(vectors.tobytes(order="C"))

    def add_many(self, items):
        new_keys = []
        new_vectors = []
        new_metadata = []
        for item in items:
            if isinstance(item, dict):
                key = item["key"]
                vector = item.get("vector", item.get("value"))
                metadata = item.get("metadata", self.default_metadata_for_key(key))
            else:
                key, vector = item
                metadata = self.default_metadata_for_key(key)
            new_keys.append(key)
            new_vectors.append(vector)
            metadata = dict(metadata)
            metadata.setdefault("is_augmented", False)
            metadata.setdefault("source_key", key)
            metadata.setdefault("augment_index", None)
            new_metadata.append(metadata)

        if not new_keys:
            return

        new_vectors = np.asarray(new_vectors, dtype=np.float32)
        if new_vectors.ndim == 1:
            new_vectors = new_vectors.reshape(1, -1)

        if self.vectors.size == 0:
            self.vectors = new_vectors
        else:
            if self.vectors.shape[1] != new_vectors.shape[1]:
                raise ValueError("New vectors must have the same dimension as existing vectors.")
            self.vectors = np.vstack([self.vectors, new_vectors])
        self.keys.extend(new_keys)
        self.metadata.extend(new_metadata)
        self._validate()

    def normalized_vectors(self):
        if self.vectors.size == 0:
            return self.vectors.copy()
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        return self.vectors / (norms + 1e-8)

    def is_augmented_mask(self):
        return np.array([
            bool(item.get("is_augmented", False))
            for item in self.metadata
        ], dtype=bool)

    def original_mask(self):
        return ~self.is_augmented_mask()

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return VectorDatabase(
            keys=[self.keys[int(index)] for index in indices],
            vectors=self.vectors[indices],
            metric=self.metric,
            metadata=[self.metadata[int(index)] for index in indices],
        )

    def top_k(self, query_vector, k, include_mask=None):
        vectors = self.normalized_vectors()
        if vectors.shape[0] == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(vectors, query[0])
        if include_mask is not None:
            include_mask = np.asarray(include_mask, dtype=bool)
            if include_mask.shape[0] != similarities.shape[0]:
                raise ValueError("include_mask length must match the number of vectors.")
            similarities = np.where(include_mask, similarities, -np.inf)
        top_indices = np.argsort(similarities)[::-1][:k]
        if include_mask is not None:
            top_indices = top_indices[np.isfinite(similarities[top_indices])]
        return top_indices, similarities[top_indices]

    def to_items(self):
        return [
            {"key": key, "vector": vector.tolist(), "metadata": metadata}
            for key, vector, metadata in zip(self.keys, self.vectors, self.metadata)
        ]
