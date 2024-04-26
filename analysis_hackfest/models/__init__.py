import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeVar

import onnxruntime
import skl2onnx
from onnx.onnx_ml_pb2 import ModelProto
from skl2onnx.common.data_types import FloatTensorType

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


@dataclass
class ModelProtocol(Protocol, Generic[T, U, V]):
    model: T | None = None
    stored_model: onnxruntime.InferenceSession | None = None
    dataset: U | None = None
    split_dataset: V | None = None
    input_features: int | None = None

    def get_dataset(self) -> U: ...
    def get_split_dataset(self) -> V: ...
    def get_model(self) -> T: ...

    def try_load_model(self, path: Path) -> onnxruntime.InferenceSession | None:
        if self.stored_model is None:
            if path.is_file():
                logger.info(f"Loading model from '{path}'")
                self.stored_model = onnxruntime.InferenceSession(
                    path, providers=onnxruntime.get_available_providers()
                )
        return self.stored_model

    def load_model(self, path: Path) -> onnxruntime.InferenceSession:
        if self.stored_model is None:
            if (ret := self.try_load_model(path)) is None:
                self.store_model(path)
                ret = self.try_load_model(path)

            self.stored_model = ret

        assert self.stored_model is not None

        return self.stored_model

    def store_model(self, path: Path) -> None:
        if self.model is None:
            model = self.get_model()
            assert self.input_features is not None

            logger.info(f"Serializing {type(self.model)!r} to ONNX")
            onnx_model = skl2onnx.to_onnx(
                model,
                initial_types=[("input", FloatTensorType([None, self.input_features]))],
            )

            match onnx_model:
                case ModelProto():
                    with open(path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                case _:
                    raise ValueError(
                        f"{onnx_model=}, to_onnx() didn't return a ModelProto"
                    )
