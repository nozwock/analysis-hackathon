from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


@dataclass
class ModelProtocol(Protocol, Generic[T, U, V]):
    model: T | None = None
    stored_model: U | None = None
    dataset: V | None = None
    split_dataset: Any | None = None

    def get_dataset(self) -> V: ...
    def get_split_dataset(self) -> Any: ...
    def get_model(self) -> T: ...

    def try_load_model(self, path: Path) -> U | None: ...
    def load_model(self, path: Path) -> U: ...
    def store_model(self, path: Path) -> bool: ...
