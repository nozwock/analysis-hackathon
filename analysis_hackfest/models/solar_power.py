from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pandas as pd
import skl2onnx
from onnx.onnx_ml_pb2 import ModelProto
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .. import assets
from . import ModelProtocol

SOLAR_POWER_REGRESSOR_MODEL_PATH = Path(
    str(files(assets).joinpath("solar-power-regressor.onnx"))
)
GENERATION_DATA_PATH = Path(str(files(assets).joinpath("Plant_2_Generation_Data.csv")))
WEATHER_DATA_PATH = Path(str(files(assets).joinpath("Plant_2_Weather_Sensor_Data.csv")))


class SolarPowerRegressor(ModelProtocol):
    model: RandomForestRegressor | None = None
    stored_model: onnxruntime.InferenceSession | None = None
    dataset: pd.DataFrame | None = None
    split_dataset: Any | None = None
    input_features: int | None = None

    def get_dataset(self) -> pd.DataFrame:
        if self.dataset is None:
            generation_data = pd.read_csv(GENERATION_DATA_PATH)
            weather_data = pd.read_csv(WEATHER_DATA_PATH)

            df = pd.merge(
                generation_data.drop(columns=["PLANT_ID"]),
                weather_data.drop(columns=["PLANT_ID", "SOURCE_KEY"]),
                on="DATE_TIME",
            )

            label_encoder = LabelEncoder()
            df["SOURCE_KEY_NUMBER"] = label_encoder.fit_transform(df["SOURCE_KEY"])
            self.dataset = df

        return self.dataset

    def get_split_dataset(self) -> Any:
        if self.dataset is None:
            self.get_dataset()

        assert self.dataset is not None

        if self.split_dataset is None:
            X = self.dataset[
                [
                    "DAILY_YIELD",
                    "TOTAL_YIELD",
                    "AMBIENT_TEMPERATURE",
                    "MODULE_TEMPERATURE",
                    "IRRADIATION",
                    "DC_POWER",
                ]
            ].astype(np.float32)
            Y = self.dataset["AC_POWER"].astype(np.float32)
            self.input_features = X.shape[1]
            self.split_dataset = train_test_split(X, Y, test_size=0.2, random_state=21)

        return self.split_dataset

    def get_model(self) -> RandomForestRegressor:
        if self.model is None:
            X_train, X_test, Y_train, Y_test = self.get_split_dataset()

            model = RandomForestRegressor()
            model.fit(X_train, Y_train)
            self.model = model

        return self.model

    def try_load_model(
        self, path: Path = SOLAR_POWER_REGRESSOR_MODEL_PATH
    ) -> onnxruntime.InferenceSession | None:
        if self.stored_model is None:
            self.stored_model = (
                onnxruntime.InferenceSession(
                    path, providers=onnxruntime.get_available_providers()
                )
                if path.is_file()
                else None
            )
        return self.stored_model

    def load_model(
        self, path: Path = SOLAR_POWER_REGRESSOR_MODEL_PATH
    ) -> onnxruntime.InferenceSession:
        if self.stored_model is None:
            if (ret := self.try_load_model(path)) is None:
                self.store_model()
                ret = self.try_load_model(path)

            self.stored_model = ret

        assert self.stored_model is not None

        return self.stored_model

    def store_model(self, path: Path = SOLAR_POWER_REGRESSOR_MODEL_PATH) -> bool:
        if self.model is None:
            model = self.get_model()
            assert self.input_features is not None

            onnx_model = skl2onnx.to_onnx(
                model,
                initial_types=[("input", FloatTensorType([None, self.input_features]))],
            )

            match onnx_model:
                case ModelProto():
                    with open(path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                case _:
                    return False

        return True


# Run via:
# python -m analysis_hackathon.models.solar_power
if __name__ == "__main__":
    from sklearn.metrics import r2_score

    model = SolarPowerRegressor()
    sess = model.load_model()

    _, X_test, _, Y_test = model.get_split_dataset()
    preds = sess.run(
        [sess.get_outputs()[0].name],
        {sess.get_inputs()[0].name: X_test.to_numpy()[:10]},
    )[0]

    print(f"Score: {np.round(r2_score(preds, Y_test[:10]) * 100, 2)}%")
