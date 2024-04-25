import logging
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .. import assets
from . import ModelProtocol

logger = logging.getLogger(__name__)

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
            logger.info(f"{self.model=}, Generating a new model")
            X_train, X_test, Y_train, Y_test = self.get_split_dataset()

            model = RandomForestRegressor()
            model.fit(X_train, Y_train)

            Y_pred = model.predict(X_test)
            rmse = mean_squared_error(Y_test, Y_pred) ** (1 / 2)
            r2 = r2_score(Y_pred, Y_test)
            logger.debug(f"{rmse=}, {r2=}")

            self.model = model

        return self.model

    def try_load_model(
        self, path: Path = SOLAR_POWER_REGRESSOR_MODEL_PATH
    ) -> onnxruntime.InferenceSession | None:
        return super().try_load_model(path)

    def load_model(
        self, path: Path = SOLAR_POWER_REGRESSOR_MODEL_PATH
    ) -> onnxruntime.InferenceSession:
        return super().load_model(path)

    def store_model(self, path: Path = SOLAR_POWER_REGRESSOR_MODEL_PATH) -> None:
        return super().store_model(path)


# Run via:
# python -m analysis_hackathon.models.solar_power
if __name__ == "__main__":
    from sklearn.metrics import r2_score

    model = SolarPowerRegressor()
    sess = model.load_model()

    _, X_test, _, Y_test = model.get_split_dataset()
    Y_pred = sess.run(
        [sess.get_outputs()[0].name],
        {sess.get_inputs()[0].name: X_test.to_numpy()[:100]},
    )[0]

    r2 = r2_score(Y_pred, Y_test[:100])
    print(f"{r2=}")
