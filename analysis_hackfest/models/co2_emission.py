import logging
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import assets
from . import ModelProtocol

logger = logging.getLogger(__name__)

CO2_EMISSION_REGRESSOR_MODEL_PATH = Path(
    str(files(assets).joinpath("co2-emission-regressor.onnx"))
)
FUEL_CONSUMPTION_DATA_PATH = Path(str(files(assets).joinpath("FuelConsumptionCo2.csv")))


class Co2EmissionRegressor(ModelProtocol):
    model: Pipeline | None = None
    stored_model: onnxruntime.InferenceSession | None = None
    dataset: pd.DataFrame | None = None
    split_dataset: Any | None = None
    input_features: int | None = None

    def get_dataset(self) -> pd.DataFrame:
        if self.dataset is None:
            df = pd.read_csv(FUEL_CONSUMPTION_DATA_PATH)
            df.drop(columns=["MODEL_YEAR"], inplace=True)

            ## Dropping 'Fuel Consumption Hwy (L/100 km)' and 'Fuel Consumption City (L/100 km)'
            ## because they have perfect correlation with 'Fuel Consumption Comb (L/100 km)'
            df.drop(
                columns=["FUEL_CONSUMPTION_CITY", "FUEL_CONSUMPTION_HWY"],
                inplace=True,
            )

            # Encoding only fuel type, vehicle class and Transmission out of all the non-numeric features
            # using pd.get_dummies since the rest contain too many unique values and will increase the feature so much
            df.drop(columns=["MAKE", "MODEL", "TRANSMISSION"], inplace=True)
            fuel_type_encoded = pd.get_dummies(
                df["FUEL_TYPE"], prefix="FUEL", drop_first=True
            )
            vehicle_class_encoded = pd.get_dummies(
                df["VEHICLE_CLASS"], prefix="VEHICLE", drop_first=True
            )
            df.drop(columns=["FUEL_TYPE", "VEHICLE_CLASS"], axis=1, inplace=True)
            df_encoded = pd.concat(
                [df, fuel_type_encoded, vehicle_class_encoded], axis=1
            )

            df_encoded.drop_duplicates(keep="first", inplace=True)

            self.dataset = df_encoded

        return self.dataset

    def get_split_dataset(self) -> Any:
        if self.dataset is None:
            self.get_dataset()

        assert self.dataset is not None

        if self.split_dataset is None:
            X = self.dataset.drop(["CO2_EMISSIONS"], axis=1).astype(np.float32)
            Y = self.dataset["CO2_EMISSIONS"].astype(np.float32)

            self.input_features = X.shape[1]
            self.split_dataset = train_test_split(X, Y, test_size=0.3, random_state=42)

        return self.split_dataset

    def get_model(self) -> Pipeline:
        if self.model is None:
            logger.info(f"{self.model=}, Generating a new model")

            X_train, X_test, Y_train, Y_test = self.get_split_dataset()

            std_scaler = StandardScaler()

            regressor = GradientBoostingRegressor(
                random_state=69, n_estimators=90, learning_rate=0.655
            )
            estimator = Pipeline(
                steps=[("std_scaler", std_scaler), ("gdab_final", regressor)]
            )
            estimator.fit(X_train, Y_train)

            Y_pred = estimator.predict(X_test)
            rmse = mean_squared_error(Y_test, Y_pred) ** (1 / 2)
            r2 = r2_score(Y_pred, Y_test)
            logger.debug(f"{rmse=}, {r2=}")

            self.model = estimator

        return self.model

    def try_load_model(
        self, path: Path = CO2_EMISSION_REGRESSOR_MODEL_PATH
    ) -> onnxruntime.InferenceSession | None:
        return super().try_load_model(path)

    def load_model(
        self, path: Path = CO2_EMISSION_REGRESSOR_MODEL_PATH
    ) -> onnxruntime.InferenceSession:
        return super().load_model(path)

    def store_model(self, path: Path = CO2_EMISSION_REGRESSOR_MODEL_PATH) -> None:
        return super().store_model(path)


if __name__ == "__main__":
    model = Co2EmissionRegressor()
    sess = model.load_model()

    _, X_test, _, Y_test = model.get_split_dataset()
    Y_pred = sess.run(
        [sess.get_outputs()[0].name],
        {sess.get_inputs()[0].name: X_test.to_numpy()[:100]},
    )[0]

    r2 = r2_score(Y_pred, Y_test[:100])
    print(f"{r2=}")
