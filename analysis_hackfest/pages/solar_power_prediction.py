import numpy as np
import pandas as pd
import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel

from analysis_hackfest.models.solar_power import SolarPowerRegressor


class InputFeatures(BaseModel):
    DAILY_YIELD: float = 5000
    TOTAL_YIELD: float = 200_000
    AMBIENT_TEMPERATURE: float = 30
    MODULE_TEMPERATURE: float = 40
    IRRADIATION: float = 0.5
    DC_POWER: float = 100


@st.cache_resource
def get_model() -> SolarPowerRegressor:
    model = SolarPowerRegressor()
    model.load_model()
    return model


@st.cache_data
def get_prediction(inputs: np.ndarray):
    return sess.run(
        [sess.get_outputs()[0].name],
        {sess.get_inputs()[0].name: inputs},
    )[0].reshape(-1)


@st.cache_data
def cached_samples():
    return (
        model.get_dataset().sample(100),
        pd.concat(
            [
                Y_test,
                pd.DataFrame({"Predicted AC Power": get_prediction(X_test.to_numpy())}),
            ],
            axis=1,
        ).sample(100),
    )


model = get_model()
sess = model.load_model()

_, X_test, _, Y_test = model.get_split_dataset()
X_test.reset_index(drop=True, inplace=True)
Y_test.reset_index(drop=True, inplace=True)

data_sample, series_sample = cached_samples()

st.write("""
    # Solar Power Generation Prediction
""")

st.write("### Dataset Sample")
st.write(data_sample)

st.write("### Series Line Plot")
st.line_chart(series_sample)


st.write("### Estimater")


@st.experimental_fragment
def estimator():
    input_features_data = sp.pydantic_input(key="form", model=InputFeatures)
    inputs = np.fromiter(input_features_data.values(), dtype=np.float32).reshape(-1, 6)
    st.html(
        f"<p style='font-size: 1.25em;'>Predicted Ac Power: <code>{get_prediction(inputs)[0]}</code></p>",
    )


estimator()
