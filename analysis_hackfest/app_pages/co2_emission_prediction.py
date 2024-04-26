import numpy as np
import pandas as pd
import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel

from analysis_hackfest.models.co2_emission import Co2EmissionRegressor


class InputFeatures(BaseModel): ...


@st.cache_resource
def get_model() -> Co2EmissionRegressor:
    model = Co2EmissionRegressor()
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
                pd.DataFrame(
                    {"Predicted CO2 Emission": get_prediction(X_test.to_numpy())}
                ),
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
    # CO2 Emission Prediction
""")

st.write("### Dataset Sample")
st.write(data_sample)

st.write("### Series Line Plot")
st.line_chart(series_sample)
