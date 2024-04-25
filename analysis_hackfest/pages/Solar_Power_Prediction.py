import pandas as pd
import streamlit as st

from analysis_hackfest.models.solar_power import SolarPowerRegressor


@st.cache_resource
def get_solar_power_model() -> SolarPowerRegressor:
    model = SolarPowerRegressor()
    model.load_model()
    return model


@st.cache_data
def get_prediction():
    return sess.run(
        [sess.get_outputs()[0].name],
        {sess.get_inputs()[0].name: X_test.to_numpy()},
    )[0].reshape(-1)


model = get_solar_power_model()
sess = model.load_model()

st.write("""
    # Solar Power Generation Prediction
""")

st.write("> **Dataset**")
st.write(model.get_dataset().sample(10))

_, X_test, _, Y_test = model.get_split_dataset()
X_test.reset_index(drop=True, inplace=True)
Y_test.reset_index(drop=True, inplace=True)

st.line_chart(
    pd.concat(
        [
            Y_test[:100],
            pd.DataFrame({"Predicted AC Power": get_prediction()})[:100],
        ],
        axis=1,
    )
)
