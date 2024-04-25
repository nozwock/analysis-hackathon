import streamlit as st

from ..models.solar_power import SolarPowerRegressor


@st.cache_data
def get_solar_power_model() -> SolarPowerRegressor:
    model = SolarPowerRegressor()
    model.load_model()
    return model


get_solar_power_model()
