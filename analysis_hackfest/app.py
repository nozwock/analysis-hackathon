import streamlit as st

main_page = st.navigation(
    {
        "Overview": [
            st.Page(
                "app_pages/home.py", title="Home", default=True, icon=":material/home:"
            ),
            st.Page(
                "app_pages/future.py", title="Future", icon=":material/star_border:"
            ),
        ],
        "Prediction": [
            st.Page("app_pages/co2_emission_prediction.py", title="CO2 Emission"),
            st.Page(
                "app_pages/solar_power_prediction.py", title="Solar Power Generation"
            ),
        ],
    }
)

main_page.run()
