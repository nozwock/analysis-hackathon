import streamlit as st

main_page = st.navigation(
    {
        "Overview": [
            # Load pages from functions
            st.Page(
                "pages/home.py", title="Home", default=True, icon=":material/home:"
            ),
            st.Page("pages/future.py", title="Future", icon=":material/star_border:"),
        ],
        "Prediction": [
            st.Page("pages/co2_emission_prediction.py", title="CO2 Emission"),
            st.Page("pages/solar_power_prediction.py", title="Solar Power Generation"),
        ],
    }
)

main_page.run()
