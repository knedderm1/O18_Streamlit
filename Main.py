import streamlit as st

st.set_page_config(
    page_title="O18 Modeling",
    page_icon="💦",
)

st.write("Modeling O18 Fractionation Over Time")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
   In nature, Oxygen Isotopes can commonly be found as O16 and O18 respectively.
   There are many fluxes that contribute to the variation in the isotopic make up of
   the oceans and atmosphere. O18 concentrations within the ocean may have 
   varied throughout time as a result of said fluxes.
   
   Variation of Oxygen isotopes can be indicative of various changes. 
   O18 isotopes can be used to predict paleo-altitudes as heavier water molecules fall preferentially first, 
   leaving lighter molecules at higher altitudes. We can also use this to predict continental emergence.
   
   The following models attempt to use our knowledge of various oxygen related fluxes to model the O18 proportion of oceans over time.
   For accuracy, we attempt to compare this to other methods of determining O18 concentrations such as inversion estimates of hydrothermal waters.
   Take a look at the various models on the sidebar. Each attempts to determine O18 using a separate model, with user input to vary parameters and observe the results.   
"""
)
