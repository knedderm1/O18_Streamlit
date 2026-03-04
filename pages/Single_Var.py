# !/usr/bin/env python3
# -*- coding: utf-8 -*- 5+3, 4+1, 2
"""
Created on Thu Dec 27 16:11:25 2018
#TODO gregory 1991 fluxes model
#more parameters, interface
@author: benjohnson
"""
# TODO website post w/ github, orig graph plot on site (adjust params)
# TODO start doing params with other papers
import numpy as np
import scipy as sp
import streamlit as st
# import scipy.misc
# from scipy.integrate import odeint
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import Models.Models_Working as mw
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb, ListedColormap
from matplotlib.colors import PowerNorm
import matplotlib.style as mplstyle

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
mplstyle.use('fast')
#%% Seawater oxygen isotope exchange model

def reset(model):
    if model == "Muelenbach":
        # rate constants in Gyr-1, from Muehlenbachs, 1998
        k_weath = 8  # nominal 8continental weathering
        k_growth = 1.2  # nominal 1.2continental growth
        k_hiT = 14.6  # nominal 14.6high temperature seafloor
        k_loT = 1.7  # nominal 1.7low temp seafloor/seafloor weathering
        k_W_recycling = 0.6  # nominal 0.6water recycling at subduction zones

        # fractionations (permil) btwn rock and water, from Muehlenbachs, 1998 except weathering, which we tuned to reproduce -1permil ocean
        Delt_weath = 13  # mueh = 9.6 nominal 13, newer 17
        Delt_growth = 9.8  # mueh = 9.8
        Delt_hiT_mid = 1.5  # meuh = 4.1
        Delt_hiT = 4.1
        # Delt_hiT_mid = np.zeros(len(time))

        Delt_lowT = 9.3  # mueh = 9.3
        Delt_water_recycling = 2.5  # mueh = 2.5

        parms = [k_weath, k_hiT, Delt_weath, Delt_hiT,
                 k_growth, k_loT, k_W_recycling,
                 Delt_growth, Delt_lowT, Delt_water_recycling]
    if model == "Gregory":
        delt_weath = 13
        Delt_hiT = 4.1
        Delt_lowT = 9.3
        delt_hi_and_l = Delt_lowT - Delt_hiT
        k_weath = .24  # 1/k = 420
        k_hi_lo_t = 1.6  # .016 includes hi and low temp thermal weath.
        parms = [k_weath, k_hi_lo_t, delt_weath, delt_hi_and_l, 0, 0, 0, 0, 0, 0]
    return parms

parms = {}
parms = reset("Muelenbach")
param_values = np.array(parms, dtype=float)
st.session_state.setdefault("param_values", param_values)

st.set_page_config(page_title="Variable Graph", page_icon="📈")

st.title("Interactive D18O Model")
st.markdown("Variable Tuning Graph")
st.sidebar.header("Variable Tuning Graphs")

model = st.radio(
    "Select a model",
    ["Muelenbach", "Gregory"]
)
with st.form("parameters_form"):
    st.header("Model Parameters")

    # Parameter selector (display names vs. internal values)
    parameter_options = {
        "Nom. Cont. Weathering": 0,
        "Nom. Cont. Growth": 4,
        "Nom. High Temp. Seafloor": 1,
        "Nom. Low Temp. Seafloor": 5,
        "Nom. Water Recycling": 6,
        "Cont. Weathering Fract.": 2,
        "Cont. Growth Fract.": 7,
        "High Temp Seafloor Fract.": 3,
        "Low Temp Seafloor Fract.": 8,
        "Water Recycling Fract.": 9,
    }
    if model == "Gregory":
        parameter_options = {
            "Nom. Cont. Weathering": 0,
            "Nom. High + Low Seafloor": 1,
            "Cont. Weathering Fract.": 2,
            "High + Low Seafloor Fract.": 3
        }

    if st.button("Reset"):
        defaults = np.array(reset(model), dtype=float)
        st.session_state.param_values = defaults
        for label, idx in parameter_options.items():
            st.session_state[f"param_{idx}"] = float(defaults[idx])
    st.write("Enter integer values for each parameter:")

    cols = st.columns(2)
    working_values = st.session_state.param_values.copy()

    for i, (label, key) in enumerate(parameter_options.items()):
        col = cols[i % 2]
        idx = key
        working_values[idx] = col.number_input(
            label,
            min_value=-9999.0,
            max_value=9999.0,
            value=float(st.session_state.param_values[idx]),  # <- use session_state directly
            step=1.0,
            key=f"param_{idx}",
        )
    submitted = st.form_submit_button("Run Simulation")

if submitted:
    with st.spinner("Running simulation..."):
        # st.session_state.param_values = working_values
        print("SUBMIT" + str(st.session_state.param_values))
        parms = working_values
        if (model=="Gregory"):
            mw.run_gregory(parms)
        else:
            early, middle, late, twostep, time_new = mw.run_Muelenbach(parms)
        fig, ax = plt.subplots(figsize=(12, 6))
        # Reference points
        new_data = [0.23, -0.51, -1.08, -0.2, 0.72, 2.8, 3.7, 3.3]
        new_error = [0.2, 0.5, 0.2, 0.3, 0.2, 0.03, 0.16, 0.1]
        new_ages = np.subtract(4.5, [0.002, 0.0142, 0.0916, 1.72, 1.89, 2.682, 2.735,
                                     3.24])

        other_color = 'xkcd:dull green'  # np.divide([219,168,133],255)
        our_color = 'xkcd:pumpkin'  # np.divide([144,110,110],255)
        mag_color = 'xkcd:dusky rose'
        pope_color = 'xkcd:dark teal'

        mag_oc = patches.Rectangle((0, 6), 0.1, 2, linewidth=1, edgecolor='k', facecolor=mag_color)
        ax.add_patch(mag_oc)

        Pope = patches.Rectangle((4.5 - 3.75, 0.8), 0.1, 3, linewidth=1, edgecolor='k', facecolor=pope_color,
                                 label='Ophiolite')
        ax.add_patch(Pope)

        Hodel = patches.Rectangle((4.5 - 0.71, -2.28), 0.1, 1.95, linewidth=1, edgecolor='k', facecolor=other_color)
        ax.add_patch(Hodel)
        plt.plot(time_new, early, 'k-.', linewidth=2)
        # ax3.set_yticklabels('')
        plt.plot(time_new, middle, 'k:', linewidth=2)
        plt.plot(time_new, late,'k-',linewidth=2)
        plt.plot(time_new, twostep, 'k', linewidth=2)


        inverse = ax.errorbar(new_ages, new_data, yerr=new_error, fmt='o', markerfacecolor=our_color,
                              mec='k', markersize=12, elinewidth=4, capsize=5, ecolor='k')

        inverse = ax.errorbar(new_ages, new_data, yerr=new_error, fmt='o',
                              markerfacecolor=our_color, mec='k', markersize=12,
                              elinewidth=4, capsize=5, ecolor='k')

        ax.set_xlim([0, 4.5])
        time_labels = ['4.5', '4', '3.5', '3', '2.5', '2', '1.5', '1', '0.5', '0']
        ax.set_xticklabels(time_labels)
        ax.set_xlabel('Age (Ga)')
        ax.set_ylabel('Seawater $\\delta^{18}$O')
        plt.tight_layout()
        st.pyplot(fig)




