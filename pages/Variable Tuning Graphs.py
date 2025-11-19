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

def reset():
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
    return parms
def add_Param(choice, amount, arr):
    if (choice == 'k_w'):
        arr[0] = amount

    elif (choice == 'k_h'):
        arr[1] = amount

    elif (choice == 'd_w'):
        arr[2] = amount

    elif (choice == 'd_h'):
        arr[3] = amount

    elif (choice == 'k_g'):
        arr[4] = amount

    elif (choice == 'k_l'):
        arr[5] = amount
    elif (choice == 'k_Wr'):
        arr[6] = amount
    elif (choice == 'd_g'):
        arr[7] = amount
    elif (choice == 'd_l'):
        arr[8] = amount
    elif (choice == 'd_Wr'):
        arr[9] = amount




def run_simulation(parms):
    t = 4.4  # time in Gyr

    loopCount = 0
    inLoop = True

    Wo = 7  # original seawater d18O
    W_SS = -1  # steady state
    num_steps = 100  # num of initial model steps
    time = np.linspace(0, 4.5, num=num_steps)  # sample every 250 myr
    weath_time_on_twostep = 4.5 - 2.4  # in Ga
    weath_time_on = 4.5 - 1.9
    weath_time_early = 4.5 - 4.43
    weath_time_late = 4.5 - 0.9

    Delt_hiT_mid = 1.5

    Delt_lowT = 9.3  # mueh = 9.3
    Delt_water_recycling = 2.5  # mueh = 2.5

    k_weath, k_hiT, Delt_weath, Delt_hiT = parms[0], parms[1], parms[2], parms[3]
    k_growth, k_loT, k_W_recycling, = parms[4], parms[5], parms[6]
    Delt_growth, Delt_lowT, Delt_water_recycling = parms[7], parms[8], parms[9]

    # calculate steady state in 250 myr increments
    del_graniteo = np.linspace(7.8, 7.8, num=num_steps)

    del_basalto = 5.5
    del_WR = 7
    bb = 0.4
    bb2 = 0.1

    Delt_hiT_change = (Delt_hiT - Delt_hiT_mid) + (Delt_hiT - Delt_hiT_mid) * 0.5 * (
            1 + np.tanh((np.subtract(time, weath_time_on) / bb))) - 1
    Delt_hiT_change_late = (Delt_hiT - Delt_hiT_mid) + (Delt_hiT - Delt_hiT_mid) * 0.5 * (
            1 + np.tanh((np.subtract(time, weath_time_late) / bb))) - 1
    Delt_hiT_twostep = (Delt_hiT - Delt_hiT_mid) + (Delt_hiT - Delt_hiT_mid) * 0.5 * (
            1 + np.tanh((np.subtract(time, weath_time_on_twostep) / bb2))) - 1

    k_growth_change = 0.5 * k_growth * (1 + np.tanh((np.subtract(time, weath_time_on) / bb)))
    k_growth_late = 0.5 * k_growth * (1 + np.tanh((np.subtract(time, weath_time_late) / bb)))
    k_growth_early = 0.5 * k_growth * (1 + np.tanh((np.subtract(time, weath_time_early) / bb)))

    k_weathering_change = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_on) / bb)))
    k_weathering_late = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_late) / bb)))
    k_weathering_early = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_early) / bb)))

    two_step_time = 4.5 - 2
    k_growth_twostep = 0.5 * k_growth * (1 + np.tanh((np.subtract(time, weath_time_on_twostep) / (bb2))))

    weath_time_mid = 4.5 - 1.65
    k_weathering_mid = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_mid) / bb)))
    k_weathering_two_step = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_mid) / bb2)))

    k_loT_change = k_loT * np.ones(time.size)  # keep it the same
    k_hiT_change = k_hiT * np.ones(time.size)

    k_water_change = k_W_recycling * np.ones(time.size)  #

    del_steady_change = np.zeros(time.size)
    del_steady_early = np.zeros(time.size)
    del_steady_late = np.zeros(time.size)
    del_steady_two_step = np.zeros(time.size)

    k_sum = np.zeros(time.size)
    k_sum_early = np.zeros(time.size)
    k_sum_late = np.zeros(time.size)
    k_sum_twostep = np.zeros(time.size)

    for istep in range(0, time.size):
        top = np.sum([k_weathering_change[istep] * (del_graniteo[istep] - Delt_weath), \
                      k_growth_change[istep] * (del_graniteo[istep] - Delt_growth), \
                      k_hiT_change[istep] * (del_basalto - Delt_hiT_change[istep]), \
                      k_loT_change[istep] * (del_basalto - Delt_lowT), \
                      k_water_change[istep] * (del_WR - Delt_water_recycling)])
        top_two_step = np.sum([k_weathering_two_step[istep] * (del_graniteo[istep] - Delt_weath), \
                               k_growth_twostep[istep] * (del_graniteo[istep] - Delt_growth), \
                               k_hiT_change[istep] * (del_basalto - Delt_hiT_twostep[istep]), \
                               k_loT_change[istep] * (del_basalto - Delt_lowT), \
                               k_water_change[istep] * (del_WR - Delt_water_recycling)])
        top_early = np.sum([k_weathering_early[istep] * (del_graniteo[istep] - Delt_weath), \
                            k_growth_early[istep] * (del_graniteo[istep] - Delt_growth), \
                            k_hiT_change[istep] * (del_basalto - Delt_hiT), \
                            k_loT_change[istep] * (del_basalto - Delt_lowT), \
                            k_water_change[istep] * (del_WR - Delt_water_recycling)])
        top_late = np.sum([k_weathering_late[istep] * (del_graniteo[istep] - Delt_weath), \
                           k_growth_late[istep] * (del_graniteo[istep] - Delt_growth), \
                           k_hiT_change[istep] * (del_basalto - Delt_hiT_change_late[istep]), \
                           k_loT_change[istep] * (del_basalto - Delt_lowT), \
                           k_water_change[istep] * (del_WR - Delt_water_recycling)])

        k_sum[istep] = np.sum(
            [k_weathering_change[istep], k_growth_change[istep], k_hiT_change[istep], k_loT_change[istep],
             k_water_change[istep]])
        k_sum_early[istep] = np.sum(
            [k_weathering_early[istep], k_growth_early[istep], k_hiT_change[istep], k_loT_change[istep],
             k_water_change[istep]])
        k_sum_late[istep] = np.sum(
            [k_weathering_late[istep], k_growth_late[istep], k_hiT_change[istep], k_loT_change[istep],
             k_water_change[istep]])
        k_sum_twostep[istep] = np.sum(
            [k_weathering_two_step[istep], k_growth_twostep[istep], k_hiT_change[istep], k_loT_change[istep],
             k_water_change[istep]])

        del_steady_change[istep] = top / k_sum[istep]
        del_steady_early[istep] = top_early / k_sum_early[istep]
        del_steady_late[istep] = top_late / k_sum_late[istep]
        del_steady_two_step[istep] = top_two_step / k_sum_twostep[istep]

        # calculate dW at for each steady state
        time_new = np.linspace(0.01, 4.5, num=1000)
        f1 = sp.interpolate.interp1d(time, del_steady_change)
        f2 = sp.interpolate.interp1d(time, k_sum)
        steady_interp = f1(time_new)
        k_sum_interp = f2(time_new)
        f1_late = sp.interpolate.interp1d(time, del_steady_late)
        f2_late = sp.interpolate.interp1d(time, k_sum_late)
        steady_interp_late = f1_late(time_new)
        k_sum_interp_late = f2_late(time_new)

        steady_interp_late = f1_late(time_new)
        k_sum_interp_late = f2_late(time_new)

        f1_early = sp.interpolate.interp1d(time, del_steady_early)
        f2_early = sp.interpolate.interp1d(time, k_sum_early)
        steady_interp_early = f1_early(time_new)
        k_sum_interp_early = f2_early(time_new)

        f1_twostep = sp.interpolate.interp1d(time, del_steady_two_step)
        f2_twostep = sp.interpolate.interp1d(time, k_sum_twostep)
        steady_interp_twostep = f1_twostep(time_new)
        k_sum_interp_twostep = f2_twostep(time_new)

        dW_middle = np.add(np.subtract(Wo, steady_interp) * np.exp(-np.multiply(time_new, k_sum_interp)), steady_interp)
        dW_early = np.add(np.subtract(Wo, steady_interp_early) * np.exp(-np.multiply(time_new, k_sum_interp_early)),
                          steady_interp_early)
        dW_late = np.add(np.subtract(Wo, steady_interp_late) * np.exp(-np.multiply(time_new, k_sum_interp_late)),
                         steady_interp_late)
        dW_twostep = np.add(
            np.subtract(Wo, steady_interp_twostep) * np.exp(-np.multiply(time_new, k_sum_interp_twostep)),
            steady_interp_twostep)

        decay_const_low = 0.02
        decay_const_high = 0.04
        dW_decay_low = []  # np.zeros(len(time_new))
        dW_decay_high = []  # np.zeros(len(time_new))
        whatstep = []

    return dW_early, dW_middle, dW_late, dW_twostep, time_new

parms = {}
parms = reset()
param_values = np.array(parms, dtype=float)
st.session_state.setdefault("param_values", param_values)

st.set_page_config(page_title="Variable Graph", page_icon="📈")

st.title("Interactive D18O Model")
st.markdown("Variable Tuning Graph")
st.sidebar.header("Variable Tuning Graphs")

if st.button("Reset"):
    defaults = reset()
    st.session_state.param_values = np.array(defaults, dtype=float)
    for i, val in enumerate(defaults):
        st.session_state[f"param_{i}"] = val
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
        early, middle, late, twostep, time_new = run_simulation(parms)
        fig, ax = plt.subplots(figsize=(12, 6))
        # Reference points
        # Load or define your data (from original code)
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




