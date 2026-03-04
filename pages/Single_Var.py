# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:11:25 2018

@author: benjohnson
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 16}

matplotlib.rc('font', **font)

def run_gregory(parms):

    inLoop = True
    while inLoop:
        t = 4.4  # time in Gyr

        Wo = 7  # original seawater d18O
        W_SS = -1  # steady state
        num_steps = 100  # num of initial model steps
        time = np.linspace(0, 4.5, num=num_steps)  # sample every 250 myr
        weath_time_on_twostep = 4.5 - 2.4  # in Ga
        weath_time_on = 4.5 - 1.9 #-1.9
        weath_time_early = 4.5 - 4.43 #4.43
        weath_time_late = 4.5 - 0.9

        # # rate constants in Gyr-1, from Muehlenbachs, 1998
        # # calculate steady state in 250 myr increments

        bb = 0.4
        bb2 = 0.1

        # rate constants, gregory
        del_WR = 7
        del_graniteo = np.linspace(7.8, 7.8, num=num_steps)
        del_basalto = 5.5
        delt_weath = parms[3]
        Delt_hiT = 4.1
        delt_hi_and_l = parms[2]
        k_weath = parms[0]  # 1/k = 420
        k_hi_lo_t = parms[1]  # .016 includes hi and low temp thermal weath.
        # delt_hi_and_l = 4.7
        # delt_weath = 11

        # delt_hi_low_change = (del_basalto - delt_hi_and_l) * k_hi_lo_t
        # delt_weath_change = (del_graniteo - delt_weath) * k_weath
        hiT_mid = 1.5

        Delt_hiT_change = (Delt_hiT- hiT_mid)+(delt_hi_and_l - hiT_mid) *.5 * (
                1 + np.tanh((np.subtract(time, weath_time_on) / bb))) - 1
        Delt_hiT_change_late = (Delt_hiT- hiT_mid)+(delt_hi_and_l - hiT_mid)*.5  * (
                1 + np.tanh((np.subtract(time, weath_time_late) / bb))) - 1
        Delt_hiT_twostep = (Delt_hiT- hiT_mid)+(delt_hi_and_l - hiT_mid) *.5  * (#TODO .5 here
                1 + np.tanh((np.subtract(time, weath_time_on_twostep) / bb2))) - 1

        k_weathering_change = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_on) / bb)))
        k_weathering_late = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_late) / bb)))
        k_weathering_early = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_early) / bb)))
        two_step_time = 4.5 - 2
        # k_growth_twostep = 0.5 * k_growth * (1 + np.tanh((np.subtract(time, weath_time_on_twostep) / (bb2))))

        weath_time_mid = 4.5 - 1.65
        k_weathering_mid = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_mid) / bb)))
        k_weathering_two_step = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_mid) / bb2)))


        k_hi_lo_change = (k_hi_lo_t) * np.ones(time.size)
        k_weath = (k_weath) * np.ones(time.size)

        del_steady_change = np.zeros(time.size)
        del_steady_early = np.zeros(time.size)
        del_steady_late = np.zeros(time.size)
        del_steady_two_step = np.zeros(time.size)

        k_sum = np.zeros(time.size)
        k_sum_early = np.zeros(time.size)
        k_sum_late = np.zeros(time.size)
        k_sum_twostep = np.zeros(time.size)

        for istep in range(0, time.size):
            top = np.sum([k_weath[istep] * (del_graniteo[istep] - delt_weath),
                          k_hi_lo_change[istep] * (del_basalto - Delt_hiT_change[istep])
                          ])
            top_two_step = np.sum([k_weathering_two_step[istep] * (del_graniteo[istep] - delt_weath),
                                   k_hi_lo_change[istep] * (del_basalto - Delt_hiT_twostep[istep])
                                   ])
            top_early = np.sum([k_weathering_early[istep] * (del_graniteo[istep] - delt_weath),
                                k_hi_lo_change[istep] * (del_basalto - delt_hi_and_l)
                                ])
            top_late = np.sum([k_weathering_late[istep] * (del_graniteo[istep] - delt_weath),
                              k_hi_lo_change[istep] * (del_basalto - Delt_hiT_change_late[istep])])

            k_sum[istep] = np.sum(
                [k_weath[istep], k_hi_lo_change[istep]])
            k_sum_early[istep] = np.sum(
                [k_weathering_early[istep], k_hi_lo_change[istep]])
            k_sum_late[istep] = np.sum(
                [k_weathering_late[istep], k_hi_lo_change[istep]])
            k_sum_twostep[istep] = np.sum(
                [k_weathering_two_step[istep], k_hi_lo_change[istep]])

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
        dW_twostep = np.add(np.subtract(Wo, steady_interp_twostep) * np.exp(-np.multiply(time_new, k_sum_interp_twostep)),
                            steady_interp_twostep)

        return dW_early, dW_middle, dW_late, dW_twostep, time_new

def run_Muelenbach(parms):
    t = 4.4  # time in Gyr
    Wo = 7  # original seawater d18O
    num_steps = 100  # num of initial model steps
    time = np.linspace(0, 4.5, num=num_steps)  # sample every 250 myr
    weath_time_on_twostep = 4.5 - 2.4  # in Ga
    weath_time_on = 4.5 - 1.9
    weath_time_early = 4.5 - 4.43
    weath_time_late = 4.5 - 0.9

    Delt_hiT_mid = 1.5

    k_weath, k_hiT, Delt_weath, Delt_hiT = parms[0], parms[1], parms[2], parms[3]
    k_growth, k_loT, k_W_recycling, = parms[4], parms[5], parms[6]
    Delt_growth, Delt_lowT, Delt_water_recycling = parms[7], parms[8], parms[9]

    # calculate steady state in 250 myr increments
    del_graniteo = np.linspace(7.8, 7.8, num=num_steps)

    del_basalto = 5.5
    del_WR = 7
    bb = 0.4
    bb2 = 0.1

    Delt_hiT_change = (Delt_hiT - hiT_mid) + (delt_hi_and_l - hiT_mid) * .5 * (
            1 + np.tanh((np.subtract(time, weath_time_on) / bb))) - 1
    Delt_hiT_change_late = (Delt_hiT - hiT_mid) + (delt_hi_and_l - hiT_mid) * .5 * (
            1 + np.tanh((np.subtract(time, weath_time_late) / bb))) - 1
    Delt_hiT_twostep = (Delt_hiT - hiT_mid) + (delt_hi_and_l - hiT_mid) * .5 * (  # TODO .5 here
            1 + np.tanh((np.subtract(time, weath_time_on_twostep) / bb2))) - 1

    k_weathering_change = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_on) / bb)))
    k_weathering_late = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_late) / bb)))
    k_weathering_early = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_early) / bb)))
    two_step_time = 4.5 - 2
    # k_growth_twostep = 0.5 * k_growth * (1 + np.tanh((np.subtract(time, weath_time_on_twostep) / (bb2))))

    weath_time_mid = 4.5 - 1.65
    k_weathering_mid = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_mid) / bb)))
    k_weathering_two_step = 0.5 * k_weath * (1 + np.tanh((np.subtract(time, weath_time_mid) / bb2)))

    k_hi_lo_change = (k_hi_lo_t) * np.ones(time.size)
    k_weath = (k_weath) * np.ones(time.size)

    del_steady_change = np.zeros(time.size)
    del_steady_early = np.zeros(time.size)
    del_steady_late = np.zeros(time.size)
    del_steady_two_step = np.zeros(time.size)

    k_sum = np.zeros(time.size)
    k_sum_early = np.zeros(time.size)
    k_sum_late = np.zeros(time.size)
    k_sum_twostep = np.zeros(time.size)

    for istep in range(0, time.size):
        top = np.sum([k_weath[istep] * (del_graniteo[istep] - delt_weath),
                      k_hi_lo_change[istep] * (del_basalto - Delt_hiT_change[istep])
                      ])
        top_two_step = np.sum([k_weathering_two_step[istep] * (del_graniteo[istep] - delt_weath),
                               k_hi_lo_change[istep] * (del_basalto - Delt_hiT_twostep[istep])
                               ])
        top_early = np.sum([k_weathering_early[istep] * (del_graniteo[istep] - delt_weath),
                            k_hi_lo_change[istep] * (del_basalto - delt_hi_and_l)
                            ])
        top_late = np.sum([k_weathering_late[istep] * (del_graniteo[istep] - delt_weath),
                           k_hi_lo_change[istep] * (del_basalto - Delt_hiT_change_late[istep])])

        k_sum[istep] = np.sum(
            [k_weath[istep], k_hi_lo_change[istep]])
        k_sum_early[istep] = np.sum(
            [k_weathering_early[istep], k_hi_lo_change[istep]])
        k_sum_late[istep] = np.sum(
            [k_weathering_late[istep], k_hi_lo_change[istep]])
        k_sum_twostep[istep] = np.sum(
            [k_weathering_two_step[istep], k_hi_lo_change[istep]])

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
    dW_twostep = np.add(np.subtract(Wo, steady_interp_twostep) * np.exp(-np.multiply(time_new, k_sum_interp_twostep)),
                        steady_interp_twostep)

    return dW_early, dW_middle, dW_late, dW_twostep, time_new
