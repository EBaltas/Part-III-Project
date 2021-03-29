#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 21:20:57 2021

@author: elisavetbaltas
"""


import obspy as ob
import matplotlib.pyplot as plt
from eigenvalue3 import SAndC_eigen

import numpy as np
import pandas as pd


st = ob.read("./FLAT/2013101_000000_FLAT_Z2.m")
st += ob.read("./FLAT/2013101_000000_FLAT_N2.m")
st += ob.read("./FLAT/2013101_000000_FLAT_E2.m")

#st = ob.read("/Users/elisavetbaltas/Documents/Part_III_Physics/Project/FLAT/2013101_000000_FLAT_Z2.m")
#st += ob.read("/Users/elisavetbaltas/Documents/Part_III_Physics/Project/FLAT/2013101_000000_FLAT_N2.m")
#st += ob.read("/Users/elisavetbaltas/Documents/Part_III_Physics/Project/FLAT/2013101_000000_FLAT_E2.m")

# Copy Stream before filtering and trimming
st_filt = st.copy()

# Plot before detrending and filtering
st_filt[0].plot()
st_filt[1].plot()
st_filt[2].plot()

# Detrend and filter
st_filt.detrend("demean")
st_filt.detrend("linear")
st_filt.filter("bandpass", freqmin=1, freqmax=8)

# Use S pick time to estimate starttime and endtime of time series to be used
# The 24 hour time-series is trimmed between starttime and endtime
timeOffsetStart = 10.
timeOffsetEnd   = 20.
s_pick = ob.UTCDateTime("2013-04-11T11:50:25.85")
starttime = s_pick - timeOffsetStart
endtime   = s_pick + timeOffsetEnd

# Define filtered traces for each direction
tr_Z = st_filt[0]
tr_N = st_filt[1]
tr_E = st_filt[2]

# Plot after detrending and filtering
tr_E.plot()
tr_N.plot()
tr_Z.plot()

####### Trimming and plotting #########
tr_Z.trim(starttime, endtime)
tr_N.trim(starttime, endtime)
tr_E.trim(starttime, endtime)

tr_E.plot()
tr_N.plot()
tr_Z.plot()
print("tr_E=",tr_E.data.shape)

# Create a new Stream with trimmed traces to be used for analysis
st_trimmed = ob.Stream(traces=[tr_Z, tr_N, tr_E])

print(st_trimmed)
#######################################

tr_N.plot(color='red', number_of_ticks=7,
                   tick_rotation=5, tick_format='%I:%M %p',
                   starttime=s_pick-2, endtime=s_pick + 15 )
tr_E.plot(color='blue', number_of_ticks=7,
                   tick_rotation=5, tick_format='%I:%M %p',
                   starttime=s_pick-2, endtime=s_pick + 15)

# Define the S-wave windows (winBegin, winEnd) and timeDelay
sps = st_trimmed[0].stats.sampling_rate
winBegin  = s_pick - 0.9
print("winBegin=",winBegin)
winBegin  = winBegin.timestamp - ob.UTCDateTime("2013-04-11T00:00:00.00").timestamp
print("winBegin.timestamp=",winBegin)
winBegIndx = int(winBegin * sps)
print("winBegIndx=",winBegIndx)

winEnd    = s_pick + 1.0
print("winEnd=",winEnd)
winEnd    = winEnd.timestamp - ob.UTCDateTime("2013-04-11T00:00:00.00").timestamp
print("winEnd.timestamp=",winEnd)
winEndIndx = int(winEnd * sps)
print("winEndIndx=",winEndIndx)

# Convert timeDelay from secs to samples
timeDelay = 0.6 * sps

print("Starttime=",starttime," endtime=",endtime)
print("Starttime=",starttime-ob.UTCDateTime("2013-04-11T00:00:00.00")," endtime=",endtime)
starttime = starttime.timestamp - ob.UTCDateTime("2013-04-11T00:00:00.00").timestamp
endtime   = endtime.timestamp - ob.UTCDateTime("2013-04-11T00:00:00.00").timestamp

print("starttime=",starttime," endtime=",endtime)
starttimeIndx = int(starttime*sps)
endtimeIndx   = int(endtime*sps)
print(st[1].data[starttimeIndx])
print(st[1].data[starttimeIndx])
print(st[1].data[endtimeIndx])
print("starttimeIndex=",int(starttime*sps)," endtimeIndex=",int(endtime*sps))

print(tr_N.data[0], tr_N.data[3000])
print(tr_E.data[0], tr_E.data[3000])
print(tr_Z.data[0], tr_Z.data[3000])

x0 = tr_E.data
y0 = tr_N.data
winBegIndx -= starttimeIndx
winEndIndx -= starttimeIndx
print("winBegIndx=",winBegIndx)
print("winEndIndx=",winEndIndx)


phi, dtt, _, _, _, _, _, _ = SAndC_eigen(x0, y0, winBegIndx, winEndIndx, timeDelay, sps)

print("phi=",phi, " dt=",dtt)

