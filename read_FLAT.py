#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 21:20:57 2021

@author: elisavetbaltas
"""


import obspy as ob
import matplotlib.pyplot as plt
from eigenvalue2 import SAndC_eigen

st = ob.read("/Users/elisavetbaltas/Documents/Part_III_Physics/Project/FLAT/2013101_000000_FLAT_Z2.m")
st += ob.read("/Users/elisavetbaltas/Documents/Part_III_Physics/Project/FLAT/2013101_000000_FLAT_N2.m")
st += ob.read("/Users/elisavetbaltas/Documents/Part_III_Physics/Project/FLAT/2013101_000000_FLAT_E2.m")

st_filt = st.copy()

st_filt.detrend("demean")
st_filt.detrend("linear")
st_filt.filter("bandpass", freqmin=1, freqmax=8)

#using S pick time for FLAT event

s_pick = ob.UTCDateTime("2013-04-11T11:50:25.86")
starttime = s_pick - 1
endtime =s_pick + 1



tr_E = st_filt[0]
tr_N = st_filt[1]
tr_Z = st_filt[2]

tr_E.plot()
tr_N.plot()
tr_Z.plot()

#st_filt.plot()

#st.trim(starttime, endtime)
#st.plot()

starttime = starttime.timestamp - ob.UTCDateTime("2013-04-11T00:00:00.00").timestamp
endtime = endtime.timestamp- ob.UTCDateTime("2013-04-11T00:00:00.00").timestamp
#print(starttime, endtime)

phi, dtt, _, _, _, _, _, _ = SAndC_eigen(st_filt, starttime, endtime, timeDelay=0.4)

print("phi=",phi, " dt=",dtt)

#st_filt.trim(starttime, endtime)

#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#ax.plot(tr.times("matplotlib"), tr.data, "b-")
#ax.xaxis_date()
#fig.autofmt_xdate()
#plt.savefig("plot.png", bbox_inches="tight")

#starttime_event = ob.UTCDateTime("2013-04-11T11:50:21.0")
#endtime_event = starttime_event + 3600

#print(st[0].stats)

#2013101_000000_FLAT_E2.m 
#2013101_000000_FLAT_N2.m
#2013101_000000_FLAT_Z2.m