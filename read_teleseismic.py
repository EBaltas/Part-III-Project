#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:55:47 2021

@author: elisavetbaltas
"""

import obspy as obs
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from minimum_energy import SAndC_ME
from eigenvalue2 import SAndC_eigen

#Configuring client object and specifying event metadata

client = Client("NCEDC")

event_start = UTCDateTime("2011-07-19T19:35:43.0")
event_end = event_start +3600

model = TauPyModel(model="iasp91")
arrival = model.get_travel_times(source_depth_in_km = 19.5,
                                 distance_in_degree = 101.42,
                                 phase_list = ["SKS"])[0].time

print("arrival =", arrival)

#Retrieving data and pre-processing

st = client.get_waveforms("BK", "BKS", "*", "BH*", event_start, event_end)

st_filt = st.copy()

st_filt.detrend("demean")
st_filt.detrend("linear")
st_filt.filter("bandpass", freqmin=0.02, freqmax=1/6)

starttime = arrival-50
endtime = arrival+50

#phi, dtt = SAndC_ME(st_filt, 349.4, starttime, endtime, timeDelay=2.0)

phi, dtt, _, _, _, _ = SAndC_eigen(st_filt, starttime, endtime, timeDelay=2.0)

print("phi =", phi, " dtt =", dtt)