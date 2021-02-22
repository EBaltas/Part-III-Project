# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 00:27:56 2021

@author: elisavetbaltas
"""

from retrieve import Archive
#import util as util
import util
import pandas as pd
from obspy import UTCDateTime
from obspy.core import read
import matplotlib

#from quakemigrate.io import Archive, read_lut, read_stations

def read_stations(station_file, **kwargs):
    """
    Reads station information from file.
    Parameters
    ----------
    station_file : str
        Path to station file.
        File format (header line is REQUIRED, case sensitive, any order):
            Latitude, Longitude, Elevation (units matching LUT grid projection;
            either metres or kilometres; positive upwards), Name
    kwargs : dict
        Passthrough for `pandas.read_csv` kwargs.
    Returns
    -------
    stn_data : `pandas.DataFrame` object
        Columns: "Latitude", "Longitude", "Elevation", "Name"
    Raises
    ------
    StationFileHeaderException
        Raised if the input file is missing required entries in the header.
    """

    stn_data = pd.read_csv(station_file, **kwargs)

    if ("Latitude" or "Longitude" or "Elevation" or "Name") \
       not in stn_data.columns:
        raise util.StationFileHeaderException

    stn_data["Elevation"] = stn_data["Elevation"].apply(lambda x: -1*x)

    # Ensure station names are strings
    stn_data = stn_data.astype({"Name": "str"})

    return stn_data


# --- i/o paths ---
station_file = "./iceland_stations.txt"
data_in = "./mSEED"


# --- Set time period over which to run detect ---
starttime = "2014-06-29T18:41:55.0"
endtime = "2014-06-29T18:42:20.0"

# --- Read in station file ---
stations = read_stations(station_file)
print(stations)

# --- Create new Archive and set path structure ---
archive = Archive(archive_path=data_in, stations=stations,
                  archive_format="YEAR/JD/*_STATION_*")
print(archive)


UTCStarttime = UTCDateTime(starttime)
UTCEndtime = UTCDateTime(endtime)

data1 = archive.read_waveform_data(UTCStarttime, UTCEndtime, pre_pad=0., post_pad=0.)

print(data1)

print(archive.archive_path)

filename = "./" + str(archive.archive_path) + "/2014/180/2014180_180000_SKG08_E2.m"

print(filename)

st = read(filename)

tr = st[0]

print(tr.data)

print(len(tr.data))

st.plot()




