#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 16:42:47 2021

@author: elisavetbaltas
"""
"""

Module for processing waveform files stored in a data archive.
:copyright:
    2020 - 2021, QuakeMigrate developers.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)
    
"""

from itertools import chain
import logging
import pathlib

from obspy import read, Stream, UTCDateTime

import util

class Archive:
    def __init__(self, archive_path, stations, archive_format=None, **kwargs):
        """Instantiate the Archive object."""

        self.archive_path = pathlib.Path(archive_path)
        self.stations = stations["Name"]
        if archive_format:
            channels = kwargs.get("channels", "*")
            self.path_structure(archive_format, channels)
        else:
            self.format = kwargs.get("format")
            
        self.read_all_stations = kwargs.get("read_all_stations", False)
    
    def __str__(self, response_only=False):
        """
        Returns a short summary string of the Archive object.
        Parameters
        ----------
        response_only : bool, optional
            Whether to just output the a string describing the instrument
            response parameters.
        Returns
        -------
        out : str
            Summary string.
        """

        out = ("QuakeMigrate Archive object"
               f"\n\tArchive path\t:\t{self.archive_path}"
               f"\n\tPath structure\t:\t{self.format}")
        out += "\n\tStations:"
        for station in self.stations:
            out += f"\n\t\t{station}"

        return out
    
    def path_structure(self, archive_format="YEAR/JD/STATION", channels="*"):
        """
        Define the directory structure and file naming format of the data
        archive.
        Parameters
        ----------
        archive_format : str, optional
            Directory structure and file naming format of the data archive.
            This may be the name of a generic archive format (e.g. SeisComp3),
            or one of a selection of additional formats built into
            QuakeMigrate.
        channels : str, optional
            Channel codes to include. E.g. channels="[B,H]H*". (Default "*")
        Raises
        ------
        ArchivePathStructureError
            If the `archive_format` specified by the user is not a valid option.
        """

        if archive_format == "SeisComp3":
            self.format = ("{year}/*/{station}/"+channels+"/*.{station}.*.*.D."
                           "{year}.{jday:03d}")
        elif archive_format == "YEAR/JD/*_STATION_*":
            self.format = "{year}/{jday:03d}/*_{station}_*"
        elif archive_format == "YEAR/JD/STATION":
            self.format = "{year}/{jday:03d}/{station}*"
        elif archive_format == "STATION.YEAR.JULIANDAY":
            self.format = "*{station}.*.{year}.{jday:03d}"
        elif archive_format == "/STATION/STATION.YearMonthDay":
            self.format = "{station}/{station}.{year}{month:02d}{day:02d}"
        elif archive_format == "YEAR_JD/STATION*":
            self.format = "{year}_{jday:03d}/{station}*"
        elif archive_format == "YEAR_JD/STATION_*":
            self.format = "{year}_{jday:03d}/{station}_*"
        else:
            raise util.ArchivePathStructureError(archive_format)
        
    def read_waveform_data(self, starttime, endtime,
                           pre_pad=0., post_pad=0.):
        """
        Read in waveform data from the archive between two times.
        Supports all formats currently supported by ObsPy, including: "MSEED",
        "SAC", "SEGY", "GSE2" .
        Optionally, read data with some pre- and post-pad, and for all stations
        in the archive - this will be stored in `data.raw_waveforms`, while
        `data.waveforms` will contain only data for selected stations between
        `starttime` and `endtime`.
        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object
            Timestamp from which to read waveform data.
        endtime : `obspy.UTCDateTime` object
            Timestamp up to which to read waveform data.
        pre_pad : float, optional
            Additional pre pad of data to read. Defaults to 0.
        post_pad : float, optional
            Additional post pad of data to read. Defaults to 0.
        Returns
        -------
        data : :class:`~quakemigrate.io.data.WaveformData` object
            Object containing the waveform data read from the archive that
            satisfies the query.
        """

        # Ensure pre-pad and post-pad are not negative.
        pre_pad = max(0., pre_pad)
        post_pad = max(0., post_pad)

        data = WaveformData(starttime=starttime, endtime=endtime,
                            stations=self.stations, pre_pad=pre_pad,
                            post_pad=post_pad)
        
        
        files =self._load_from_path(starttime - pre_pad, endtime + post_pad)
        
        st = Stream()
        try:
            first = next(files)
            files = chain([first], files)
            for file in files:
                file = str(file)
                try:
                    read_start = starttime - pre_pad
                    read_end = endtime + post_pad
                    st += read(file, starttime=read_start, endtime=read_end,
                               nearest_sample=True)
                except TypeError:
                    logging.info(f"File not compatible with ObsPy - {file}")
                    continue

            # Merge all traces with contiguous data, or overlapping data which
            # exactly matches (== st._cleanup(); i.e. no clobber)
            st.merge(method=-1)

            # Make copy of raw waveforms to output if requested
            data.raw_waveforms = st.copy()

            # Ensure data is timestamped "on-sample" (i.e. an integer number
            # of samples after midnight). Otherwise the data will be implicitly
            # shifted when it is used to calculate the onset function /
            # migrated.
            
            if self.read_all_stations:
               # Re-populate st with only stations in station file
                st_selected = Stream()
                for station in self.stations:
                    st_selected += st.select(station=station)
                st = st_selected.copy()
                del st_selected
            
            if pre_pad != 0. or post_pad != 0.:
                # Trim data between start and end time
                for tr in st:
                    tr.trim(starttime=starttime, endtime=endtime,
                            nearest_sample=True)
                    if not bool(tr):
                        st.remove(tr)
            
            # Test if the stream is completely empty
            # (see __nonzero__ for `obspy.Stream` object)
            if not bool(st):
                raise util.DataGapException

            # Add cleaned stream to `waveforms`
            data.waveforms = st

        except StopIteration:
            raise util.ArchiveEmptyException

        return data
    
    def _load_from_path(self, starttime, endtime):
        """
        Retrieves available files between two times.
        Parameters
        ----------
        starttime : `obspy.UTCDateTime` object
            Timestamp from which to read waveform data.
        endtime : `obspy.UTCDateTime` object
            Timestamp up to which to read waveform data.
        Returns
        -------
        files : generator
            Iterator object of available waveform data files.
        Raises
        ------
        ArchiveFormatException
            If the Archive.format attribute has not been set.
        """

        if self.format is None:
            raise util.ArchiveFormatException

        # Loop through time period by day adding files to list
        # NOTE! This assumes the archive structure is split into days.
        files = []
        loadstart = UTCDateTime(starttime)
        while loadstart < endtime:
            temp_format = self.format.format(year=loadstart.year,
                                             month=loadstart.month,
                                             day=loadstart.day,
                                             jday=loadstart.julday,
                                             station="{station}",
                                             dtime=loadstart)
            if self.read_all_stations is True:
                file_format = temp_format.format(station="*")
                files = chain(files, self.archive_path.glob(file_format))
            else:
                for station in self.stations:
                    file_format = temp_format.format(station=station)
                    files = chain(files, self.archive_path.glob(file_format))
            loadstart = UTCDateTime(loadstart.date) + 86400

        return files

class WaveformData:
    def __init__(self, starttime, endtime, stations=None, read_all_stations=False,
                 pre_pad=0., post_pad=0.):
        """Instantiate the WaveformData object."""

        self.starttime = starttime
        self.endtime = endtime
        self.stations = stations
        
        self.read_all_stations = read_all_stations
        self.pre_pad = pre_pad
        self.post_pad = post_pad

        self.raw_waveforms = None
        self.waveforms = Stream()
        self.wa_waveforms = None
        self.real_waveforms = None
    
    def check_availability(self, st, all_channels=False, n_channels=None,
                           allow_gaps=False, full_timespan=True,
                           check_sampling_rate=False, sampling_rate=None,
                           check_start_end_times=False):
        
        availability = {}
        available = 0
        timespan = self.endtime - self.starttime
        
        # Check if any channels in stream
        if bool(st):
            # Loop through channels with unique SEED id's
            for tr_id in sorted(set([tr.id for tr in st])):
                st_id = st.select(id=tr_id)
                availability[tr_id] = 0

                # Check it's not flatlined
                if any(tr.data.max() == tr.data.min() for tr in st_id):
                    continue
                # Check for overlaps
                overlaps = st_id.get_gaps(max_gap=-0.000001)
                if len(overlaps) != 0:
                    continue
                # Check for gaps (if requested)
                if not allow_gaps:
                    gaps = st_id.get_gaps()  # Overlaps already dealt with
                    if len(gaps) != 0:
                        continue
                # Check sampling rate
                if check_sampling_rate:
                    if not sampling_rate:
                        raise TypeError("Please specify sampling_rate if you "
                                        "wish to check all channels are at the"
                                        " correct sampling rate.")
                    if any(tr.stats.sampling_rate != sampling_rate \
                        for tr in st_id):
                        continue
                # Check data covers full timespan (if requested) - this
                # strictly checks the *timespan*, so uses the trace sampling
                # rate as provided. To check that as well, use
                # `check_sampling_rate=True` and specify a sampling rate.
                if full_timespan:
                    n_samples = timespan * st_id[0].stats.sampling_rate + 1
                    if len(st_id) > 1:
                        continue
                    elif st_id[0].stats.npts < n_samples:
                        continue
                # Check start and end times of trace are exactly correct
                if check_start_end_times:
                    if len(st_id) > 1:
                        continue
                    elif st_id[0].stats.starttime != self.starttime or \
                        st_id[0].stats.endtime != self.endtime:
                        continue

                # If passed all tests, set availability to 1
                availability[tr_id] = 1

            # Return availability based on "all_channels" setting
            if all(ava == 1 for ava in availability.values()):
                if all_channels:
                    # If all_channels requested, must also check that the
                    # expected number of channels are present
                    if not n_channels:
                        raise TypeError("Please specify n_channels if you wish"
                                        " to check all channels meet the "
                                        "availability criteria.")
                    elif len(availability) == n_channels:
                        available = 1
                else:
                    available = 1
            elif not all_channels \
                and any(ava == 1 for ava in availability.values()):
                available = 1

        return available, availability
    
    def get_real_waveform(self, tr, velocity=True):
        
        # Copy the Trace before operating on it
        tr = tr.copy()
        tr.detrend('linear')
        
        try:
            self.real_waveforms.append(tr.copy())
        except AttributeError:
            self.real_waveforms = Stream()
            self.real_waveforms.append(tr.copy())

        return tr
    
    def get_wa_waveform(self, tr, velocity=False):
        """
        Calculate simulated Wood Anderson displacement waveform for a Trace.
        Parameters
        ----------
        tr : `obspy.Trace` object
            Trace containing the waveform to be corrected to a Wood-Anderson
            response
        velocity : bool, optional
            Output velocity waveform, instead of displacement. Default: False.
            NOTE: all attenuation functions provided within the QM local_mags
            module are calculated for displacement seismograms.
        Returns
        -------
        tr : `obspy.Trace` object
            Trace corrected to Wood-Anderson response.
        """

        # Copy the Trace before operating on it
        tr = tr.copy()
        tr.detrend('linear')

        # Remove instrument response
        tr = self.get_real_waveform(tr, velocity)

        # Simulate Wood-Anderson response
        tr.simulate(paz_simulate=util.wa_response(obspy_def=True),
                    pre_filt=self.pre_filt,
                    water_level=self.water_level,
                    taper=True,
                    sacsim=True,  # To replicate remove_response()
                    pitsasim=False)  # To replicate remove_response()

        try:
            self.wa_waveforms.append(tr.copy())
        except AttributeError:
            self.wa_waveforms = Stream()
            self.wa_waveforms.append(tr.copy())

        return tr