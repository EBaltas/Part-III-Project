#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:52:19 2021

@author: elisavetbaltas
"""

import numpy as np
import obspy as obs
import itertools

def RotMat2D(phi):
    
    phi = np.deg2rad(phi)
    return np.array([[np.cos(phi), -np.sin(phi)], #changed matrix so that it is consistent with MFAST
                     [np.sin(phi), np.cos(phi)]])


def SAndC_eigen(st_filt, starttime, endtime, timeDelay=0.25):
    
    EN = np.array((
        st_filt.select(component="E")[0].data,
        st_filt.select(component="N")[0].data))
    
    sps = st_filt[0].stats.sampling_rate
    timeDelay *= sps
    step = 1
    
    phiSamples = np.arange(-90, 90+step, step, dtype=int)
    dttSamples = np.arange(0, timeDelay+step, step, dtype=int)
    
    #phiSamples = np.arange(15, 24+step, step, dtype=int)
    
    phiNumSamples = phiSamples.size
    dttNumSamples = dttSamples.size
    
    Lambda2 = np.zeros((phiNumSamples, dttNumSamples))
    Lambda1 = np.zeros((phiNumSamples, dttNumSamples))
    EigVect = np.zeros((phiNumSamples, dttNumSamples))
    
    selectionWin = np.array((int(starttime*sps), int(endtime*sps)))
    
    for i in range(phiNumSamples):
        phi = phiSamples[i]
        
        R = RotMat2D(phi)
        
        FS = np.dot(R, EN)
        
        for j in range(dttNumSamples):
            dtt = dttSamples[j]
            delayWin = selectionWin + dtt
            print("SelectWin=",selectionWin," delayWin=",delayWin)
        
            FS_lag = np.array((
                FS[0][delayWin[0]:delayWin[1]],         #this is lagged (consistent with MFAST)
                FS[1][selectionWin[0]:selectionWin[1]]  #this is fast (again consistent with MFAST)
            ))
            print(FS_lag.shape)
            print(R.T.shape)
        
            CovMat = np.cov(FS_lag[0], FS_lag[1])
        
            eigenval, eigenvect = np.linalg.eig(CovMat)
            Lambda2[i, j] = eigenval.min()
            print(Lambda2[i, j])
        
    EVindex = np.where(Lambda2 == Lambda2.min())
    
    if len(EVindex) > 2:
        raise IndexError("More than 1 solution found")
    
    phiCalc = phiSamples[EVindex[0]][0]
    dttCalc = dttSamples[EVindex[1]][0]
    
    return phiCalc, dttCalc/sps
    
    
    
        
        
                                                     