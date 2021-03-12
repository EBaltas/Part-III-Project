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

def SourcePol(phicalc, V1_x, V1_y):
    
    return phicalc + np.rad2deg(np.arctan2(V1_x, V1_y)) + 180


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
    Vector2 = np.zeros((phiNumSamples, dttNumSamples, 2))
    Vector1 = np.zeros((phiNumSamples, dttNumSamples, 2))
    
    selectionWin = np.array((int(starttime*sps), int(endtime*sps)))
    
    for i in range(phiNumSamples):
        phi = phiSamples[i]
        
        R1 = RotMat2D(phi)
        
        FS = np.dot(R1, EN)
        
        for j in range(dttNumSamples):
            dtt = dttSamples[j]
            delayWin = selectionWin + dtt
            print("SelectWin=",selectionWin," delayWin=",delayWin)
        
            FS_lag = np.array((
                FS[0][delayWin[0]:delayWin[1]],         #this is lagged (consistent with MFAST)
                FS[1][selectionWin[0]:selectionWin[1]]  #this is fast (again consistent with MFAST)
            ))
        
            CovMat = np.cov(FS_lag[0], FS_lag[1])
        
            eigenval, eigenvect = np.linalg.eig(CovMat)
            Lambda2[i, j] = eigenval.min()
            Lambda1[i, j] = eigenval.max()
            max_index = np.argmax(eigenval)
            Vector1[i, j] = eigenvect[:, max_index]
            print(Lambda2[i, j])
        
    EVindex = np.where(Lambda2 == Lambda2.min())
    V1 = Vector1[EVindex][0]    
    
    print(V1)
    
    if len(EVindex) > 2:
        raise IndexError("More than 1 solution found")
    
    phiCalc = phiSamples[EVindex[0]][0]
    dttCalc = dttSamples[EVindex[1]][0]
    
    source_pol = SourcePol(phiCalc, V1[0], V1[1])
    
    print("source polarisation =", source_pol)
    

    return phiCalc, dttCalc/sps
    
    
    
        
        
                                                     