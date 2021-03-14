#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:52:19 2021

@author: elisavetbaltas
"""

import numpy as np
import obspy as obs
from obspy.signal.util import next_pow_2
import itertools
from scipy.stats import f
import matplotlib.pyplot as plt

def RotMat2D(phi):
    
    phi = np.deg2rad(phi)
    return np.array([[np.cos(phi), -np.sin(phi)], #changed matrix so that it is consistent with MFAST
                     [np.sin(phi), np.cos(phi)]])

def SourcePol(phicalc, V1_x, V1_y):
    
    return phicalc + np.rad2deg(np.arctan2(V1_x, V1_y)) + 180

def CR95(L2, NDF, C=2, a=0.05):
    
    return L2*(1+((C/(NDF-C))*f.isf(a, C, NDF-C)))


def SAndC_eigen(st_filt, starttime, endtime, timeDelay=0.25):
    
    EN = np.array((
        st_filt.select(component="E")[0].data,
        st_filt.select(component="N")[0].data))
    
    sps = st_filt[0].stats.sampling_rate
    timeDelay *= sps
    step = 1
    
    phiSamples = np.arange(-90, 90+step, step, dtype=int)
    dttSamples = np.arange(0, timeDelay+step, step, dtype=int)
    
    phiNumSamples = phiSamples.size
    dttNumSamples = dttSamples.size
    
    Lambda2 = np.zeros((phiNumSamples, dttNumSamples))
    Lambda1 = np.zeros((phiNumSamples, dttNumSamples))
    Vector2 = np.zeros((phiNumSamples, dttNumSamples, 2))
    Vector1 = np.zeros((phiNumSamples, dttNumSamples, 2))
    
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

    
    #Calculate rotatation matrix and delay window using phiCalc and dtt
    R2 = RotMat2D(phiCalc)
    delayWin2 = selectionWin + int(dttCalc)
    
    
    #Rotate orginal EN using phiCalc
    FS2 = np.dot(R2, EN)
    
    #Apply dtt
    FS2_Lag = np.array((
        FS2[0][delayWin2[0]:delayWin2[1]],
        FS2[1][selectionWin[0]:selectionWin[1]]
        ))
   
    #Rotate Fast-slow by source_pol - phiCalc
    RSP = RotMat2D(source_pol - phiCalc)
    
    FS_SP = np.dot(RSP, FS2_Lag)
    
    #Calculating number of degrees of freedom
    npoints = FS_SP[0].size
    N2 = next_pow_2(npoints)
    
    spec_mag = abs(np.fft.fft(FS_SP[0], N2))
    
    F2 = (spec_mag**2).sum()-((spec_mag[0]**2)-(spec_mag[-1]**2))/2
    F4 = (4/3)*(spec_mag**4).sum()-((spec_mag[0]**4)-(spec_mag[-1]**4))/3
    
    NDF = int(round((2*(F2**2)/F4)-1))
    
    print("Number of degrees of freedom =", NDF)
    
    L2 = Lambda2.copy()
    
    conf95 = CR95(L2.min(), NDF)
    
    L2_norm = L2/conf95
    
    cs = plt.contour(dttSamples/100, phiSamples, L2_norm)

    return phiCalc, dttCalc/sps, FS2[0], FS2[1], FS2_Lag[0], FS2_Lag[1]
    
    
    
        
        
                                                     