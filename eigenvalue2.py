#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:07:50 2021

@author: elisavetbaltas
"""


import numpy as np
import pandas as pd
import obspy as obs
from obspy.signal.util import next_pow_2
import itertools
from scipy.stats import f
from scipy.fft import fft
import matplotlib.pyplot as plt
import csv

def RotMat2D(phi):
    phi = np.deg2rad(phi)
    return np.array([[np.cos(phi), -np.sin(phi)], #changed matrix so that it is consistent with MFAST
                     [np.sin(phi), np.cos(phi)]])

def SourcePol(phicalc, V1_x, V1_y):
    spol = phicalc + np.rad2deg(np.arctan2(V1_x, V1_y))
    if spol > 90:
        spol -= 180
    elif spol < -90:
        spol += 180
    else:
        spol = spol
    return spol

def CR95(L2, NDF, C=2, a=0.05):
    return L2*(1+((C/(NDF-C))*f.ppf(1-a, C, NDF-C)))


def SAndC_eigen(x, y, iwbegix, iwendix, timeDelay, sps):
    
    EN = np.array((
         x,
         y))
    
    delta = 1./sps
    step = 1
    
    phiSamples = np.arange(-90, 90+step, step, dtype=int)
    dttSamples = np.arange(0, timeDelay + step, step, dtype=int)
    
    phiNumSamples = phiSamples.size
    dttNumSamples = dttSamples.size
    print("phiNumSamples=",phiNumSamples," dttNumSamples=",dttNumSamples)
    
    Lambda2 = np.zeros((phiNumSamples, dttNumSamples))
    Lambda1 = np.zeros((phiNumSamples, dttNumSamples))
    Vector2 = np.zeros((phiNumSamples, dttNumSamples, 2))
    Vector1 = np.zeros((phiNumSamples, dttNumSamples, 2))
    
    #selectionWin = np.array((1471,1593))
    selectionWin = np.array((iwbegix, iwendix))
  
    
    print("selectionWin=",selectionWin)
    
    for i in range(phiNumSamples):
        phi = phiSamples[i]
        
        R = RotMat2D(phi)
        
        FS = np.dot(R, EN)
        
        for j in range(dttNumSamples):
            dtt = dttSamples[j]
            delayWin = selectionWin + dtt
        
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
        
    EVindex = np.where(Lambda2 == Lambda2.min())
    print("Lambda2_min=",Lambda2.min())
    V1 = Vector1[EVindex][0]    
    
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
    
    #Trimming FS2 to selection window for plotting
    FS2_trim = np.array((
        FS2[0][selectionWin[0]:selectionWin[1]],
        FS2[1][selectionWin[0]:selectionWin[1]]
        ))
    
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
    print("npoints=",npoints," N2=",N2)
    print("norig =", selectionWin[1] - selectionWin[0])
    spec_mag = abs(np.fft.fft(FS_SP[0], N2))
    print("npoints2 =",int(len(FS_SP[0])))
    
    F2 = (spec_mag**2).sum()-((spec_mag[0]**2)+(spec_mag[-1]**2))/2
    F4 = (4/3)*(spec_mag**4).sum()-((spec_mag[0]**4)+(spec_mag[-1]**4))
    print("F2=",F2," F4=",F4)
    
    NDF = int(round(2*(2*(F2**2)/F4)-2))
    
    print("Number of degrees of freedom =", NDF)
    
    L2 = Lambda2.copy()
    
    conf95 = CR95(L2.min(), NDF)
    print("conf95=",conf95)
    
    
    L2_norm = L2/conf95
    
    #contour = plt.contour(dttSamples/sps, phiSamples, L2_norm)
    
    fig, ax = plt.subplots()
    CS = ax.contour(dttSamples/sps, phiSamples, L2_norm, level=2)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Normalised Lambda_2')
    
######### MFAST Error loops
    j_min = dttNumSamples
    j_max = 0
    line_test = []
    
    for i in range(phiNumSamples):
        for j in range(dttNumSamples):
            if (L2_norm[i, j] <= 1.0):
                j_min = min(j_min, j)
                j_max = max(j_max, j)
    
    j_range = j_max - j_min
    print("j_min =", j_min, " j_max =", j_max, " j_range =", j_range)
    
    i_range = 0 
    line = np.zeros(phiNumSamples + 1, dtype=int)
    
    for j in range (dttNumSamples):
        for i in range(phiNumSamples):
            if (L2_norm[i, j] <= 1.0):
                line[i] = 1
    
    i_range_min = 0
    
    for i in range (phiNumSamples):
        i_range_min = i_range_min + line[i]
    print("i_range_min=",i_range_min)
    
    i_range_max = phiNumSamples
    
    label_1 = False
    label_11 = False

    for i in range(i_range_min, i_range_max):
        for i_start in range(phiNumSamples):
            line_test = np.zeros(phiNumSamples + 1, dtype=int)
            for k in range(i_start, i_start + i):
                if (k > phiNumSamples):
                    k1 = k - phiNumSamples
                else:
                    k1 = k
                line_test[k1] = 1
            label_1 = False
            for k in range(phiNumSamples):
                if ((line[k] == 1) and (line_test[k] != 1)): 
                    label_1 = True
                    break
            if not label_1:
                i_range = i
                label_11 = True
                break
        print("New i,i_start,k=",i,i_start,k)
        if label_11:
            break    
    
    print("Final i_range=",i_range)  
    
    jerror = 0.25*j_range
    dtError = jerror * delta * step
    
    ierror = 0.25*i_range
    phiError = 180.  * ierror / (phiNumSamples - 1) 
    
    print("---------------------------------------")
    print("dtError = ",dtError)
    print("phiError=",phiError)

    return phiCalc, dttCalc/sps, FS2[0], FS2[1], FS2_trim[0], FS2_trim[1], FS2_Lag[0], FS2_Lag[1]












