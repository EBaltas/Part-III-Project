#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:44:49 2021

@author: elisavetbaltas
"""

import numpy as np
import obspy as obs
from obspy.signal.util import next_pow_2
import itertools
from scipy.stats import f
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel


def RotMat2D(phi):
    
    phi = np.deg2rad(phi)
    return np.array([[np.cos(phi), np.sin(phi)],
                     [-np.sin(phi), np.cos(phi)]])

def SourcePol(phicalc, V1_x, V1_y):
    
    return phicalc + np.rad2deg(np.arctan2(V1_x, V1_y)) + 180

def CR95(L2, NDF, C=2, a=0.05):
    
    return L2*(1+((C/(NDF-C))*f.isf(a, C, NDF-C)))

def SAndC_ME(st_filt, baz, starttime, endtime, timeDelay=0.25):
    
    NE = np.array((
        st_filt.select(component="N")[0].data,
        st_filt.select(component="E")[0].data))
    
    sps = st_filt[0].stats.sampling_rate
    timeDelay *= sps
    step = 1
    
    phiSamples = np.arange(-90, 90+step, step, dtype=int)
    dttSamples = np.arange(0, timeDelay+step, step, dtype=int)
    
    phiNumSamples = phiSamples.size
    dttNumSamples = dttSamples.size
    
    Ematrix = np.zeros((phiNumSamples, dttNumSamples))
    Lambda2 = np.zeros((phiNumSamples, dttNumSamples))
    Lambda1 = np.zeros((phiNumSamples, dttNumSamples))
    Vector1 = np.zeros((phiNumSamples, dttNumSamples, 2))
    
    #Rotate and grab Q and T components
    M = RotMat2D(baz)
    QT = np.dot(M, NE)
    
    selectionWin = np.array((int(starttime*sps), int(endtime*sps)))
    
    for i in range(phiNumSamples):
        phi = phiSamples[i]
        
        R = RotMat2D(phi)
        
        FS = np.dot(R, QT)
        
        for j in range(dttNumSamples):
            dtt = dttSamples[j]
            delayWin = selectionWin + dtt
            print("SelectWin=",selectionWin," delayWin=",delayWin)
        
            FS_lag = np.array((
                FS[0][selectionWin[0]:selectionWin[1]],         
                FS[1][delayWin[0]:delayWin[1]]  
            ))
            
            #Rotate back to QT
            
            QT_lag = np.dot(R.T, FS_lag)
            
            #Get energy of transverse component
            Ematrix[i, j] = np.sum(np.square(QT_lag[1]))
        
        
    MEindex = np.where(Ematrix == Ematrix.min()) 
    

    if len(MEindex) > 2:
        raise IndexError("More than 1 solution found")
    
    phiCalc = phiSamples[MEindex[0]][0]
    dttCalc = dttSamples[MEindex[1]][0]
    
    return phiCalc, dttCalc/sps