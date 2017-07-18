import argparse
import json
import sys
import numpy as np
from collections import deque

class Observation():
    
    def __init__(self, tid, z):
        
        self.tid = tid
        self.z = z
        

class Ktraxel():
    
    numOfStates = 6 # to be used to valid dimen later
    
    A = np.array([[1., 0., 1., 0., 0., 0.],
                  [0., 1., 0., 1., 0., 0.],
                  [0., 0., 1., 0., 1., 0.],
                  [0., 0., 0., 1., 0., 1.],
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 1.]]) #6x6 float64
    
    H = np.array([[1., 0., 0., 0., 0., 0.], 
                  [0., 1., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 1.]]) #2x6float64
    
    # assume gaussian noise model
    Q = np.identity(numOfStates, dtype='float64') #6x6
    R = np.identity(numOfStates, dtype='float64') #6x6
    
    def __init__(self, tid, x, p):
        
        self.tid = tid
        
        self.x = x
        
        self.p = p
        
        self.histZ = deque()
        
        self.histZ.append(Observation(tid, x))
        
        self.updated = False


def initTraxels(traxelList, data, startIndex):
    
    for i in data:
        timestamp = data[i]['time']
        
        if timestamp == startIndex:
            
            com = data[i]['com']
            cx = float(com[1:com.index(',')])
            cy = float(com[com.index(',')+1:len(com)-1])
            
            traxelList.append(Ktraxel(i, 
                                           x=np.array([[cx],
                                                       [cy],
                                                       [0.],
                                                       [0.],
                                                       [0.],
                                                       [0.]]), 
                                           p=np.identity(6, dtype='float64')))

def retrieveObs(tid, data, obsList):
    
    hasObs = False
    
    # traxels with lost track will not have fcs
    if tid == '(-1, -1)':
        return hasObs
    
    fc = data[tid]['future_connections'];
    
    for i in range(len(fc)):
        
        com = data[fc[i]]['com']
        cx = float(com[1:com.index(',')])
        cy = float(com[com.index(',')+1:len(com)-1])
        
        obsList.append(Observation(fc[i], z=np.array([[cx],
                                                      [cy],
                                                      [0.],
                                                      [0.],
                                                      [0.],
                                                      [0.]])))
        
        hasObs = True

    return hasObs


def computeCandidateTracks(tracksList, X, data):
    
    A = Ktraxel.A
    H = Ktraxel.H
    Q = Ktraxel.Q
    R = Ktraxel.R
    idM = np.identity(Ktraxel.numOfStates, dtype='float64')
    z = [] # for storing observations (future connection)
    
    for ix in range(len(X)):
            
        del z[:]
        
        print('X ', X[ix].tid, '\n', X[ix].x)
        
        pp = A.dot((X[ix].p).dot(A.transpose())) + Q
                        
        k = pp.dot(H.transpose().dot(np.linalg.inv(H.dot(pp.dot(H.transpose())) + R)))
        
        p = (idM-k.dot(H)).dot(pp)
        
        xp = A.dot(X[ix].x);
        #print('xp', xp)
        
        # update motion prediction regardless of having valid tracks
        X[ix].x[0] = xp[0]
        X[ix].x[1] = xp[1]
        X[ix].x[2] = xp[2]
        X[ix].x[3] = xp[3]
        X[ix].x[4] = xp[4]
        X[ix].x[5] = xp[5]
        
        # update P regardless of having valid tracks
        X[ix].p = p
    
        # observations in next timestamp
        hasObs = retrieveObs(X[ix].tid, data, z)
        
        if not hasObs:
            continue
        
        numPrev = len(X[ix].histZ) #number of previous observations stored for a traxel
        print('History observations: ', numPrev)
        print(X[ix].histZ)
        
        for iz in range(len(z)):
             
            if numPrev > 0:
                z[iz].z[2] = z[iz].z[0] - X[ix].histZ[numPrev-1].z[0]
                z[iz].z[3] = z[iz].z[1] - X[ix].histZ[numPrev-1].z[1]
                
            if numPrev > 1:
                z[iz].z[4] = z[iz].z[2] - X[ix].histZ[numPrev-1].z[2]
                z[iz].z[5] = z[iz].z[3] - X[ix].histZ[numPrev-1].z[3] 
                           
            print('\tobs',iz, ' ', z[iz].tid)
            print('\t', z[iz].z)
            
            x = X[ix].x + k.dot(z[iz].z - H.dot(X[ix].x))
            print('x', x)
            
            # four different ways for comparison
            #diff = np.array([(X[ix][0]-x[0]), (X[ix][1]-x[1])]) #case 1: comparable to case 4, better than 4 due to influence of xp on z
            diff = np.array([(z[iz].z[0]-x[0]), (z[iz].z[1]-x[1])]) #case 2
            #diff = np.array([(Z[iz][0]-xp[0]), (Z[iz][1]-xp[1])]) #case 3: same as case 2, as case 2 & 3 depends on the distance to respective z
            #diff = np.array([(Z[iz][0]-X[ix][0]), (Z[iz][1]-X[ix][1])]) #case 4: no good, cannot different equidistant candidates
            dist = np.linalg.norm(diff, 2)
            
            print('\tdist: ', dist)
            
            tracksList.append((ix, p, z[iz].tid, z[iz].z, x, dist))

    
    
            
def track(data, startIndex, stopIndex):
    
    X = [] # list of traxels
    gList = [] #stores tuples of (...) for all traxels and their all observations

        
    
    initTraxels(X, data, startIndex)
    
    fIndex = startIndex
    
    # perform tracking until stopping frame
    while fIndex < stopIndex:
        
        del gList[:]
        
        computeCandidateTracks(gList, X, data)
        
        input("Press any key for next timestamp...\n")
        
        
        
        
        
    
    
    



def loadDataFromJsonFile(fpath):
    
    try:
        with open(fpath, 'r') as fp:
            data = json.load(fp)
    except:
        print('Error opening file:', fpath)
        sys.exit()
    
    return data

def checkFrameIndex(data, fstart, fstop):
    
    timestamps = np.zeros(len(data), dtype=np.uint64)
    cnt = 0
    
    for i in data:
        timestamps[cnt] = data[i]['time']
        cnt += 1
    
    start = np.amin(timestamps)
    stop = np.amax(timestamps)
    
    if fstop == -1:
        fstop = stop
    
    if fstart > stop or fstop > stop or fstart < start or fstop < start:
        print('Error in start-stop frame index: start-', fstart, ', stop-', fstop)
        print('Perform tracking on all frames.')
        
        return start, stop
    
    return fstart, fstop


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Kalman tracking')
    
    parser.add_argument('-f', '--file', default='data/twoArcs_Frame__COM-json2.json', type=str, dest='jFile',
                        help='Filename of the json data file')
    
    parser.add_argument('--start', default=0, type=int, dest='fstart',
                        help='Index of frame to start tracking')
    
    parser.add_argument('--stop', default=-1, type=int, dest='fstop',
                    help='Index of frame to stop tracking')
    
    args = parser.parse_args()
    
    jdata = loadDataFromJsonFile(args.jFile)
    
    # check validity of fstart and fstop
    start, stop = checkFrameIndex(jdata, args.fstart, args.fstop)
    
    track(jdata, start, stop)
    
    
    pass