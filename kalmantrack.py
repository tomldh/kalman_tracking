import argparse
import json
import sys
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import logging
from datetime import datetime
import copy
from sympy.geometry import curve


class Observation():
    
    def __init__(self, tid, z):
        
        self.tid = tid
        self.z = z
        

class Ktraxel():
    
    lostId = '(-1, -1)'
    numOfStates = 6 #number of states to update
    numOfObs = 3 #number of history observation
    
    A = np.array([[1., 0., 1., 0., 0., 0.],
                  [0., 1., 0., 1., 0., 0.],
                  [0., 0., 1., 0., 1., 0.],
                  [0., 0., 0., 1., 0., 1.],
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 1.]])
    
    H = np.array([[1., 0., 0., 0., 0., 0.], 
                  [0., 1., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 1.]])
    
    # assume gaussian noise model
    Q = np.identity(numOfStates, dtype='float64')
    R = np.identity(numOfStates, dtype='float64')
    
    def __init__(self, tid, x, k, p):
        
        self.tid = tid
        
        self.x = x
        
        self.k = k
        
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
            
            traxelList.append(Ktraxel(i, x=np.array([[cx],
                                                       [cy],
                                                       [0.],
                                                       [0.],
                                                       [0.],
                                                       [0.]]),
                                         k = np.identity(Ktraxel.numOfStates, dtype='float64'),
                                         p = np.identity(Ktraxel.numOfStates, dtype='float64')))

def retrieveObs(tid, data, obsList):
    
    hasObs = False
    
    # traxels with lost track will not have fcs
    if tid == Ktraxel.lostId:
        return hasObs
    
    fc = data[tid]['future_connections'];
    
    # add all future connection into list of observations
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


# update the parameters which do not depend on observation
# i.e. regardless of whether there is any valid track
def updateKalmanParamsWithoutObs(X):
    
    A = Ktraxel.A
    H = Ktraxel.H
    Q = Ktraxel.Q
    R = Ktraxel.R
    
    idM = np.identity(Ktraxel.numOfStates, dtype='float64')
    
    for ix in range(len(X)):
        
        #print('X ', X[ix].tid, '\n', X[ix].x)
        logging.debug('X\n%s\n%s', X[ix].tid, X[ix].x)
        
        pp = A.dot((X[ix].p).dot(A.transpose())) + Q
                    
        X[ix].k = pp.dot(H.transpose().dot(np.linalg.inv(H.dot(pp.dot(H.transpose())) + R)))
        
        X[ix].p = (idM-(X[ix].k).dot(H)).dot(pp)
        
        xp = A.dot(X[ix].x);
        
        X[ix].x[0] = xp[0]
        X[ix].x[1] = xp[1]
        X[ix].x[2] = xp[2]
        X[ix].x[3] = xp[3]
        X[ix].x[4] = xp[4]
        X[ix].x[5] = xp[5]
        

def computeCandidateTracks(tracksList, X, data):
    
    obsList = [] # for storing observations (future connection)
    
    for ix, traxel in enumerate(X):
            
        del obsList[:]
        
        #print('X ', traxel.tid)
        logging.debug('X %s', traxel.tid)
        
        # observations obtained from future connections
        hasObs = retrieveObs(traxel.tid, data, obsList)
        
        # no track can be computed if there is no observation
        if not hasObs:
            continue
        
        numPrev = len(traxel.histZ) #number of previous observations stored for a traxel
        #print('History observations: ', numPrev)
        logging.debug('History observations: %s', numPrev)
        
        for ob in obsList:
            
            # velocity can be derived when there is at least 1 history observation 
            if numPrev > 0:
                ob.z[2] = ob.z[0] - traxel.histZ[numPrev-1].z[0]
                ob.z[3] = ob.z[1] - traxel.histZ[numPrev-1].z[1]
            # acceleration can be derived when there is at least 2 history observations
            if numPrev > 1:
                ob.z[4] = ob.z[2] - traxel.histZ[numPrev-1].z[2]
                ob.z[5] = ob.z[3] - traxel.histZ[numPrev-1].z[3] 
                           
            #print('\tobs', ob.tid)
            #print('\t', ob.z)
            logging.debug('\tobs%s', ob.tid)
            logging.debug('\t%s', ob.z)
            
            x = traxel.x + (traxel.k).dot(ob.z - (Ktraxel.H).dot(traxel.x))
            
            # four different ways for comparison
            #diff = np.array([(X[ix][0]-x[0]), (X[ix][1]-x[1])]) #case 1: comparable to case 4, better than 4 due to influence of xp on z
            diff = np.array([(ob.z[0]-x[0]), (ob.z[1]-x[1])]) #case 2
            #diff = np.array([(Z[iz][0]-xp[0]), (Z[iz][1]-xp[1])]) #case 3: same as case 2, as case 2 & 3 depends on the distance to respective z
            #diff = np.array([(Z[iz][0]-X[ix][0]), (Z[iz][1]-X[ix][1])]) #case 4: no good, cannot different equidistant candidates

            dist = np.linalg.norm(diff, 2)
            
            #print('\tdist: ', dist)
            logging.debug('\tdist: %s', dist)
            
            # add candidate track to global list
            tracksList.append((ix, x, ob, dist))
    

def assignTrackByDistance(tracksList, X):
    
    # track[0] - traxel index in X
    # track[1] - aposterior state estimate
    # track[2] - corresponding observation object
    # track[3] - distance
    
    tracksList.sort(key=lambda x: x[3]) # sort according to distance
    
    while len(tracksList):
            
        # deal with the case of the smallest distance first
        track = tracksList.pop(0)
        
        # if updated before, should not consider again
        if X[track[0]].updated:
            continue
        
        ix = track[0]
        
        X[ix].updated = True
        
        # update corresponding traxel, history obs
        X[ix].tid = track[2].tid
        
        #X[ix].x = track[1] # a bug if use this line to copy array
        
        X[ix].x[0] = track[1][0]
        X[ix].x[1] = track[1][1]
        X[ix].x[2] = track[1][2]
        X[ix].x[3] = track[1][3]
        X[ix].x[4] = track[1][4]
        X[ix].x[5] = track[1][5]
        
        if len(X[ix].histZ) == Ktraxel.numOfObs:
            X[ix].histZ.popleft()
        
        X[ix].histZ.append(track[2])
        
        # remove all occurrences of this observation
        rList = [] #list of to-be-removed items
        
        for idx, elem in enumerate(tracksList):
            if elem[2].tid == track[2].tid:
                rList.append(idx)
        
        while len(rList):
            i = rList.pop()
            tracksList.pop(i)
        
            for idx in range(len(rList)):
                if rList[idx] > i:
                    rList[idx] -= 1

        del rList[:]
    
    

# get a list of remaining un-used observations at timestamp
def retrieveRemainingObs(X, obsList, timestamp, data):
    
    tid = []
    
    # FIXME: get the timestampe from m_Xtid instead?
    for i in data:
        if data[i]['time'] == timestamp:
            tid.append(i)
    
    #in m_Xtid, there are 3 cases
    #case 1: obs at timestamp
    #case 2: not updated traxel, id at timestamp-1
    #case 3: lost traxel, id = (-1,-1)
    for traxel in X:
        if not traxel.tid == Ktraxel.lostId:
            if data[traxel.tid]['time'] == timestamp:
                tid.remove(traxel.tid)
        
    for i in tid:
        com = data[i]['com']
        cx = float(com[1:com.index(',')])
        cy = float(com[com.index(',')+1:len(com)-1])
        
        obsList.append(Observation(i, z=np.array([[cx],
                                                  [cy]])))

def reassignLostTraxel(X, obsList):
    # Deal with case of re-appearance
    #FIXME: still consideration at all after n frames?
    #FIXME: use global list of tracks?
    
    for traxel in X:
        if traxel.tid == Ktraxel.lostId:
            mbdist = float('inf')
            mbid = -1
            for idx, ob in enumerate(obsList):
                mdiff = np.array([(traxel.x[0]-ob.z[0]), (traxel.x[1]-ob.z[1])])
                mdist = np.linalg.norm(mdiff, 2)
                if mdist <50 and mdist < mbdist:
                    mbid = idx
                    mbdist = mdist
            
            if mbid > -1:
                traxel.tid = obsList[mbid].tid
                traxel.x[0] = obsList[mbid].z[0]
                traxel.x[1] = obsList[mbid].z[1]
                #FIXME: update history obs?
                traxel.updated = True
                obsList.pop(mbid)

def newTraxel(traxelList, obsList):
    
    # Deal with appearance of new traxel
    for ob in obsList:
        #print("new traxel: ", ob.tid)
        logging.debug('new traxel: %s', ob.tid)
        
        traxelList.append(Ktraxel(ob.tid, x=np.array([ob.z[0],
                                                       ob.z[1],
                                                       [0.],
                                                       [0.],
                                                       [0.],
                                                       [0.]]),
                                  k = np.identity(Ktraxel.numOfStates, dtype='float64'),
                                  p = np.identity(Ktraxel.numOfStates, dtype='float64')))


def drawFrame(traxelList, timestamp, cstr):
    print('frame: ', timestamp)
    logging.debug('frame: %s', timestamp)
    
    fname = 'track' + str(timestamp) + '.png'
    
    frame = np.ones((300,300,3), dtype='uint8')
    frame *= 255
    
    fig = plt.figure(1)
    plt.axis('off')
    ax = fig.add_subplot(1,1,1)
    ax.imshow(frame)
    
    annHist = []
   
    for idx, traxel in enumerate(traxelList):
        ax.add_patch(Circle((traxel.x[0], traxel.x[1]), 10, color=cstr[idx]))
        #annHist.append(ax.annotate('{0}'.format(traxel.tid), xy=(traxel.x[0], traxel.x[1]), xytext=(traxel.x[0], traxel.x[1]+15)))
    
    fig.savefig(fname)
    
    for i in range(len(annHist)):
        annHist[i].remove()

def drawFrameArrow(traxelListPrev, traxelListCurr, timestamp, cstr):
    print('frame: ', timestamp)
    logging.debug('frame: %s', timestamp)
    
    fname = 'track' + str(timestamp) + '.png'
    
    frame = np.ones((300,300,3), dtype='uint8')
    frame *= 255
    
    fig = plt.figure(1)
    plt.axis('off')
    ax = fig.add_subplot(1,1,1)
    ax.imshow(frame)
    
    annHist = []
   
    for idx, cur in enumerate(traxelListCurr):
        
        if (idx < len(traxelListPrev)):
            prev = traxelListPrev[idx] #(traxelListPrev[idx].x[0], traxelListPrev[idx].x[1])
            annHist.append(ax.add_patch(Circle((cur.x[0], cur.x[1]), 10, color=cstr[idx], alpha = 0.5)))
            annHist.append(ax.add_patch(Circle((prev.x[0], prev.x[1]), 10, color=cstr[idx], alpha = 0.5)))
            annHist.append(ax.arrow(float(prev.x[0]), float(prev.x[1]), float(cur.x[0]-prev.x[0]), float(cur.x[1]-prev.x[1]), head_width=5, head_length=5, fc='k', ec='k'))
            
            if cur.tid == Ktraxel.lostId:
                annHist.append(ax.arrow(float(prev.x[0]), float(prev.x[1]), float(cur.x[0]-prev.x[0]), float(cur.x[1]-prev.x[1]), head_width=5, head_length=5, fc='r', ec='r'))
            if prev.tid == Ktraxel.lostId:
                annHist.append(ax.arrow(float(prev.x[0]), float(prev.x[1]), float(cur.x[0]-prev.x[0]), float(cur.x[1]-prev.x[1]), head_width=5, head_length=5, fc='r', ec='r'))
            
        else:
            annHist.append(ax.add_patch(Circle((cur.x[0], cur.x[1]), 10, color=cstr[idx], alpha = 0.5)))
        
    
    fig.savefig(fname, bbox_inches='tight', pad_inches=0, frameon=True)
    
    for i in range(len(annHist)):
        annHist[i].remove()

def track(data, startIndex, stopIndex):
    
    X = [] #list of traxels
    Xprev = [] #a copy of last frame
    gList = [] #stores tuples of (...) for all traxels and their all observations
    unusedObs = [] #list of un-used observation
    color_traxel = [] #colors for drawing the object 
    
    np.random.seed(10)
    
    for i in range(100):
        color_traxel.append(np.random.rand(3,1))

    
    initTraxels(X, data, startIndex)
    
    Xprev = copy.deepcopy(X)
    #for item in X:
    #    Xprev.append((item.x[0], item.x[1]))
    
    #drawFrame(X, startIndex, color_traxel)
    
    drawFrameArrow(Xprev, X, startIndex, color_traxel)
    
    fIndex = startIndex
    
    # perform tracking until stopping frame
    while fIndex < stopIndex:
        
        del Xprev[:]
        Xprev = copy.deepcopy(X)
        #for item in X:
        #    Xprev.append((float(item.x[0]), float(item.x[1])))
        
        fIndex += 1
        
        del gList[:]
        del unusedObs[:]
        
        updateKalmanParamsWithoutObs(X)
        
        computeCandidateTracks(gList, X, data)

        assignTrackByDistance(gList, X)
        
        retrieveRemainingObs(X, unusedObs, fIndex, data)
        
        reassignLostTraxel(X, unusedObs)
        
        # if no update, no fc found
        # FIXME: clear the history observations as well?
        for traxel in X:
            if traxel.updated == False:
                traxel.tid = Ktraxel.lostId
        
        # reset indicators
        for traxel in X:
            traxel.updated = False
        
        newTraxel(X, unusedObs)
        
        #drawFrame(X, fIndex, color_traxel)
        
        drawFrameArrow(Xprev, X, fIndex, color_traxel)
        
        
        if (fIndex > 10000):
            input("Press any key for next timestamp...\n")



def loadDataFromJsonFile(fpath):
    
    try:
        with open(fpath, 'r') as fp:
            data = json.load(fp)
    except:
        print('Error opening file:', fpath)
        logging.debug('Error opening file: %s', fpath)
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
        logging.debug('Error in start-stop frame index: start-%s, stop-%s', fstart, fstop)
        logging.debug('Perform tracking on all frames.')
        
        return start, stop
    
    return fstart, fstop



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Kalman tracking')
    
    parser.add_argument('-f', '--file', default='data/sporos_m3.json.json', type=str, dest='jFile',
                        help='Filename of the json data file')
    
    parser.add_argument('--start', default=0, type=int, dest='fstart',
                        help='Index of frame to start tracking')
    
    parser.add_argument('--stop', default=-1, type=int, dest='fstop',
                    help='Index of frame to stop tracking')
    
    parser.add_argument('-l', '--log', default=True, type=bool, dest='enableLog',
                        help='logging')
    
    args = parser.parse_args()
    
    if args.enableLog:
        logging.basicConfig(filename=datetime.now().strftime('%Y_%m_%d_%H_%M.log'),level=logging.DEBUG)
    
    jdata = loadDataFromJsonFile(args.jFile)
    
    # check validity of fstart and fstop
    start, stop = checkFrameIndex(jdata, args.fstart, args.fstop)
    
    track(jdata, start, stop)
    
    
    pass