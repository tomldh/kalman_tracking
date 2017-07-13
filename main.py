import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
from collections import deque

def loadDataFromJsonFile():
    
    '''
    sample syntax
    --dict obj
    
     "(0, 1)"-this is string: {
         
         --dict obj: "future_connections", "com", "id", "time"
     
        "future_connections": [
            "(1, 1)" --this is list
        ], 
        "com": "(16.5, 51.5)" -- this is string, 
        "id": 1,---this is int 
        "time": 0
        
        
    }
    '''
    
    index = int(input("Please provide json file path index:\n \
                    1 --> twoArcs_Frame__COM-json.json\n \
                    2 --> twoArcs_Frame__COM-json2.json\n \
                    3 --> sporos_m3.json.json\n"))
    
    if index == 1:
        fpath = 'data/twoArcs_Frame__COM-json.json'
    
    elif index == 2:
        fpath = 'data/twoArcs_Frame__COM-json2.json'
    
    elif index == 3:
        fpath = 'data/sporos_m3.json.json'
        
    try:
        with open(fpath, 'r') as fp:
            data = json.load(fp)
    except:
        print('Error opening file:', fpath)
        sys.exit()
    
    return data


def initStates(m_X, m_tid, m_updated, m_P, data):
    
    for i in data:
        timestamp = data[i]['time']
        
        if timestamp == 0:
            
            com = data[i]['com']
            cx = float(com[1:com.index(',')])
            cy = float(com[com.index(',')+1:len(com)-1])
            
            m_tid.append(i)
            m_X.append(np.array([[cx],
                                 [cy],
                                 [0.],
                                 [0.],
                                 [0.],
                                 [0.]]))
            m_updated.append(False)
            m_P.append(np.identity(6, dtype='float64'))

def initObs(m_Z, m_Ztid, m_X, m_Xtid):
    
    for i in range(len(m_X)):
        m_Z.append(deque())
        m_Ztid.append(deque())
        m_Z[i].append(np.copy(m_X[i]))
        m_Ztid[i].append(np.copy(m_Xtid[i]))
    
            
def retrieveObs(tid, data, m_obs, m_tid):
    
    hasData = False
    
    # traxels with lost track will not have fcs
    if tid == '(-1, -1)':
        return hasData
    
    fc = data[tid]['future_connections'];
    
    for i in range(len(fc)):
        
        com = data[fc[i]]['com']
        cx = float(com[1:com.index(',')])
        cy = float(com[com.index(',')+1:len(com)-1])
        
        m_tid.append(fc[i])
        m_obs.append(np.array([[cx],
                               [cy],
                               [0.],
                               [0.],
                               [0.],
                               [0.]]))
        
        hasData = True

    return hasData

# get a list of remaining un-used observations at timestamp
def retrieveRemainingObs(m_rz, m_rztid, m_Xtid, timestamp, data):
    
    # FIXME: get the timestampe from m_Xtid instead?
    for i in data:
        if data[i]['time'] == timestamp:
            m_rztid.append(i)
    
    #in m_Xtid, there are 3 cases
    #case 1: obs at timestamp
    #case 2: not updated traxel, id at timestamp-1
    #case 3: lost traxel, id = (-1,-1)
    for i in m_Xtid:
        if not i == '(-1, -1)':
            if data[i]['time'] == timestamp:
                m_rztid.remove(i)
        
    for i in m_rztid:
        com = data[i]['com']
        cx = float(com[1:com.index(',')])
        cy = float(com[com.index(',')+1:len(com)-1])
        
        m_rz.append(np.array([[cx],
                              [cy]]))
              

def drawFrame(m_X, m_Xtid, fIdx, cstr):
    print('frame: ', fIdx)
    # FIXME: change this for more tracks!
    '''
    cstr = ['#000000', '#800000', '#FF0000', '#FFC9DE', '#AA6E28', '#FF9900', '#FFD8B1', '#808000',
            '#FFEA00', '#FFFAC8', '#BEFF00', '#00BE00', '#AAFFC3', '#008080', '#64FFFF', '#000080', 
            '#4385FF', '#820096', '#E6BEFF', '#FF00FF', '#808080', '#002300', '#563342', '#F24F2F']
    '''
    
    fname = 'track' + str(fIdx) + '.png'
    
    frame = np.ones((300,300,3), dtype='uint8')
    frame *= 255
    
    fig = plt.figure(1)
    plt.axis('off')
    ax = fig.add_subplot(1,1,1)
    ax.imshow(frame)
    
    annHist = []
    
    for i in range(len(m_X)):
        ax.add_patch(Circle((m_X[i][0], m_X[i][1]), 10, color=cstr[i]))
        annHist.append(ax.annotate('{0}'.format(m_Xtid[i]), xy=(m_X[i][0], m_X[i][1]), xytext=(m_X[i][0], m_X[i][1]+15)))
    
    fig.savefig(fname)
    
    for i in range(len(annHist)):
        annHist[i].remove()

if __name__ == '__main__':
    
    X = [] #stores state 4x1 (x,y,vx,vy,ax,ay) of each id node
    X_tid = [] # store time-id tuple corresp. to items in X
    X_updated = [] # indicator to avoid multiple updates for same traxel in one timestamp
    P = []
    
    Z = []
    Z_tid = []
    
    gList = [] #stores tuples of (...) for all traxels and their all observations
    
    rz = [] #stores list of NEW traxels not in the X_tid list AT CURRENT TIMESTAMP
    rz_tid = []
    
    color_traxel = []
    
    for i in range(100):
        color_traxel.append(np.random.rand(3,1))
    
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
    Q = np.identity(6, dtype='float64') #6x6
    R = np.identity(6, dtype='float64') #6x6
    
    #P = np.identity(4, dtype='float64') #4x4
    #K = np.zeros((4,2), dtype='float64') #4x2
    
    jdata = loadDataFromJsonFile()
    
    #initialize state
    #assumption: we only track the traxel which exists in the initial states
    #            if there is new traxel in future timestamp, it will be ignored
    initStates(X, X_tid, X_updated, P, jdata)
    initObs(Z, Z_tid, X, X_tid)
    '''
    print(type(Z))
    print(type(Z[0]))
    print(len(Z))
    print(len(Z[0]))
    print(Z[1])
    print(Z[0])
    '''
    
    tcnt = 0
    
    drawFrame(X, X_tid, 0, color_traxel)
    
    
    
    hasTrack = True
    
    while hasTrack:
        
        tcnt += 1
        
        hasTrack = False
        
        del gList[:]

        # loop through each traxel
        for ix in range(len(X)):
            
            print('X ', X_tid[ix], '\n', X[ix])
            
            # for storing observations (future connection)
            z = []
            z_tid = []
            
            # observations in next timestamp
            valid = retrieveObs(X_tid[ix], jdata, z, z_tid)
            
            #if not valid:
                #do something if there is no future connection
                #continue
            
            #hasTrack = True
            
            # if-case is to be deleted, no further use
            # unique track in next timestamp, update X with item 
            if len(Z) == 0:
                X_tid[ix] = Z_tid[0]
                
                X[ix][2] = Z[0][0] - X[ix][0]
                X[ix][3] = Z[0][1] - X[ix][1]
                X[ix][0] = Z[0][0]
                X[ix][1] = Z[0][1]
                
                pp = A.dot(P[ix].dot(A.transpose())) + Q
                    
                k = pp.dot(H.transpose().dot(np.linalg.inv(H.dot(pp.dot(H.transpose())) + R)))
                
                p = (np.identity(4, dtype='float64')-k.dot(H)).dot(pp)
                
                P[ix] = p
                
                
                
            
            # more than one possible track
            else:
                
                # id of best observation
                bid = -1
                bdist = float('inf')
                bP = np.copy(P[ix])
                bX = np.copy(X[ix])
                
                pp = A.dot(P[ix].dot(A.transpose())) + Q
                    
                k = pp.dot(H.transpose().dot(np.linalg.inv(H.dot(pp.dot(H.transpose())) + R)))
                
                p = (np.identity(6, dtype='float64')-k.dot(H)).dot(pp)
                
                xp = A.dot(X[ix]);
                print('xp', xp)
                
                # update motion prediction regardless of having valid tracks
                X[ix][0] = xp[0]
                X[ix][1] = xp[1]
                X[ix][2] = xp[2]
                X[ix][3] = xp[3]
                X[ix][4] = xp[4]
                X[ix][5] = xp[5]
                
                # update P regardless of having valid tracks
                P[ix] = p
                
                if not valid:
                    continue
                
                hasTrack = True
                
                numPrev = len(Z[ix]) #number of previous observations stored for a traxel
                print('History observations: ', numPrev)
                print(Z[ix])
                
                for iz in range(len(z)):
                     
                    if numPrev > 0:
                        z[iz][2] = z[iz][0] - Z[ix][numPrev-1][0]
                        z[iz][3] = z[iz][1] - Z[ix][numPrev-1][1]
                        
                    if numPrev > 1:
                        z[iz][4] = z[iz][2] - Z[ix][numPrev-1][2]
                        z[iz][5] = z[iz][3] - Z[ix][numPrev-1][3] 
                                   
                    print('\tobs',iz, ' ', z_tid[iz])
                    print('\t', z[iz])
                    
                    x = xp + k.dot(z[iz] - H.dot(xp))
                    print('x', x)
                    
                    # four different ways for comparison
                    #diff = np.array([(X[ix][0]-x[0]), (X[ix][1]-x[1])]) #case 1: comparable to case 4, better than 4 due to influence of xp on z
                    diff = np.array([(z[iz][0]-x[0]), (z[iz][1]-x[1])]) #case 2
                    #diff = np.array([(Z[iz][0]-xp[0]), (Z[iz][1]-xp[1])]) #case 3: same as case 2, as case 2 & 3 depends on the distance to respective z
                    #diff = np.array([(Z[iz][0]-X[ix][0]), (Z[iz][1]-X[ix][1])]) #case 4: no good, cannot different equidistant candidates
                    dist = np.linalg.norm(diff, 2)
                    
                    print('\tdist: ', dist)
                    
                    gList.append((ix, p, z_tid[iz], z[iz], x, dist))
                    
                    if dist < bdist:
                        bdist = dist
                        bid = iz
                        bX = np.copy(x)
                        
                    
                if bid < 0:
                    print('Error: best observation id cannot be zero')
                
                if np.array_equal(bX, X[ix]):
                    print('Error: no updated states from kalman filter')
                    
                print('\tbest id ', bid, ' ', z_tid[bid])
                '''
                X_tid[ix] = z_tid[bid]
                
                X[ix][0] = bX[0]
                X[ix][1] = bX[1]
                X[ix][2] = bX[2]
                X[ix][3] = bX[3]
                X[ix][4] = bX[4]
                X[ix][5] = bX[5]
                
                P[ix] = p
                
                if numPrev == 3:
                    Z[ix].popleft()
                    Z_tid[ix].popleft()
                
                Z[ix].append(z[bid])
                Z_tid[ix].append(z_tid[bid])
                '''
        
        # dynamic fc selection
        # fc may be removed if already selected by other traxel
        
        gList.sort(key=lambda x: x[5]) # sort according to distance with fcs
        
        while len(gList):
            
            # deal with the case of the smallest distance first
            tpl = gList.pop(0)
            
            # if updated before, should not consider again
            if X_updated[tpl[0]]:
                continue
            
            X_updated[tpl[0]] = True
            
            # update corresponding states, history obs
            X_tid[tpl[0]] = tpl[2]
            
                
            X[tpl[0]][0] = tpl[4][0]
            X[tpl[0]][1] = tpl[4][1]
            X[tpl[0]][2] = tpl[4][2]
            X[tpl[0]][3] = tpl[4][3]
            X[tpl[0]][4] = tpl[4][4]
            X[tpl[0]][5] = tpl[4][5]
            
            P[tpl[0]] = tpl[1]
            
            if len(Z[tpl[0]]) == 3:
                Z[tpl[0]].popleft()
                Z_tid[tpl[0]].popleft()
            
            Z[tpl[0]].append(tpl[3])
            Z_tid[tpl[0]].append(tpl[2])
            
            # remove all occurrences of this fc
            rList = [] #list of to-be-removed items
            
            for elem in gList:
                if elem[2] == tpl[2]:
                    rList.append(elem)
            
            for elem in rList:
                gList.remove(elem)
            
            del rList[:]
            
        # The remaining un-used observations at current timestamp may include:
        # 1) re-appearance of traxel with lost tracks
        # 2) appearance of new traxel
        # 3) dividing etc.
        
        del rz[:]
        del rz_tid[:]
        
        # retrieve un-used observations
        retrieveRemainingObs(rz, rz_tid, X_tid, tcnt, jdata)
        
        # Deal with case of re-appearance
        #FIXME: still consideration at all after n frames?
        #FIXME: use global list of tracks?
        for ix in range(len(X_tid)):
            if X_tid[ix] == '(-1, -1)':
                removeId = -1
                mbdist = float('inf')
                mbid = -1
                for inx in range(len(rz)):
                    mdiff = np.array([(X[ix][0]-rz[inx][0]), (X[ix][1]-rz[inx][1])])
                    mdist = np.linalg.norm(mdiff, 2)
                    if mdist <50 and mdist < mbdist:
                        mbid = inx
                        mbdist = mdist
                
                if mbid > -1:
                    X_tid[ix] = rz_tid[mbid]
                    X[ix][0] = rz[mbid][0]
                    X[ix][1] = rz[mbid][1]
                    #FIXME: update history obs?
                    X_updated[ix] = True
                    rz.pop(mbid)
                    rz_tid.pop(mbid)
        
         
        # if no update, no fc found
        # FIXME: clear the history observations as well?
        for i in range(len(X_updated)):
            if X_updated[i] == False:
                X_tid[i] = '(-1, -1)'
                
        # reset indicators
        for i in range(len(X_updated)):
            X_updated[i] = False
        
        # Deal with appearance of new traxel
        for i in range(len(rz)):
            print("new traxel: ", rz_tid[i])
            X_tid.append(rz_tid[i])
            X.append(np.array([[rz[i][0][0]],
                               [rz[i][1][0]],
                               [0.],
                               [0.],
                               [0.],
                               [0.]]))
            X_updated.append(False)
            P.append(np.identity(6, dtype='float64'))
            Z.append(deque())
            Z_tid.append(deque())
            Z[len(X)-1].append(np.copy(X[len(X)-1]))
            Z_tid[len(X)-1].append(np.copy(X_tid[len(X)-1]))
        
        
        # draw a frame for each timestamp
        #drawFrame(X, X_tid, jdata[X_tid[0]]['time'])
        
        drawFrame(X, X_tid, tcnt, color_traxel)
        
        if tcnt > 170:
            input("Press any key for next timestamp...\n")
    
    print('Tracking ends.')
    
    
