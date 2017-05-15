import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys

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
    
    try:
        with open('twoArcs_Frame__COM-json.json', 'r') as fp:
            data = json.load(fp)
    except:
        print('Error opening file')
        sys.exit()
    
    return data

def initStates(m_X, m_tid, m_P, data):
    
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
                                 [0.]]))
            m_P.append(np.identity(4, dtype='float64'))
            
def retrieveObs(tid, data, m_obs, m_tid):
    
    hasData = False
    
    fc = data[tid]['future_connections'];
    
    for i in range(len(fc)):
        
        com = data[fc[i]]['com']
        cx = float(com[1:com.index(',')])
        cy = float(com[com.index(',')+1:len(com)-1])
        
        m_tid.append(fc[i])
        m_obs.append(np.array([[cx],
                               [cy]]))
        
        hasData = True

    return hasData
            

def drawFrame(m_X, fIdx):
    
    # FIXME: change this for more tracks!
    cstr = ['#000000', '#800000', '#FF0000', '#FFC9DE', '#AA6E28', '#FF9900', '#FFD8B1', '#808000',
            '#FFEA00', '#FFFAC8', '#BEFF00', '#00BE00', '#AAFFC3', '#008080', '#64FFFF', '#000080', 
            '#4385FF', '#820096', '#E6BEFF', '#FF00FF', '#808080', '#002300', '#563342', '#F24F2F']
    
    fname = 'track' + str(fIdx) + '.png'
    
    frame = np.ones((200,200,3), dtype='uint8')
    frame *= 255
    
    fig = plt.figure(1)
    plt.axis('off')
    ax = fig.add_subplot(1,1,1)
    ax.imshow(frame)
    
    for i in range(len(m_X)):
        ax.add_patch(Circle((m_X[i][0], m_X[i][1]), 10, color=cstr[i]))
    
    fig.savefig(fname)
    

if __name__ == '__main__':
    
    X = [] #stores state 4x1 (x,y,vx,vy) of each id node
    X_tid = [] # store time-id tuple corresp. to items in X
    P = []
    
    Z = []
    Z_tid = []
    
    A = np.array([[1., 0., 1., 0.],
                  [0., 1., 0., 1.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]]) #4x4 float64
    
    H = np.array([[1., 0., 0., 0.], 
                  [0., 1., 0., 0.]]) #2x4float64
    
    # assume gaussian noise model
    Q = np.identity(4, dtype='float64') #4x4
    R = np.identity(2, dtype='float64') #2x2
    
    #P = np.identity(4, dtype='float64') #4x4
    #K = np.zeros((4,2), dtype='float64') #4x2
    
    jdata = loadDataFromJsonFile()
    
    #initialize state
    initStates(X, X_tid, P, jdata)
    
    drawFrame(X, 0)
    
    hasTrack = True
    
    while hasTrack:
        
        hasTrack = False
        
        # loop through each traxel
        for ix in range(len(X)):
            
            print('X ', X_tid[ix], '\n', X[ix])
            
            # empty old observations
            del Z[:]
            del Z_tid[:]
            
            # observations in next timestamp
            valid = retrieveObs(X_tid[ix], jdata, Z, Z_tid)
            
            if not valid:
                #do something if there is no future connection
                continue
            
            hasTrack = True
            
            # unique track in next timestamp, update X with item 
            if len(Z) == 1:
                X_tid[ix] = Z_tid[0]
                
                X[ix][2] = Z[0][0] - X[ix][0]
                X[ix][3] = Z[0][1] - X[ix][1]
                X[ix][0] = Z[0][0]
                X[ix][1] = Z[0][1]
                
                print('vel', X[ix][2])
                
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
                
                pp = A.dot(P[ix].dot(A.transpose())) + Q
                    
                k = pp.dot(H.transpose().dot(np.linalg.inv(H.dot(pp.dot(H.transpose())) + R)))
                
                p = (np.identity(4, dtype='float64')-k.dot(H)).dot(pp)
                
                for iz in range(len(Z)):
                    
                    print('\tZ',iz, ' ', Z_tid[iz])
                    print('\t', Z[iz])
                    
                    xp = A.dot(X[ix]);
                    print('xp', xp[0], xp[1])
                    
                    x = xp + k.dot(Z[iz] - H.dot(xp))
                    print('x', x[0], x[1])
                    
                    # four different ways for comparison
                    diff = np.array([(X[ix][0]-x[0]), (X[ix][1]-x[1])]) #correct
                    #diff = np.array([(Z[iz][0]-x[0]), (Z[iz][1]-x[1])]) #wrong
                    #diff = np.array([(Z[iz][0]-xp[0]), (Z[iz][1]-xp[1])]) #wrong
                    #diff = np.array([(Z[iz][0]-X[ix][0]), (Z[iz][1]-X[ix][1])]) #cannot differentiate
                    dist = np.linalg.norm(diff, 2)
                    
                    print('\tdist: ', dist)
                    
                    if dist < bdist:
                        bdist = dist
                        bid = iz
                        
                    
                if bid < 0:
                    print('Error: best observation id cannot be zero')
                
                print('\tbest id ', bid, ' ', Z_tid[bid])
                
                X_tid[ix] = Z_tid[bid]
                
                X[ix][2] = Z[bid][0] - X[ix][0]
                X[ix][3] = Z[bid][1] - X[ix][1]
                X[ix][0] = Z[bid][0]
                X[ix][1] = Z[bid][1]
                
                print('vel', X[ix][2])
                
                P[ix] = p
            
               
        print('P\n', P[ix])
        
        # draw a frame for each timestamp
        drawFrame(X, jdata[X_tid[0]]['time'])
        
        input("Press any key for next timestamp...\n")
    
    print('Tracking ends.')
    
    
