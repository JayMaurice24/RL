from FourRooms import FourRooms
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser("Make it stochastic")
parser.add_argument('-s','--stochastic', action='store_true')
args = parser.parse_args()

def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v

def policy(fourRoomsObj, Q):
        x,y = fourRoomsObj.getPosition()
        v = probs(Q[x,y])
        a = random.choices([a for a in range(4)],weights=v)
        return a

def reward(gridType, isTerminal):
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']
    if(isTerminal): 
           return 10
    else:
         return -0.2
                 
def main():
    # Create FourRooms Object
    stocflag = False
    if args.stochastic:
        fourRoomsObj = FourRooms('simple', True)
        print("exploration")
    else:
        fourRoomsObj = FourRooms('simple')
    # This will try to draw a zero
    actSeq = [FourRooms.LEFT, FourRooms.LEFT, FourRooms.LEFT,
              FourRooms.UP, FourRooms.UP, FourRooms.UP,
              FourRooms.RIGHT, FourRooms.RIGHT, FourRooms.RIGHT,
              FourRooms.DOWN, FourRooms.DOWN, FourRooms.DOWN]

    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    
    learningRate = 0.1
    discountFactor = 0.9 

    n = 13 #width x height
    #Q = np.ones((n,n,len(aTypes)),dtype=float)*1.0/len(aTypes)
    Q = np.zeros((n,n,len(aTypes)),dtype=float)

    print('Agent starts at: {0}'.format(fourRoomsObj.getPosition()))

    for epoch in range(500):
        fourRoomsObj.newEpoch()
        cumulativeReward = 0
        n = 0 
        while True:
            currX, currY = fourRoomsObj.getPosition()
            v = probs(Q[currX,currY])
            act = np.argmax(v)
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(act)
            nextX, nextY = newPos
            r = reward(gridType, isTerminal)
            cumulativeReward+=r
            #print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[act], newPos, gTypes[gridType]))
            #Q[currX,currY,act] = (1 - learningRate) * Q[currX,currY,act] + learningRate * (r + discountFactor* Q[nextX, nextY].max())
            Q[currX, currY, act] += learningRate * \
            (r + discountFactor *
             np.max(Q[nextX, nextY]) - Q[currX, currY, act])
            if isTerminal or cumulativeReward < -1000:
                break

    print("Done")
    #print(Q)
    # Show Path
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()
