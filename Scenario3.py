from FourRooms import FourRooms
import numpy as np
import random
from collections import deque
import argparse

parser = argparse.ArgumentParser("Make it stochastic")
parser.add_argument('-s','--stochastic', action='store_true')
args = parser.parse_args()


def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v

def Reward(grid_type, current_goal, packageOrder):
    cellType = ['EMPTY', 'RED', 'GREEN', 'BLUE']
    goals = ['RED', 'GREEN', 'BLUE']
    if cellType[grid_type] == goals[current_goal]:
        packageOrder.append(goals[current_goal])
        return 10, packageOrder
    elif cellType[grid_type] in goals and cellType[grid_type] != goals[current_goal]:
         packageOrder.append(goals[current_goal])
         return -5, packageOrder
    else:
        return -0.3, packageOrder
    
def greedyEpsilon(V):
    if np.random.rand() < 0.1:
        return np.random.randint(len(V))
    else:
        return np.argmax(V)

    
        
def main():
     # Create FourRooms Object
     if args.stochastic:
        fourRoomsObj = FourRooms('rgb', True)
        
     else:
        fourRoomsObj = FourRooms('rgb')
     

     aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
     goals = ['RED', 'GREEN', 'BLUE']
     learningRate = 0.5
     discountFactor = 0.9 
     replay_memory_size = 5000  
     batch_size = 32  
     wxh = 13 
     numActions = 4
     replay_memory = deque(maxlen=replay_memory_size)
   
     Q = {}
     for goal in goals:
          Q[goal] = {}
          for x in range(wxh):
               for y in range(wxh):
                    for g in goals:
                         Q[goal][(x, y, g)] = np.zeros(numActions)

     packageOrder = []
     for epoch in range(2000):
          fourRoomsObj.newEpoch()
          cumulativeReward = 0
          currentGoal = 0
          currPackages = fourRoomsObj.getPackagesRemaining()
          p = []
          while True:
               currX, currY = fourRoomsObj.getPosition()
               currentState = (currX, currY, goals[currentGoal])
               V = probs(Q[goals[currentGoal]][currentState])
               action = np.argmax(V)
               
               gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
               nextX, nextY = newPos
               r, p = Reward(gridType, currentGoal, p)
               cumulativeReward+=r
               nextState = (nextX, nextY, goals[currentGoal]) 
               replay_memory.append((currentState, action, r, nextState, isTerminal))
        
               minibatch = random.sample(replay_memory, min(len(replay_memory), batch_size))
               for state, action, reward, next_state, terminal in minibatch:
                    target = reward

               if not terminal:
                    target += discountFactor * np.max(Q[goals[currentGoal]][next_state])
               Q[goals[currentGoal]][state][action] += learningRate * (target - Q[goals[currentGoal]][state][action])
          
               if(currPackages > packagesRemaining):
                    currentGoal = (currentGoal + 1) % len(goals)
                    currPackages =  packagesRemaining
               if isTerminal:
                    break
          packageOrder.append(p)
     print("Done")
     # Show Path
     print(packageOrder[-1])
     fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()
