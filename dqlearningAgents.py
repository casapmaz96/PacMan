import learningAgents as la
import game
import DQNet
from DQNet import Observation
import numpy as np
import torch
import torch.optim as optim
import util
import random
import torch.nn.functional as func
from qlearningAgents import PacmanQAgent
import time
import matplotlib.pyplot as plt
class DQLAgent(la.ReinforcementAgent):
    """This code is implemented with the help of the official Pytorch tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""

    """Functions to know:
	-observationFunction: called by the environment in state transitions. Calls observeTransition.
	Functions to override:
	-observeTransition: the function that is responsible of calling update at every transition.
				      In DQLAgent it should add transitions to experience replay memory instead
				      and call update based on batchsize.
	-update: called by observeTransition to update the q function estimations.
		     In DQLAgent it should perform batch train with replay memory.
        -final: called by the environment when an episode ends. Calls observeTransition, records ep. statistics.
	-getAction: returns action based on state according to policy.
	-getQValue: returns q value of state and action
	-getValue: returns the highest estimated qvalue of a state """

    def __init__(self, **args):
        self.epsilon=0.1 #args['epsilon']
        self.alpha=0.2 #args['alpha']
        self.gamma=0.8 #args['gamma']
        self.width=args['width']; self.height=args['height']
        if self.height == 7:
            if self.width == 7:
                self.lo = 'sg'
            elif self.width == 8:
                self.lo = 'mg'
            else: self.lo = 'sc'
        elif self.height == 11:
            self.lo = 'mc'
        else:
            self.lo = 'tc'
        if torch.cuda.is_available(): self.device = 'cuda'
        else: self.device = 'cpu'

        self.numTraining = args['numTraining']
        la.ReinforcementAgent.__init__(self, epsilon=self.epsilon, alpha=self.alpha, gamma=self.gamma, numTraining=self.numTraining)
        self.it = 0
        self.policy_dqn = DQNet.DQN(args['height'], args['width']).to(self.device)
        self.target_dqn = DQNet.DQN(args['height'], args['width']).to(self.device)
#        self.resume = args['resume']
        self.erm = DQNet.ERMemory()
        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=0.0005, betas=(0.5, 0.999)) #this was 0.0005
        self.ep=0
        self.target_dqn.eval()
        self.policy_dqn.train()
        self.lossfn = torch.nn.MSELoss().to(self.device)
        self.hepscores=[0]
        self.epscores=[]
        if self.numTraining == 0:
            self.epsilon=0
            model = torch.load('dqn_parameter_' + self.lo + '.pt', map_location=torch.device(self.device))
            self.policy_dqn.load_state_dict(model['policy'])
            self.target_dqn.load_state_dict(model['target'])

    def update(self, state, action, nextState, reward ):
        if len(self.erm.memory) < 32:
            return
        transitions = self.erm.batch(32)

        batch = Observation(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        nextstate_batch = torch.stack(batch.nextState).to(self.device)
        action_batch = torch.tensor(batch.action).to(self.device)
        reward_batch = torch.tensor(batch.reward).to(self.device)

        self.it +=1
        action_batch = action_batch.long()
        action_batch = action_batch.view(-1, 1)
        qvals = self.policy_dqn(state_batch).gather(1, action_batch)

        # Get value of next states.
        nextStateVals = self.target_dqn(nextstate_batch).max(1)[0].detach()

       # Get expected Q values
        expectedQVals = ((nextStateVals) * self.gamma) + reward_batch

        loss = self.lossfn(qvals, expectedQVals.unsqueeze(1))
#        if self.it % 300 == 0:
          #  print("eqval: ", expectedQVals.unsqueeze(1))
           # print("qval: ", qvals)
           # print("n: ", normalizer)
 #           print("loss: ", loss)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        #return None

    def getAction(self, state):
        action = None
        legalActions = self.getLegalActions(state)
        qvals = []
        if legalActions == []: action
        if util.flipCoin(self.epsilon): action = random.choice(legalActions)
        else:
            for a in legalActions:
                #qvals.append([a, self.getQValue(state, a)])
                qvals.append(self.getQValue(state, a))
            action = legalActions[np.argmax(np.array(qvals))]
            #action = legalActions[np.argmax(np.array(qvals))]
        self.doAction(state, action)
        return action


    def getQValue(self, state, action):
        numAct = 0

        if action == 'South': numAct = 1
        elif action == 'East': numAct = 2
        elif action == 'West': numAct = 3
        elif action == 'Stop': numAct = 4

        stateTensor = self.getStateTensor(state)
        ss = torch.empty(1, 6, self.height, self.width).to(self.device)
        ss[0] = stateTensor
        qtable = self.target_dqn.forward(ss)


        return float(qtable[0, numAct])


    def observeTransition(self, state, action, nextState, deltaReward):
        """Called during state transitions
            -Update total epsiode reward / done
	    -Add transition to memory /done
	    -Train the network when necessary"""
        #Update rewards
        self.episodeRewards += deltaReward
        #Add transition to memory
        if not (action == None):
            numAct = 0

            if action == 'South': numAct = 1
            elif action == 'East': numAct = 2
            elif action == 'West': numAct = 3
            elif action == 'Stop': numAct = 4

            s1 = self.getStateTensor(state); s2 = self.getStateTensor(nextState)
            self.erm.observe(s1, numAct, s2, deltaReward)

        self.update(state, action, nextState, deltaReward)



    def getStateTensor(self, state):
        """Turns states into 6 channel one hot tensors:
		C1: Pacman Location
		C2: Scared Ghost Location
		C3: Unscared Ghost Location -Merge these two channels for now
		C4: Food Location
		C3: Capsule Location
		C6: Wall Location """
    ##turns states into 6 channel one hot images:
     ##0.pacman location, 1.scared ghosts, 2.unscared ghosts, 3.foods, 4.capsules, 5.walls
        pPos = state.getPacmanPosition() ##Pacman's position

        gsPos = [] ; gusPos = []					  ##List of ghost positions
                                                          ##Separate scared ghosts from unscared ones later. currently 5 channel
        for i in range(1, state.getNumAgents()):
            if state.data.agentStates[i].scaredTimer > 0:
                gsPos.append(state.getGhostPosition(i))
            else:
                gusPos.append(state.getGhostPosition(i))

        cPos = state.getCapsules()		  ##Grid of booleans for capsule positions 
        fPos = state.getFood()		  ##Grid of booleans for food positions 
        fPos = np.array(fPos[:]).T
        wPos = state.getWalls()		  ##Grid of booleans for wall positions
        wPos = np.array(wPos[:]).T

        stateGrid = torch.zeros(6 ,self.height, self.width)
        stateGrid += 0.1
        stateGrid[0, (self.height-pPos[1]-1), pPos[0]] = 1

        for i in gusPos:
            stateGrid[1,(int(self.height)-int(i[1])-1), (int(i[0]))] = 1
        for i in gsPos:
            stateGrid[2,(int(self.height)-int(i[1])-1), (int(i[0]))] = 1

        for cap in cPos:
            stateGrid[3, (int(self.height)-int(cap[1])-1), (int(cap[0]))] = 1
        stateGrid[4, fPos.tolist()] = 1
        stateGrid[5, wPos.tolist()] = 1
        stateGrid=stateGrid.to(self.device)

        return stateGrid

    def final(self, state):
        #self.episodeRewards += self.last_reward
        #print("state: ", self.lastState)
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Do observation
   #     print("episode ", ep, "complete!!")
     #   PacmanQAgent.final(self, state)

   #     ep+=1
     #   if ep % 10 == 0:
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()
        #if self.episodesSoFar % 10 == 0:
          #  self.10epscores.append(state.getScore())
            #self.epscores[len(self.epscores)-1] = self.epscores[len(self.epscores)-1]/NUM_EPS_UPDATE
            #self.epscores.append(state.getScore())

        NUM_EPS_UPDATE = 100
        self.epscores.append(state.getScore())
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            self.hepscores[len(self.hepscores)-1] =  self.hepscores[len(self.hepscores)-1]/NUM_EPS_UPDATE
            self.hepscores.append(state.getScore())
 #           self.setEpsilon(self.epsilon*0.98)
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (
                        trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg))
            print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()
        else:
            self.hepscores[len(self.hepscores)-1] += state.getScore()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            self.setEpsilon(0)
            print('%s\n%s' % (msg,'-' * len(msg)))
            plt.plot([i for i in range(self.numTraining)], self.epscores)
            plt.plot([i*100 for i in range(int(self.numTraining/100) +1)], self.hepscores)
            plt.xlabel('Episode number')
            plt.ylabel('Score')
            plt.title('Performance over episodes')
            plt.savefig('DQL_perf_' + self.lo + '.png')
            if not self.numTraining == 0:
                model={}
                model['policy'] = self.policy_dqn.state_dict()
                model['target'] = self.target_dqn.state_dict()
                torch.save(model, 'dqn_parameter_' + self.lo + '.pt')

