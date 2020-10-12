from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint
from util import manhattanDistance
import util, layout
import sys, types, time, random, os
from pacman import runGames
from pacman import readCommand

if __name__ == '__main__':

#    print("---------------------BASIC: Q-LEARNING WITH EPSILON GREEDY POLICY-------------------\n")
#    print("-Experiment 1: Small grid layout with default parameters\n")
#    print("-Parameters: Epsilon:0.05	| Learning rate: 0.2	| Discount: 0.8\n") 
#    args = readCommand(['-p', 'PacmanQAgent', '-x', '2000', '-n', '2010', '-l', 'smallGrid', '-g', 'DirectionalGhost'])
#    runGames(**args)

    print("\n-Experiment 2: Small grid layout with more explorative pacman\n")
    print("-Parameters: Epsilon:0.1	| Learning rate: 0.2	| Discount: 0.8\n") 
    args = readCommand(['-p', 'PacmanQAgent', '-x', '2000', '-n', '2010', '-l', 'smallGrid', '-g', 'DirectionalGhost', '-a', 'epsilon=0.1'])
    runGames(**args)

    print("\n-Experiment 3: Small grid layout with faster learning pacman\n")
    print("-Parameters: Epsilon:0.05	| Learning rate: 0.4	| Discount: 0.8\n") 
    args = readCommand(['-p', 'PacmanQAgent', '-x', '2000', '-n', '2010', '-l', 'smallGrid', '-g', 'DirectionalGhost', '-a', 'alpha=0.4'])
    runGames(**args)

    print("\n-Experiment 4: Small grid layout with more nearsighted pacman\n")
    print("-Parameters: Epsilon:0.05	| Learning rate: 0.2	| Discount: 0.5\n") 
    args = readCommand(['-p', 'PacmanQAgent', '-x', '2000', '-n', '2010', '-l', 'smallGrid', '-g', 'DirectionalGhost', '-a', 'gamma=0.5'])
    runGames(**args)

    print("\n-Experiment 5: Small grid layout with completely farsighted pacman\n")
    print("-Parameters: Epsilon:0.05	| Learning rate: 0.2	| Discount: 1\n") 
    args = readCommand(['-p', 'PacmanQAgent', '-x', '2000', '-n', '2010', '-l', 'smallGrid', '-g', 'DirectionalGhost', '-a', 'gamma=1.0'])
    runGames(**args)

    print("---------------------BASIC: APPROXIMATE Q-LEARNING WITH DEFAULT FEATURE EXTRACTOR-------------------\n")

