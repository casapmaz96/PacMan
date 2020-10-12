PACMAN
Cansin Aysegul Sapmaz

### CONTRIBUTION ###

Files that contain my reinforcement learning contributions:
	-qlearningAgents.py: 
		Code for simple q-learning algorithm, update of the q-table, calculating expected values,
		picking an action based on epsilon greedy.

	-DQNet.py:
		Code fot the deep q-learning network. Implemented using the official Pytorch tutorial on
		deep learning, found here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

	-dqlearningAgents.py:
		Code for deep q-learning training/testing algorithm. Integration to Berkeley Pacman Projects.


### HOW TO RUN ###

In general:
	python pacman.py -p <PacmanType> -l <layoutName> -x <NumberOfTrainingGames> -n <NumberofTotalGames>
		-q (optional) -t (optional)

	<PacmanType> can be PacmanQAgent for simple Q-learning, or DQLAgent for deep Q-learning
	<layoutName> can be smallGrid, mediumGrid or smallClassic
	<Number of training games> should be an integer, determines how many silent games to play for training
	<Number of total games> shouls be an integer equal to or greater than number of training games. The 
	difference between total games and training games is the number of test games.
	If -q is written, all games (including test games) are played in silent mode without graphics. Only scores are
	printed. Suitable for using on Camber.
	If -t is written, test games are printed as console text instead of graphics. Suitable for using on Camber.
	If -q or -t is not written, test games are played with graphics display. Not suitable for Camber.

	Example:
		python pacman.py -p PacmanQAgent -l smallGrid -x 2000 -n 2010 : Plays 2000 silent training games, then 
		10 test games with graphics display.

		python pacman.py -p DQLAgent -l smallGrid -x 0 -n 10 : Doesn't train. Uses saved model parameters on the
		same directory to play 10 test games with graphics display. (Please do not change default parameter file names or directories,
		the code saves the models for each layout with appropriate name. Models have to be in the same directory as the code.) 

On camber: 
	simple-sg-script.sh : Trains simple Q-learning agent on small grid for 3000 games, then plays 20 test games 
	silently.
	simple-mg-script.sh: Trains simple Q-learning agent on medium grid for 5000 games, then plays 20 test games 
	silently.

	dql-sg-test.sh: Uses saved model parameters to play 20 test games on small grid layout with deep Q-learning agent
	silently.
	dql-mg-test.sh: Uses saved model parameters to play 20 test games on medium grid with deep Q-learning agent silently.
	dql-sc-test.sh: Uses saved model parameters to play 20 test games on small classic with deep Q-learning agent silently.

	dql-sg-train.sh: Trains deep Q-learning agent on small grid for 3000 games, plays 10 test games silently. Takes minutes. 
	dql-mg-train.sh: Trains deep Q-learning agent on medium grid for 5000 games, plays 10 test games silently. Does not take long.
	dql-sc-train.sh: Trains deep Q-learning agent on small classic for 15000 games, plays 10 test games silently. Takes 10 hours!

Important: Training the DQLAgent overwrites saved model parameters in the same directory.

### REFERENCES ###

This code uses Berkeley University's Pacman Projects: http://ai.berkeley.edu/reinforcement.html
Deep Q-learning is implemented with the help of the official Pytorch tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
