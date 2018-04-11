# SuperBot
A repo with experiments on control using supervised learning (uses PyTorch, numpy, and pygame)


## Structure 

* Grid is a module holding some classes necessary for A* pathfinding 
* Grid Reacher is a modification of my Reacher environnement. Once the env is defined, it is possible to use the `step_astar()` method to control the robot using the A* algorithm 
* GridData is used for data generation and creating DataLoader structure 
* GridLearner is the supervised model. Can be trained, can display loss at the end of training. Can also use dropout and has a method for test environnemnts (`demo()`) 
* Finally, GridWrapper is a wrapper, sufficient for using the whole chain and setting up testing. 
