# RL_driving
Autonomous driving with deep reinforcement learning.  

## To-Dos


## Questions
* Is using an autoencoder to construct representations really a good idea?  
    Generally we like the approach as it allows us to decouple the perception task from the model
    Also MujoCo environments seem to represent states as vectors -> algorithms are trained on vector input
    Further questions:
    How can we ensure the lanes are sufficiently represented in lower dim state?
    Can this possibly degrade the performance of our model?
* What do you think about model-based approaches?  
    First: Do we understand this correctly?
    Advantage: Reduced sample complexity
    Can this work in a complex environment like CARLA
