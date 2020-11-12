# Abstract

# Objectives
Autonomous vehicles (AVs) are expected to dramatically redefine the future of transportation. However, there are still significant engineering challenges to be solved before one can fully realize the benefits of self-driving cars. One such challenge is building models that reliably predict the movement of traffic agents around the AV, such as cars, cyclists, and pedestrians. The ridesharing company Lyft released a challenge to predict the motion of traffic agents (e.g. pedestrians, vehicles, bikes, etc.). In this competition, you'll have access to the largest Prediction Dataset ever released to train and test your models.
The goal of this shared code is to predict the trajectories of traffic agents. Amulti-modal ones generating three hypotheses (mode of transportation).
## Output

# Input



# Method
## Model Archeticture
## Optimizer
## Input shape
## Loss function
We calculate the negative log-likelihood of the ground truth data given the multi-modal predictions. Let us take a closer look at this. Assume, ground truth positions of a sample trajectory are



# Suggestions / Next Steps

# Limitations
