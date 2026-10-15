CPSC4420 Assignment 2: Q-Learning in Gym's Blackjack Environment
Tabular Q-learning agent for Gym's Blackjack-v1, with hyperparameter experiments on learning rate, discount factor, and exploration decay
Environment Description:
1. Env: Blackjack-v1 with sab=True(Sutton & Barto rules)
2. Goal: maximize expected reward per hand by choosing when to hit or stick
3. Observation: tuple (player_sum, dealer_card, usable_ace). It is already discrete (32 x 11 x 2), so no discretization is used
4. Actions: 0=stick, 1=hit
5. Rewards: +1=win, 0=draw, and -1=loss
6. Termination: the player busts or sticks
7. Dynamics: stochastic (random card draws). Because of the house edge, even a good policy has a slightly negative expected reward per hand



