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

Setup:
git clone https://github.com/andydidib/CPSC4420-Assignment-2.git
cd REPO-LOCATION
python -m venv .venv
Window: .venv\Scripts\activate 
Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

How to Run:
python agent.py
python baseline.py

Files:
1. agent.py: Q-learning agent and training loop
2. baseline.py baseline configuration and reward plot
3. results/ plots and the MP4 of the trained agent
4. requirements.txt: pinned dependencies

Baseline Configuration:
alpha=0.1, gamma=1.0, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99995
