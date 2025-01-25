---
layout: default
title: Proposal
---

## Summary of the Project
Our project will focus on the world famous gambling card game, poker, more specifically no-limit, heads-up (2 player) Texas Hold ‘Em Poker. We plan on leveraging an existing Python poker library to help us create our state space. Inputs to the AI will be the hand dealt, the opponent’s actions, and both players’ chip count. The AI will output one of the three legal actions in Poker: call, raise, and fold. The goal is to create an AI that will optimally play the hand dealt to it, and that will bankrupt/defeat the opponent as often as possible.

## AI/ML Algorithms

We anticipate using the proximal policy optimization (PPO) and neural fictitious self-play (NFSP) reinforcement learning algorithms to approximate the Nash equilibrium for zero-sum games, essentially training our AI by making it play against itself.

Resources:

https://thegradient.pub/libratus-poker/

https://www.science.org/doi/10.1126/science.aao1733

## Evaluation Plan

We’ll be evaluating the AI’s performance by measuring its win/loss rate against humans; i.e., our team members with varying skill levels in poker. As a baseline, we’d like to achieve above 50% win rate against humans, and continually improve the performance to at least 60% win rate against humans. Another metric we may evaluate on is the amount of rounds the games last, with more rounds being better if it loses, and less rounds being better if it wins.

We will visualize our algorithm by “plugging it in” to a Python poker GUI, and playing against it. To smoke test our bot we can employ certain toy problems such as giving it the best hand in poker, a royal flush, and seeing if it decides to raise, and giving it the worst hand in poker, a high card, and seeing if it decides to fold. Additionally, we’ll test the AI in more complex situations, including cases such as the opponent continuously raising no matter what, and cases where opponents act with more randomness that comes with human nature. Ultimately, what would be ideal is that our AI will consistently outperform humans - that is to say, achieve an 80%+ win rate against us.

## Meet the Instructor
The earliest meeting date for team All in N Ballin’ is in-person on Thursday, February 6 at 1:00pm (Week 5).

## AI Tool Usage
No AI tools were used in the creation of this proposal. Any utilization of AI tools in the future will be reported below:
