This repo contains the code for the paper "Reinforcement Learning for Dynamic Memory Allocation" - https://www.overleaf.com/read/grcgrdrwxhyy#45e67d

# Installation

1. Clone the repository: git clone https://github.com/curry2736/rl-malloc.git
2. Install the required packages: pip install -r requirements.txt


# Running the Code

## Experiment 1

To run experiment 1 from the paper, execute the following command: python experiment1.py

If you would like to retrain from scratch, follow these steps:
1. Comment out line 24 in `experiment1.py`, which loads a previously trained model.
2. Uncomment lines 31 and 32 in `experiment1.py` for training.

## Experiments 2 and 3

To run experiments 2 and 3 from the paper, open `test_pol_sb3.ipynb`.

To get the ith graph of experiment j:
1. Uncomment the 6 lines of code in the second cell which have the experiment i graph j commented above it.

Run the full notebook to get each individual graph
