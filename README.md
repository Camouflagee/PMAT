# Prioritized Multi-Agent Transformer

This is the implementation of **Prioritized Multi-Agent Transformer (PMAT)**. Built on Multi-Agent Transformer, PMAT is a novel sequential decision-making multi-agent reinforcement learning (MARL) algorithm that pays close attention to *action generation order optimization* in the MARL domain.

## Installation

### Dependencies

```
pip install -r requirements.txt
```

### Environments

#### StarCraft II Multi-Agent Challenge (SMAC)

Just follow the instructions in https://github.com/oxwhirl/smac to setup a SMAC environment.

#### Google Research Football (GRF)

Just follow the instructions in https://github.com/google-research/football to setup a GRF environment.

#### Multi-agent MuJoCo (MA MuJoCo)

Just follow the instructions in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a MA MuJoCo environment.

## How to run

When the environments are ready, you can run shells in the "scripts" folder with algo="pmat". Specifically:

```
./train_smac.sh # run PMAT on SMAC
```
```
./train_football.sh # run PMAT on GRF
```
```
./train_mujoco.sh # run PMAT on MA MuJoCo
```
