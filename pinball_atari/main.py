import sys
import gymnasium as gym
import ale_py
import shimmy
#print("Python:", sys.executable)
#print("Gymnasium path:", sys.modules['gymnasium'].__file__)

from train import train

if __name__ == "__main__":
    train()
