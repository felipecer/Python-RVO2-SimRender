#!/usr/bin/env python
from rl_environments.single_agent.simple_v6 import RVOSimulationEnv
import sys
import select
import termios
import tty

KEYBINDINGS = {
    'w': (0, 1),
    'a': (-1, 0),
    's': (0, -1),
    'd': (1, 0),
    'q': (-1, 1),
    'e': (1, 1),
    'z': (-1, -1),
    'c': (1, -1),
}

STEP_SIZE = 0.5


def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 3)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    env = RVOSimulationEnv(
        './simulator/worlds/simple.yaml', render_mode='rgb')
    observations = env.reset()
    done = False
    i = 0
    action = (0, 0)
    while not done:
        # accumulate reward
        cumulative_reward = []
        key = getKey()
        
        if key in KEYBINDINGS.keys():
            new_action = KEYBINDINGS[key]
            action = list(action)
            action[0] += new_action[0] * STEP_SIZE
            action[1] += new_action[1] * STEP_SIZE
            action = tuple(action)
            print(f"Action: {action}")
            observations, reward, done, truncated, info = env.step(action)
        else:
            if key == '\x03':
                break
            
            print(f"Action: {action}")
            observations, reward, done, truncated, info = env.step(action)

        if done or truncated:
            print(f"Episode done: {done}, truncated: {truncated}")
            cumulative_reward.append((i, reward))
            break
        print(f"Step {i} reward: {reward}")
        i += 1
