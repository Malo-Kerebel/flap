import tensorflow as tf
import numpy as np
#from tf_agents.environments import py_environment
#from tf_agents.environments import tf_environment
#from tf_agents.environments import tf_py_environment
#from tf_agents.environments import utils
#from tf_agents.specs import array_spec
#from tf_agents.environments import wrappers
#from tf_agents.environments import suite_gym
#from tf_agents.trajectories import time_step as ts

import matplotlib.pyplot as plt
from matplotlib import animation

import tkinter as tk

from datetime import datetime

class flappy:

    def __init__(self, length):
        self.N_action = 2
        self.max_y = 100
        self.max_dy = 2.55/2

        self.length = length

        self.pipe = pipe()
        
        self.x = 0
        self.y = 50
        self.dy = 0

    def _state(self):
        if self.y > self.max_y:
            return self.max_y, int(self.dy), self.pipe.y-h_pipe
        
        return int(self.y), int(self.dy), self.pipe.y-h_pipe

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def reset(self):
        # state at the start of the game
        self.x = 0
        self.y = 50
        self.dy = 0
        return self._state()

    def step(self, action):
        done = False
        reward = 0

        if self.y > 0 and self.y < self.max_y:
            self.x += 1

            if self.x > self.length:
                done = True
                reward += 100
                return self._state(), reward, done
            
            self.dy -= gravity/FPS
            if self.dy < -3*self.max_dy:
                self.dy = -self.max_dy

            if (self.pipe.x == 0) and (self.y < self.pipe.y - h_pipe/2 or self.y > self.pipe.y + h_pipe/2):
                done = True
                reward -= 100
                # print("DEATH")
                self.pipe = pipe()
                return self._state(), reward, done


            if (self.pipe.x == 0):
                self.pipe = pipe()
                # print("Got through")
                reward += 5
            else:
                self.pipe.move()
                
            if action == 1:
                self.dy = self.max_dy  # Speed for which we get an increase of 10 for a gravity of 9.8 with 30 FPS
                reward -= 0.35
                if self.y > 80:
                    reward -= 1
                
            self.y += self.dy

            # if self.x % log_interval == 0:
            #     print(f"Position : x = {self.x}, y = {self.y}, dy = {self.dy}")
            
        else:
            done = True
            reward -= 100

        return self._state(), reward, done


class pipe(object):

    def __init__(self):
        self.y = np.random.randint(20, 80)
        self.x = pipe_interval

    def move(self):
        self.x -= 1

        
def train():

    N_episode = 5000
    exploration_rate = 0.1
    learning_rate = 0.1
    discount_factor = 0.9

    env = flappy(1000)
    q_values = np.zeros((env.max_y+1, 4, 60+1, env.N_action))

    t1 = datetime.now()

    times = []
    episodes = []

    for i in range(N_episode):
        if (i+1) % (N_episode//100) == 0:
            t2 = datetime.now()
            print(f"Episode {i+1}/{N_episode}, in {t2-t1}")
            episodes.append(i//100)
            times.append((t2-t1).total_seconds()/(N_episode//100))
            t1 = t2

        state = env.reset()
        done = False

        while not done:
            if np.random.random() > exploration_rate:
                action = np.argmax(q_values[state])
            else:
                action = np.random.choice(env.N_action)

            next_state, reward, done = env.step(action)

            target = reward + discount_factor * np.max(q_values[next_state])
            error = target - q_values[state][action]
            q_values[state][action] += learning_rate * error
            # Update state
            state = next_state

    # print(q_values)

    env = flappy(5000)

    state = env.reset()
    done = False

    y = []
    dy = []

    x = []
    y_pipe = []

    while not done:
        action = np.argmax(q_values[state])

        next_state, reward, done = env.step(action)

        y.append(env.y)
        dy.append(env.dy)
        
        x.append(env.pipe.x)
        y_pipe.append(env.pipe.y)

        state = next_state

    # plt.figure("Episodes vs times")
    # plt.plot(episodes, times)
    # plt.title(f"Average time to execute {N_episode//100} episodes")
    # plt.xlabel(f"bins of {N_episode//100} episode")
    # plt.ylabel("time")

    # plt.savefig("Episodes vs times")
    # plt.show()

    fig, ax = plt.subplots(1,2, gridspec_kw = {'width_ratios':(3,2)}, num = 'Flappy',
                          clear = True)


    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-3, 3)
    
    ax[0].set_xlim(-10, pipe_interval)
    ax[0].set_ylim(0, 100)
    
    bird, = ax[0].plot(0, y[0], "or")
    pipe1, = ax[0].plot([180, 180], [100, 100], "g")
    pipe2, = ax[0].plot([180, 180], [0, 0], "g")
    speed, = ax[1].plot(0, dy[0], "ob")

    def step(n):
        bird.set_data(0, y[int(n/a)])
        speed.set_data(0, dy[int(n/a)])
        pipe1.set_data([x[int(n/a)], x[int(n/a)]], [100, y_pipe[int(n/a)]+h_pipe/2])
        pipe2.set_data([x[int(n/a)], x[int(n/a)]], [0, y_pipe[int(n/a)]-h_pipe/2])
    
    my_anim = animation.FuncAnimation(fig, step, frames=int(len(y)*a))
    plt.show(block=False)
    # plt.draw()
    my_anim.save("flappy_train.gif", writer='imagemagick', fps=20)

    # print(steps)
    # print(rewards)
    # plt.plot(range(len(rewards)), rewards)
    # plt.show()
        
        
# bgcolour = "#55AACC"

# # configure workspace
# ws = tk.Tk()
# ws.title("Family Feud")
# length = 1080
# height = 720
# ws.geometry(str(length) + 'x' + str(height))
# ws.configure(bg=bgcolour)

gravity = 9.8
FPS = 30

h_pipe = 20
pipe_interval = 3*60

log_interval = 100

a = 1/4

# python_environment = flappy()
# tf_env = tf_py_environment.TFPyEnvironment(python_environment)

# title = tk.Label(ws, text="Flappy train", font=("Arial", 40), bg=bgcolour)
# title.place(x=length // 2, y=height // 3, anchor="center")

# sta = tk.Button(ws, text="Démarrer", command=lambda: train())
# sta.place(x=length // 2, y=2 * height // 3, anchor="center")

# infinite loop
# ws.mainloop()
train()
