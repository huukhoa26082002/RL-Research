import os
import shutil
from pathlib import Path
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym

from stable_baselines3.common.results_plotter import load_results


def load_xy(monitor_dir, num_timesteps=None):

  df = load_results(monitor_dir)
  df.drop('t', inplace=True, axis=1)
  df.drop('index', inplace=True, axis=1)

  df['l'] = df.l.cumsum()
  if num_timesteps != None:
    df = df[df.l <= num_timesteps]
  
  x = df.l.values
  y = df.r.values
  return x, y

def env_plot(env_id, monitor_dirs_of_algos, title=True, num_timesteps=1e6, alpha=0.5):

  for algo, monitor_dirs in monitor_dirs_of_algos.items():

    lines = []
    for monitor_dir in monitor_dirs:
      assert os.path.exists(monitor_dir), f'{monitor_dir} is not exist'
      assert len(os.listdir(monitor_dir)) != 0, f'{monitor_dir} have no *.monitor.csv files'
      x, y = load_xy(monitor_dir)
      assert x[-1] >= num_timesteps, f'trained timesteps: {x[-1]} < num_timesteps: {num_timesteps}'    
      line = (x, y)
      lines.append(line)

    
    list_x, mean_y, lower_y, upper_y = line_distribution(lines, num_timesteps)

    plt.plot(list_x, mean_y, label=algo)
    plt.fill_between(list_x, lower_y, upper_y, alpha=alpha)

  if title:
    plt.title(env_id)
  plt.xlabel('Timesteps')
  plt.ylabel('Average Reward')
  plt.legend()
  plt.show()

def estimate_y(x, prev_point, next_point):
  y = prev_point[1] + (prev_point[1] - next_point[1]) * (x - prev_point[0]) / (prev_point[0] - next_point[0])
  return y

def line_distribution(lines, num_timesteps):

  n = len(lines)
  max_timestep = num_timesteps
  list_x = []
  mean_y = []
  lower_y = []
  upper_y = []

  min_timestep = max_timestep
  for line_index in range(n):
    line = lines[line_index]
    x = line[0]
    timestep0 = x[0]
    if min_timestep > timestep0:
      min_timestep = timestep0

  point_indexes = [0] * n
  timestep = min_timestep
  for line_index in range(n):
    line = lines[line_index]
    x = line[0]
    i = point_indexes[line_index]
    while x[i] < timestep:
      point_indexes[line_index] += 1
      i += 1

  while True:
    mu = 0
    for line_index in range(n):

      line = lines[line_index]
      x = line[0]
      y = line[1]
      i = point_indexes[line_index]

      if timestep < x[i]:
        yi = estimate_y(timestep, (x[i-1], y[i-1]), (x[i], y[i]))
      elif timestep == x[i]:
        yi = y[i]
      else:
        assert False, 'why timestep > x[i]?'

      mu += yi
    mu /= n
    
    sigma = 0
    for line_index in range(n):

      line = lines[line_index]
      x = line[0]
      y = line[1]
      i = point_indexes[line_index]

      if timestep < x[i]:
        yi = estimate_y(timestep, (x[i-1], y[i-1]), (x[i], y[i]))
      elif timestep == x[i]:
        yi = y[i]
      else:
        assert False, 'why timestep > x[i]?'

      sigma += abs(yi - mu)
    sigma /= n**0.5

    list_x.append(timestep)
    mean_y.append(mu)
    lower_y.append(mu - sigma)
    upper_y.append(mu + sigma)

    if timestep == max_timestep:
      break

    new_timestep = max_timestep
    for line_index in range(n):

      line = lines[line_index]
      x = line[0]
      i = point_indexes[line_index]
      
      if x[i] == timestep:
        assert len(x) != i+1, f'trained timesteps: {x[i]} < num_timesteps: {num_timesteps}'
        point_indexes[line_index] += 1
        i += 1
      elif x[i] < timestep:
        assert False, 'why x[i] < timestep'

      new_timestep = min(new_timestep, x[i])
    timestep = new_timestep

  return list_x, mean_y, lower_y, upper_y

def one_line_plot(monitor_dir, title):
    """
    plot the results

    :param monitor_dir: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = load_xy(monitor_dir)
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title(title)
    plt.show()

def record_video(model, env_id, video_dir, play=True, deterministic=True):
  from colabgymrender.recorder import Recorder
  
  env = gym.make(env_id)
  env = Recorder(env, video_dir)

  episode_rewards = 0
  done = False
  
  obs = env.reset()
  while not done:
      action, _states = model.predict(obs, deterministic=deterministic)
      obs, reward, done, info = env.step(action)
      episode_rewards += reward

  if play:
    env.play()

