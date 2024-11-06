import retro
import numpy as np
import cv2 
import neat
import pickle

env = retro.make(game = 'SuperMarioWorld-Snes', state = 'YoshiIsland2', players = 1)

img_array = []

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')

with open('best.pkl', 'rb') as file:
      genome = pickle.load(file)

network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

ob = env.reset()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)


done = False
while not done:
      env.render()

      ob = cv2.resize(ob, (inx, iny))
      ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
      ob = np.reshape(ob, (inx,iny))
      img_array = np.ndarray.flatten(ob)

      nn_output = network.activate(img_array)

      ob, reward, done, info = env.step(nn_output)
