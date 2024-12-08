import retro
import numpy as np
import cv2
import neat
import pickle

# Inicializa o ambiente do jogo usando a biblioteca Retro.
env = retro.make(game = 'SuperMarioWorld-Snes', state = 'YoshiIsland2', players = 1)

# Lista que armazenará os dados da imagem processada.
img_array = []

# Carrega as configurações da rede neural do arquivo de configuração.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')

# Carrega o melhor genoma salvo anteriormente (treinado) a partir do arquivo 'best.pkl'.
with open('best.pkl', 'rb') as file:
      genome = pickle.load(file)

# Cria a rede neural a partir do genoma carregado.
network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

# Reinicia o ambiente e obtém a imagem inicial do jogo.
ob = env.reset()

# Obtém as dimensões da tela do jogo e reduz a resolução para simplificar os dados processados.
inx, iny, inc = env.observation_space.shape
inx = int(inx/8)
iny = int(iny/8)

done = False
# Loop principal que renderiza e executa o jogo utilizando o melhor genoma.
while not done:
      env.render()

      # Redimensiona a imagem do jogo para a resolução reduzida e converte para tons de cinza.
      ob = cv2.resize(ob, (inx, iny))
      ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
      ob = np.reshape(ob, (inx,iny))
      img_array = np.ndarray.flatten(ob)

      # A rede neural processa os dados da imagem e retorna a melhor ação para o agente realizar.
      nn_output = network.activate(img_array)

      # Aplica a ação no jogo e recebe o próximo estado e informações adicionais.
      ob, reward, done, info = env.step(nn_output)
