import retro
import numpy as np
import cv2
import neat
import re
import os
import pickle
import sys
from rominfo import *

def getRam(env):
    ram = []
    for k, v in env.data.memory.blocks.items():
        ram += list(v)
    return np.array(ram)
    
# Classe que contém o método work responsável por executar o treinamento, dessa forma é possível ter vários "workers" para paralelilzar o treino.
class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    # Treinamento do agente     
    def work(self):
        arguments = sys.argv[1:]

        self.env = retro.make(game = 'SuperMarioWorld-Snes', state = 'YoshiIsland2', players = 1)

        # Inicia o ambiente do 0 e retorna a imagem observavel
        ob = self.env.reset()

        # Obtem a resolucao do jogo e divide por 8 para diminuir os inputs da rede neural
        inx, iny, _ = self.env.observation_space.shape
        inx = int(inx/8)
        iny = int(iny/8)

        # Cria a rede
        network = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)

        fitness = 0
        counter = 0
        xpos = 0
        xpos_max = 0
        score = 0
        max_score = 0

        done = False
        while not done:
            if "SHOW_GAME" in arguments:
                # Abre uma janela e renderiza o frame do jogo
                self.env.render()

            # Reduzindo a imagem e as cores para uma escala de cinza para que hajam menos inputs na rede
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            cv2.waitKey(1)
            img_array = np.ndarray.flatten(ob)

            # Informacoes da tela como input para rede neural
            nn_output = network.activate(img_array)

            # Output da rede eh usado como input de um frame no jogo
            ob, _, done, info = self.env.step(nn_output)

            # Salva score do jogo
            score = info['score']
            if score > max_score:
                max_score = score

            # Verifica se a fase acabou ou se Mario parou de evoluir
            _, xpos, _ = getInputs(getRam(self.env))
            if xpos > xpos_max:
                xpos_max = xpos
                counter = 0
            else:
                counter += 1
            if done or counter == 200:
                done = True
                print("Genoma: " + str(self.genome.key) + ", Fitness: " + str(fitness) + ", Posição Final X: " + str(xpos) + ", Score Final: " + str(max_score))

            # 3 pontos para cada pixel X, 1 ponto para cada ponto de score
            fitness = xpos * 3 + max_score

            # 5000 é um pouco após o final da fase
            if xpos_max > 5000:
                fitness = 100000 + max_score

        if "SHOW_GAME" in arguments:
            self.env.render(close=True)

        self.genome.fitness = int(fitness)
        return int(fitness)

# A função de treinamento do agente que o neat espera receber, cria um novo worker para avaliar o genoma recebido
def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()
    
def main():
    # Carrega configurações da rede neural
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')

    # Verifica se há um checkpoint de treino, se não, começa uma população do 0
    if os.path.exists('checkpoint'):
        p = neat.Checkpointer.restore_checkpoint('checkpoint')
    else:
        p = neat.Population(config)

    # Adiciona informações estatísticas à saída no console
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    print("Para parar o treinamento aperte CTRL-C após o checkpoint de uma geração ser salvo")

    while True:
        # Faz a paralelização da execução dos genomas para agilizar o treinamento, quanto maior o número de workers (primeiro parâmetro) mais processos serão
        # criados e portanto haverá um maior consumo de RAM e CPU
        pe = neat.ParallelEvaluator(10, eval_genomes)

        # Roda a função de treino da população e retorna o melhor genoma, caso o programa seja interrompido o checkpoint será a ultima geração completa. O 
        # segundo parametro representa o número de gerações até que o programa encerre
        winner = p.run(pe.evaluate, 1)

        # Salva o melhor genoma
        with open('best.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)

        # Neat salva os checkpoints como neat-checkpoint-numerodageracao, deixando o ambiente cheio de arquivos, não há (ou não encontrei)
        # uma forma de salvar apenas a última geração com um nome escolhido, então fiz essa forma de apagar as demais gerações e renomear a última para checkpoint
        # coloquei esse código em um loop, pois foi a melhor forma de poder cancelar o treinamento salvando os dados e sem aumentar a complexidade do programa
        checkpoint_files = [filename for filename in os.listdir() if re.match(r'neat-checkpoint-\d+', filename)]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key = lambda x: int(re.search(r'\d+', x).group()))
            for checkpoint_file in checkpoint_files:
                if checkpoint_file != latest_checkpoint:
                    os.remove(checkpoint_file)
                else:
                    os.rename(checkpoint_file, 'checkpoint')
        print("Checkpoint salvo! Aperte CTRL-C para sair")

if __name__ == "__main__":
    main()