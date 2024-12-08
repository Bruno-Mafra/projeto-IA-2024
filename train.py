import retro
import numpy as np
import cv2
import neat
import re
import os
import pickle
import sys
from rominfo import *

# Obtém a memória RAM do ambiente Retro e a retorna como um array NumPy.
# Isso é usado para extrair informações específicas do estado do jogo.
def getRam(env):
    ram = []
    for k, v in env.data.memory.blocks.items():
        ram += list(v)
    return np.array(ram)

# Classe que representa um "worker" responsável por treinar um genoma específico.
# Cada worker executa o treinamento de forma isolada, permitindo paralelização.
class Worker(object):
    def __init__(self, genome, config):
        # Inicializa o worker com um genoma e as configurações da rede neural.
        self.genome = genome
        self.config = config

    # Executa o treinamento do agente para um genoma específico.
    def work(self):
        arguments = sys.argv[1:]

        # Inicializa o ambiente do jogo usando a biblioteca Retro.
        self.env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)

        # Inicia/reinicia o ambiente e obtém o estado inicial (imagem observável).
        ob = self.env.reset()

        # Reduz a resolução da tela em um fator de 8 para diminuir a quantidade de dados processados.
        inx, iny, _ = self.env.observation_space.shape
        inx = int(inx / 8)
        iny = int(iny / 8)

        # Cria a rede neural para o genoma usando as configurações fornecidas.
        network = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)

        # Inicializa variáveis para acompanhar desempenho e progresso do agente.
        fitness = 0
        counter = 0
        xpos = 0
        xpos_max = 0
        score = 0
        max_score = 0

        done = False
        while not done:
            if "SHOW_GAME" in arguments:
                # Renderiza o ambiente (mostra a janela do jogo).
                self.env.render()

            # Converte a imagem do jogo para tons de cinza e reduz sua resolução para simplificar os inputs.
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            cv2.waitKey(1)  # Aguarda brevemente para evitar bugs no render.
            img_array = np.ndarray.flatten(ob)

            # Usa os dados da tela como entrada para a rede neural.
            nn_output = network.activate(img_array)

            # Aplica a saída da rede como ação no jogo.
            ob, _, done, info = self.env.step(nn_output)

            # Atualiza a pontuação do jogo.
            score = info['score']
            if score > max_score:
                max_score = score

            # Verifica se o Mario está progredindo ou se está travado.
            _, xpos, _ = getInputs(getRam(self.env))
            if xpos > xpos_max:
                xpos_max = xpos
                counter = 0  # Reinicia o contador de inatividade.
            else:
                counter += 1

            # Encerrar o loop se o agente estiver inativo por muito tempo ou a fase acabar.
            if done or counter == 200:
                done = True
                print(f"Genoma: {self.genome.key}, Fitness: {fitness}, Posição Final X: {xpos}, Score Final: {max_score}")

            # Calcula o fitness com base na posição X e na pontuação.
            fitness = xpos * 3 + max_score

            # Premia com um valor de fitness muito alto se o Mario terminar a fase.
            if xpos_max > 5000:
                fitness = 100000 + max_score

        # Fecha a janela do jogo ao finalizar.
        if "SHOW_GAME" in arguments:
            self.env.render(close=True)

        # Salva o fitness final no genoma.
        self.genome.fitness = int(fitness)
        return int(fitness)

# Avalia o desempenho de um genoma criando um worker para treinar e testar.
def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()

def main():
    # Carrega as configurações para a rede neural do arquivo de configuração.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')

    # Verifica se há um checkpoint para retomar o treinamento.
    if os.path.exists('checkpoint'):
        p = neat.Checkpointer.restore_checkpoint('checkpoint')
    else:
        p = neat.Population(config)

    # Adiciona relatórios para monitorar o progresso do treinamento.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))  # Salva um checkpoint por geração.

    print("Para parar o treinamento aperte CTRL-C após o checkpoint de uma geração ser salvo")

    while True:
        # Paraleliza a avaliação de genomas para acelerar o treinamento.
        # O número de workers define quantos processos simultâneos são executados.
        pe = neat.ParallelEvaluator(10, eval_genomes)

        # Roda o treinamento por uma geração e obtém o melhor genoma.
        winner = p.run(pe.evaluate, 1)

        # Salva o melhor genoma em um arquivo pickle.
        with open('best.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)

        # Gerencia os arquivos de checkpoint, mantendo apenas o mais recente.
        checkpoint_files = [filename for filename in os.listdir() if re.match(r'neat-checkpoint-\d+', filename)]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.search(r'\d+', x).group()))
            for checkpoint_file in checkpoint_files:
                if checkpoint_file != latest_checkpoint:
                    os.remove(checkpoint_file)
                else:
                    os.rename(checkpoint_file, 'checkpoint')

        print("Checkpoint salvo! Aperte CTRL-C para sair")

if __name__ == "__main__":
    main()
