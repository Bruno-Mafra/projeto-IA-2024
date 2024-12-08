**Nome:** Ana Paula Sales **RA:** 11201811703

**Nome:** Bruno Francisco Rodrigues Mafra **RA:** 11201811147

**Nome:** Gabriel Gomes de Oliveira Costa **RA:** 11201921471

**Nome:** João Pedro Sousa Santos **RA:** 11202021260

**Nome:** Lucas Polo Chiang **RA:** 11201811096

**Nome:** Stephany Caroline Cavaleto Santanna **RA:** 11201920287 

# Projeto: Super Mario World - Inteligência Artificial (2024.Q3)

Este projeto utiliza o algoritmo **NEAT (NeuroEvolution of Augmenting Topologies)**, uma abordagem de otimização evolutiva para o treinamento de redes neurais, implementado com a biblioteca `neat-python`. O objetivo foi treinar um agente inteligente para completar uma fase do jogo **Super Mario World** de forma autônoma.

## Algoritmo NEAT

O NEAT simula o processo de evolução biológica. Ele começa com uma população de redes neurais simples, evoluindo ao longo de várias gerações. Durante o treinamento, alterações estruturais (como adicionar ou remover conexões e neurônios) são introduzidas, permitindo que a rede se adapte e melhore gradualmente. O resultado é um modelo otimizado para a tarefa específica, neste caso, avançar no jogo de forma eficiente.

## Sistema de Recompensas

Para guiar o aprendizado do agente, utilizamos uma função de recompensa que prioriza a distância percorrida no jogo e a pontuação alcançada:

`reward = (x_pos * 3) + max_score`

O fator multiplicador (3) aplicado à posição no eixo X enfatiza a importância de avançar na fase, enquanto o `max_score` reflete a pontuação acumulada. Com esse sistema, o agente conseguiu completar a fase após cerca de **12 gerações**, com uma população de **75 genomas por geração**.

## Dependências

As seguintes bibliotecas foram utilizadas no projeto:

- `neat-python`
- `opencv`
- `numpy`
- `pickle`

### Instalação
Execute os comandos abaixo para instalar as dependências:

```bash
pyenv install 3.8
pyenv shell 3.8
python -m venv marioenv
source marioenv/bin/activate
pip install gym==0.21.0
pip install gym-retro
pip install neat-python
pip install opencv-python
pip install pickle
pip install numpy
cp rom.sfc marioenv/lib/python3.8/site-packages/retro/data/stable/SuperMarioWorld-Snes/
python train.py SHOW_GAME
```

## Estrutura do Projeto

- **`train.py`**  
  Código de treinamento do agente.  
  *Modo padrão*: não renderiza a janela do jogo.
  
*Para renderizar o jogo utilize o parâmetro `SHOW_GAME`. Ex:*
```bash
python train.py SHOW_GAME
```

- **`play.py`**  
Código que executa o jogo utilizando o melhor agente treinado.

- **`config-feedforward`**  
Arquivo de configuração da rede neural utilizado pelo `neat-python`.

- **`checkpoint`**  
Checkpoint da última geração treinada.

- **`best.pkl`**  
Arquivo contendo o agente mais bem-sucedido, salvo após o treinamento.
