# Implementação de um algoritmo adaptativo para otimização de condições de avaliação na Robótica Adaptativa

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)

Este repositório contém a implementação de funções de otimização de condições para os ambientes de controle robótico:
+ Série de experimentos que variam as condições ambientais dos ambientes;
+ Funções de plotagem da fitness landscape;
+ Algoritmos evolutivos baseline;
+ Algoritmo adaptativo de condições online baseado em Estratégias de Evolução.

## Documentação

A documentação completa do projeto está disponível no [Relatório de TCC]().

## Experimentos

Conforme descrito na seção 3. Metodologia do TCC, diversos experimentos foram realizados para otimizar as condições de avaliação nos ambientes testados, [PendulumV-1](https://gymnasium.farama.org/environments/classic_control/pendulum/) e [CartpoleV-0](https://gymnasium.farama.org/environments/classic_control/cart_pole/). Abaixo estão os detalhes dos experimentos realizados:

- **Experimento Controle**: O agente foi treinado com todas as possíveis condições iniciais. O objetivo foi obter uma visão real da fitness, que seria alcançada se fosse possível percorrer todas as condições iniciais para cada agente.

- **Experimento 1**: Modificação da quantidade de episódios de avaliação. O agente foi avaliado em N episódios com N condições iniciais diferentes, medindo seu comportamento médio.

- **Experimento 2**: Variação da duração do episódio de avaliação, ou seja, o tempo que o agente interage no ambiente e é avaliado.

- **Experimento 3**: Modificação da variação do ruído adicionado ao motor do agente.

- **Experimento 4**: Alteração do intervalo das condições iniciais, com valores aleatórios dentro do intervalo [x, y].

## Como rodar a aplicação?

1. Clonar o repositório:

   Clone o repositório para a sua máquina local:

   ```bash
   git clone https://github.com/Brenda-Machado/otimizacao-condicoes-avaliacao.git
   cd otimizacao-condicoes-avaliacao
   ```

2. Criar e ativar o ambiente virtual:
   ```bash
   make venv/bin/activate
   ```

3. Rodar os experimentos:
    Para rodar a evolução do baseline e do algortimo adaptativo, 10 seeds por padrão, execute:
    ```bash
   make run
   ```

4. Rodar experimentos:
    Para rodar os experimentos das condições de avaliação, execute, no caso do Pendulum:
    ```bash
   make opt_pen
   ```
   Ou, no caso do CartPole:
    ```bash
   make opt_cp
   ```
5. Otimizar baseline utilizando [IRACE](https://github.com/MLopez-Ibanez/irace):
    ```bash
   make opt_irace
   ```

## Autores

- [Brenda Silva Machado](https://www.github.com/Brenda-Machado).

## Citação

Trabalho de Conclusão de Curso em Ciências da Computação na Universidade Federal de Santa Catarina.

```bibtex
@article{otimizacaoCondicoesAvaliacao2025,
  title={Implementação de um algoritmo adaptativo para otimização de condições de avaliação na Robótica Adaptativa},
  author={Machado, Brenda},
  journal={..},
  year={2025}
}
