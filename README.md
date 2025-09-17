# Implementação de um algoritmo adaptativo para otimização de condições de avaliação na Robótica Adaptativa

Este repositório contém a implementação de funções de otimização de condições para os ambientes de controle robótico CartPolev-1 e PendulumV-1 [1].

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)

## Documentação

A documentação completa do projeto está disponível no [Wiki do repositório](https://github.com/Brenda-Machado/otimizacao-condicoes-avaliacao/wiki).

## Experimentos

Conforme descrito na seção 3. Metodologia do TCC, diversos experimentos foram realizados para otimizar as condições de avaliação nos ambientes testados. Abaixo estão os detalhes dos experimentos realizados:

- **Experimento Controle**: O agente foi treinado com todas as possíveis condições iniciais. O objetivo foi obter uma visão real da fitness, que seria alcançada se fosse possível percorrer todas as condições iniciais para cada agente.

- **Experimento 1**: Modificação da quantidade de episódios de avaliação. O agente foi avaliado em N episódios com N condições iniciais diferentes, medindo seu comportamento médio.

- **Experimento 2**: Variação da duração do episódio de avaliação, ou seja, o tempo que o agente interage no ambiente e é avaliado.

- **Experimento 3**: Modificação da variação do ruído adicionado ao motor do agente.

- **Experimento 4**: Alteração do intervalo das condições iniciais, com valores aleatórios dentro do intervalo [x, y].

- **Experimento 5**: Avaliação do peso da fitness de cada episódio avaliativo, considerando a maior e a menor fitness como a final, ao invés da média padrão.

- **Experimento 6**: Modificação do peso das componentes da fitness, atribuindo um valor entre 0 e 1 para o impacto de cada componente (maior, média, e menor) na fitness final.

## Autores

- [Brenda Machado](https://www.github.com/Brenda-Machado).

## Referências principais de implementação

[1] - [Gymnasium - Farama Foundation](https://github.com/Farama-Foundation/Gymnasium);
[2] - [Evorobotpy2](https://github.com/snolfi/evorobotpy2).

## Citação

Trabalho de Conclusão de Curso em Ciências da Computação na Universidade Federal de Santa Catarina.

```bibtex
@article{otimizacaoCondicoesAvaliacao2025,
  title={Implementação de um algoritmo adaptativo para otimização de condições de avaliação na Robótica Adaptativa},
  author={Machado, Brenda},
  journal={...},
  year={2025}
}
