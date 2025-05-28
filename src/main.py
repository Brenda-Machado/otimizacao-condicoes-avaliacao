from experiments.otim_cart_pole import *

def main():

    ## Experimento controle:
    # results, _, _ = experimento_controle(0.25,0.25,0.025)
    # plot_results(results)

    ## Experimento 1:
    # episodes = [2, 5, 10, 15, 20,50]
    # for ep in episodes:
    #     results = experimento_1(0.25, 0.25, ep)
    #     plot_results(results)

    # Experimento 2:
    steps = [100, 500, 1000, 1500, 2000]
    for s in steps:
        results = experimento_2(0.25, 0.25, s)
        plot_results(results)

main()