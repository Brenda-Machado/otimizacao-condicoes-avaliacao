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
    # steps = [100, 500, 1000, 1500, 2000]
    # for s in steps:
    #     results = experimento_2(0.25, 0.25, s)
    #     plot_results(results)

    # Experimento 3
    # noise = [(-0.001, 0.001), (-0.01, 0.01), (-0.05, 0.05), (-0.1, 0.1), (-0.5, 0.5), (-1, 1)]
    # for n in noise:
    #     results = experimento_3(0.25, 0.25, n)
    #     plot_results(results)   

    # Experimento 4
    # ranges = [[(0, 0), (0, 0), (-0.2, 0.2), (-2.0, 2.0)], [(0, 0), (0, 0), (-0.2, 0.2), (-1.0, 1.0)], [(0, 0), (0, 0), (-0.2, 0.2), (-3.0, 3.0)], [(0, 0), (0, 0), (-0.15, 0.15), (-3.0, 3.0)], [(0, 0), (0, 0), (-0.1, 0.1), (-3.0, 3.0)]]
    # for r in ranges:
    #     results = experimento_4(0.25, 0.25, r)
    #     plot_results(results)  

    # Experimento 5
    metodos = ["media", "max", "min"]
    for m in metodos:
        results = experimento_5(0.25, 0.25, m)
        plot_results(results)  

    # Experimento 6
    # ranges = [[(0, 0), (0, 0), (-0.2, 0.2), (-2.0, 2.0)], [(0, 0), (0, 0), (-0.2, 0.2), (-1.0, 1.0)], [(0, 0), (0, 0), (-0.2, 0.2), (-3.0, 3.0)], [(0, 0), (0, 0), (-0.15, 0.15), (-3.0, 3.0)], [(0, 0), (0, 0), (-0.1, 0.1), (-3.0, 3.0)]]
    # for r in ranges:
    #     results = experimento_6(0.25, 0.25, r)
    #     plot_results(results)  

main()