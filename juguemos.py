import argparse
from connect4 import Connect4
from agentes import Agent, RandomAgent,DefenderAgent
from principal import TrainedAgent
import torch
from pathlib import Path

def main(verbose, trained_first=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crea el agente entrenado
    agente_entrenado:Agent = TrainedAgent(
        model_path=Path("/home/g_garcia/TP-IA-NEURO/trained_model_random_search_vs_None_best.pth"),

        state_shape=(6, 7),
        n_actions=7,
        device=device
    )
    
    # Crea el RandomAgent
    agente_random:Agent = DefenderAgent("def agent")

    if trained_first:
        agent1 = agente_entrenado
        agent2 = agente_random
    else:
        agent1 = agente_random
        agent2 = agente_entrenado

    print(f"Juego: {agent1.name} (Nosotros) vs. def agent (Jugador 2)")

    juego = Connect4(agent1=agent1, agent2=agent2)
    ganador = juego.play(render=verbose)

    print("\nResultado del juego:")
    if ganador == 0:
        print("Empate")
        return 0
    elif ganador == 1:
        print(f"Gana Nosotros (Jugador 1)")
        return 1
    else:
        print("Gana el def agent (Jugador 2)")
        return 0
    
    

if __name__ == '__main__':
    contador=0
    for i in range(1000):
        print(f"Partida {i+1}")
        parser = argparse.ArgumentParser(description="Jugar una partida de Conecta 4: TrainedAgent vs RandomAgent.")
        parser.add_argument('-v', '--verbose', action='store_true', help='Mostrar el tablero en cada turno')
        
        args = parser.parse_args()
        contador+=main(args.verbose, trained_first=True)

    print(f"Ganamos {contador} de 1000 partidas")
