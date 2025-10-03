from connect4 import Connect4
from agentes import Agent, RandomAgent,DefenderAgent

def jugar_mil_partidas(verbose, trained_first=True, agent=None, contrincante=None):
    # Crea el agente entrenado
    agente_entrenado:Agent = agent

    # Crea el RandomAgent
    if contrincante == "Defender":
        agente_random:Agent = DefenderAgent("Defender agent")
    else:
        agente_random:Agent = RandomAgent("Random agent")

    if trained_first:
        agent1 = agente_entrenado
        agent2 = agente_random
    else:
        agent1 = agente_random
        agent2 = agente_entrenado

    contador=0
    for i in range(1, 1000):
        juego = Connect4(agent1=agent1, agent2=agent2)
        ganador = juego.play(render=verbose)
        if ganador == 1:
            contador+=1
    return contador
    