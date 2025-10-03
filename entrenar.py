import torch
from principal import Connect4Environment, Connect4State, DeepQLearningAgent, TrainedAgent
from agentes import Agent
import argparse
from juguemos_mejor import jugar_mil_partidas
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def entrenar(episodes:int=500,
             gamma:float=0.99, 
             epsilon_start:float=1.0, 
             epsilon_min:float=0.1, 
             epsilon_decay:float=0.995,
             alpha:float=0.001,
             batch_size:int=64, 
             memory_size:int=500,
             target_update_every:int=100,
             opponent:Agent=None,
             verbose:bool=True,
             mejor_model:float=0.0):
    ''' Entrenar un Agente DQN en la cantidad de episodios y con los 
        parámetros indicados. 
        Entrena jugando contra el agente opponent, si está definido.
        Si opponent==None, entrena jugando contra sí mismo. '''

    nombre_oponente:str = 'None' if opponent==None else opponent.name
    model_name:str = f"trained_model_random_search_vs_{nombre_oponente}"
    if verbose: print(model_name, flush=True)
    
    # Inicialización del ambiente
    env:Connect4Environment = Connect4Environment()
    
    # Inicialización del agente
    agent:Agent = DeepQLearningAgent(
        state_shape=(env.rows, env.cols),
        n_actions=env.cols,
        device=device,
        gamma=gamma, 
        lr=alpha, 
        batch_size=batch_size, 
        target_update_every=target_update_every, 
        epsilon_decay=epsilon_decay,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        memory_size=memory_size
    )
    
    # Entrenamiento
    for episode in range(episodes):
        state:Connect4State = env.reset()
        done:bool = False
        episode_losses = []  
        dqn_player = 1  # DQN siempre es jugador 1
    
        while not done:
            valid_actions = env.available_actions()
            if env.state.current_player==dqn_player or opponent==None:
                # Turno del DQN (o no hay oponente)
                action = agent.select_action(state, valid_actions)
                next_state, reward, done, _ = env.step(action)
                # Solo almacenar experiencias cuando DQN juega
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step() 
                if loss is not None:
                    episode_losses.append(loss)
            else: 
                # Turno del oponente
                action = opponent.play(state, valid_actions)
                next_state, reward, done, _ = env.step(action)
    
            state = next_state
    
        agent.update_epsilon() 
    
        if (episode + 1) % 100 == 0:
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            if verbose: print(f"Episodio {episode + 1} finalizado. Epsilon: {agent.epsilon:.4f} | Loss promedio: {avg_loss:.6f}", flush=True)

    if verbose: print(flush=True)

    trained_q_net = agent.q_network  # tu red ya entrenada

    agent_net = TrainedAgent(
        q_network=trained_q_net,
        state_shape=(6, 7),
        n_actions=7,
        device="cpu"
    )

    # Guardar el modelo si es el mejor hasta ahora
    model_value:float = jugar_mil_partidas(verbose=False, trained_first=True, agent=agent_net, contrincante=nombre_oponente)
    print(f"El modelo ganó {model_value} de 1000 partidas contra {nombre_oponente}")

    if mejor_model < model_value:
        torch.save(agent.q_network.state_dict(), f"{model_name}.pth")
    return model_value

import random

def sample_params():
    params = {
        "episodes": random.randint(1000, 3000),        # entero continuo
        "gamma": random.uniform(0.9, 0.99),           # real continuo
        "epsilon_start": random.uniform(0.8, 1.0),    # real continuo
        "epsilon_min": random.uniform(0.05, 0.2),     # real continuo
        "epsilon_decay": random.uniform(0.99, 0.999), # real continuo
        "alpha": random.uniform(0.001, 0.003),         # real continuo
        "batch_size": random.choice([32, 64, 128, 256]),   # discreto
        "memory_size": random.randint(500, 3000),   # entero continuo
        "target_update_every": random.randint(75, 150) # entero continuo
    }
    return params



if __name__ == '__main__':

    # Definimo mejor modelo inicial
    mejor_model:float = 0.0
    # Definimos los rangos de búsqueda
    param_grid = {
        "episodes": [500, 1000, 2000],   # lista de valores
        "gamma": [0.8, 0.85, 0.9, 0.95, 0.99],
        "epsilon_start": [1.0, 0.9, 0.8],
        "epsilon_min": [0.05, 0.1, 0.2],
        "epsilon_decay": [0.995, 0.99, 0.98, 0.95],
        "alpha": [0.001, 0.005, 0.01, 0.05],
        "batch_size": [32, 64, 128],
        "memory_size": [5000, 10000, 20000],
        "target_update_every": [10, 20, 50]
    }

    # Cantidad de combinaciones a probar
    N_SEARCH = 100  

    mejor_model = 0.0
    mejores_params = None

    for i in range(N_SEARCH):
        # Elegir parámetros aleatorios
        #params = {k: random.choice(v) for k, v in param_grid.items()}
        params = sample_params()
        print(f"\n🔎 Iteración {i+1} con parámetros: {params}")

        # Entrenar y obtener precisión
        precision = entrenar(
            episodes=params["episodes"],
            gamma=params["gamma"],
            epsilon_start=params["epsilon_start"],
            epsilon_min=params["epsilon_min"],
            epsilon_decay=params["epsilon_decay"],
            alpha=params["alpha"],
            batch_size=params["batch_size"],
            memory_size=params["memory_size"],
            target_update_every=params["target_update_every"],
            verbose=False,
            mejor_model=mejor_model
        )

        print(f"✅ Precisión obtenida: {precision:.4f}")

        # Guardar el mejor
        if precision > mejor_model:
            mejor_model = precision
            mejores_params = params

    print("\n🏆 Mejor modelo encontrado:")
    print("Precisión:", mejor_model)
    print("Parámetros:", mejores_params)
    
    
