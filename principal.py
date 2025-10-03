import torch.nn as nn
from agentes import Agent
from utils import *
import torch 
import random
import numpy as np
class Connect4State:
    def __init__(self,rows=6,cols=7): 
        self.rows = rows
        self.cols = cols
        self.board=create_board(self.rows, self.cols)  # arranco tablero vacio
        self.current_player = 1  # el jugador 1 empieza
        self.termino_el_juego = False
        self.ganador = None  # Nadie ganó todavia

    def copy(self):  
        nuevo_estado = Connect4State()
        nuevo_estado.board = self.board.copy()
        nuevo_estado.current_player = self.current_player
        nuevo_estado.termino_el_juego = self.termino_el_juego
        nuevo_estado.ganador = self.ganador
        return nuevo_estado

    def update_state(self):
        if check_game_over(self.board)[0]==True:
            self.ganador=check_game_over(self.board)[1]
            self.termino_el_juego = True
        else:
            if self.current_player==1:
                self.current_player=2
            else:
                self.current_player=1

    def __eq__(self, other):
        """
        Compara si dos estados son iguales.
        
        Args:
            other: Otro estado para comparar.
            
        Returns:
            True si los estados son iguales, False en caso contrario.
        """
        #dos estados son iguales si tienen el mismo tablero y el mismo jugador actual
        return np.array_equal(self.board, other.board) and self.current_player == other.current_player

    def __hash__(self): 
        """
        Genera un hash único para el estado.
        
        Returns:
            Hash del estado basado en el tablero y jugador actual.
        """
        
        pass

    def __repr__(self):
        """
        Representación en string del estado.
        
        """
        return f"Jugador actual: {self.current_player}\nTermino:\n{self.termino_el_juego}\nGanador: {self.ganador}"
    #preguntarle a chat como poronga imprimo el tablero

class Connect4Environment:
    def __init__(self,rows=6,cols=7):
        """
        Inicializa el ambiente del juego Connect4.
        
        Args:
            Definir las variables de instancia de un ambiente de Connect4

        """
        self.state=Connect4State()
        self.rows=rows
        self.cols=cols


    def reset(self):
        """
        Reinicia el ambiente a su estado inicial para volver a realizar un episodio.
        
        """
        self.state=Connect4State()
        return self.state.copy()

    def available_actions(self):
        """
        Obtiene las acciones válidas (columnas disponibles) en el estado actual.
        
        Returns:
            Lista de índices de columnas donde se puede colocar una ficha.
        """
        acciones = [c for c in range(self.state.cols) if self.state.board[0, c] == 0]
        return acciones

    def step(self, action):
        """
        Ejecuta una acción.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Devuelve la tupla: nuevo_estado, reward, terminó_el_juego?, ganador
        Si terminó_el_juego==false, entonces ganador es None.
        
        Args:
            action: Acción elegida por un agente.
            
        """
        
        insert_token(self.state.board, action, self.state.current_player)
        self.state.update_state()
        
        if self.state.termino_el_juego:
            if self.state.ganador is None:
                reward = 0  # Empate
            elif self.state.ganador == 1:
                reward = 1  # Jugador 1 gana
            else:
                reward = -1  # Jugador 2 gana
        else:
            reward = 0  # Juego continúa
        
        return self.state.copy(), reward, self.state.termino_el_juego, {"winner": self.state.ganador}


    def render(self):
            """
            Muestra visualmente el tablero en consola.
            """
            symbols = {0: ".", 1: "X", 2: "O"}
            print("\nTablero:")
            for r in range(self.rows):
                fila = " ".join(symbols[val] for val in self.board[r])
                print(f"| {fila} |")
            print("  " + " ".join(str(c) for c in range(self.cols)))
    

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim): 
        """
        Inicializa la red neuronal DQN para el aprendizaje por refuerzo.
        
        Args:
            input_dim: Dimensión de entrada (número de features del estado).
            output_dim: Dimensión de salida (número de acciones posibles).
        """
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )



    def forward(self, x):
        """
        Pasa la entrada a través de la red neuronal.
        
        Args:
            x: Tensor de entrada.
            
        Returns:
            Tensor de salida con los valores Q para cada acción.
        """
        return self.net(x)

class DeepQLearningAgent:
    def __init__(self, state_shape, n_actions, device,
                 gamma, epsilon, epsilon_min, epsilon_decay,
                 lr, batch_size, memory_size, target_update_every): 
        """
        Inicializa el agente de aprendizaje por refuerzo DQN.
        
        Args:
            state_shape: Forma del estado (filas, columnas).
            n_actions: Número de acciones posibles.
            device: Dispositivo para computación ('cpu' o 'cuda').
            gamma: Factor de descuento para recompensas futuras.
            epsilon: Probabilidad inicial de exploración.
            epsilon_min: Valor mínimo de epsilon.
            epsilon_decay: Factor de decaimiento de epsilon.
            lr: Tasa de aprendizaje.
            batch_size: Tamaño del batch para entrenamiento.
            memory_size: Tamaño máximo de la memoria de experiencias.
            target_update_every: Frecuencia de actualización de la red objetivo.
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_every = target_update_every
        self.q_network=DQN(input_dim=state_shape[0]*state_shape[1], output_dim=n_actions).to(device)
        self.memory=[]
    
     

    def preprocess(self, state):
        """
        Convierte el estado del juego a un tensor de PyTorch.
        
        Args:
            state: Estado del juego.
            
        Returns:
            Tensor de PyTorch con el estado aplanado.
        """
        state_array = state.board.flatten()
        state_tensor = torch.tensor(state_array, dtype=torch.float32).to(self.device)
        return state_tensor

    def select_action(self, state, valid_actions): 
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la acción seleccionada.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            state_tensor = self.preprocess(state).unsqueeze(0)  # Añadir dimensión de batch
            with torch.no_grad():
                q_values = self.q_network.forward(state_tensor)
            q_values = q_values.cpu().numpy().flatten()
            q_values_valid = [(a, q_values[a]) for a in valid_actions]
            best_action = max(q_values_valid, key=lambda x: x[1])[0]
            return best_action

    def store_transition(self, s, a, r, s_next, done):
        """
        Almacena una transición (estado, acción, recompensa, siguiente estado, terminado) en la memoria.
        
        Args:
            s: Estado actual.
            a: Acción tomada.
            r: Recompensa obtenida.
            s_next: Siguiente estado.
            done: Si el episodio terminó.
        """
        self.memory.append((s, a, r, s_next, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def train_step(self): 
        """
        Ejecuta un paso de entrenamiento usando experiencias de la memoria.
        
        Returns:
            Valor de la función de pérdida si se pudo entrenar, None en caso contrario.
        """
        if len(self.memory) < self.batch_size:
            return None  # No hay suficientes experiencias para entrenar

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.stack([self.preprocess(s) for s in states]).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_tensor = torch.stack([self.preprocess(s) for s in next_states]).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q_values = self.q_network.forward(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            max_next_q_values = self.q_network.forward(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q_values = rewards_tensor + (self.gamma * max_next_q_values * (1 - dones_tensor))

        loss_fn = nn.MSELoss()
        loss = loss_fn(current_q_values, target_q_values)

        optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def update_epsilon(self):
        """
        Actualiza el valor de epsilon para reducir la exploración gradualmente.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

class TrainedAgent(Agent):
    def __init__(self, state_shape: tuple, n_actions: int, device='cpu', 
                 model_path: str = None, q_network=None):
        self.device = device
        self.state_shape = state_shape
        self.n_actions = n_actions

        if q_network is not None:
            # Usar red ya entrenada
            self.q_network = q_network.to(device)
            self.name = "TrainedAgent_from_qnetwork"
        elif model_path is not None:
            # Cargar red desde archivo
            self.q_network = DQN(
                input_dim=state_shape[0]*state_shape[1], 
                output_dim=n_actions
            ).to(device)
            self.q_network.load_state_dict(torch.load(model_path, map_location=device))
            self.name = f"TrainedAgent_{model_path}"
        else:
            raise ValueError("Debes pasar 'model_path' o 'q_network' para inicializar el agente.")

        self.q_network.eval()  # eval mode

    def play(self, state, valid_actions):
        # Convertimos el estado a tensor
        state_array = state.board.flatten()
        state_tensor = torch.tensor(state_array, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()
        
        # Solo consideramos las acciones válidas
        q_values_valid = [(a, q_values[a]) for a in valid_actions]
        best_action = max(q_values_valid, key=lambda x: x[1])[0]
        return best_action
