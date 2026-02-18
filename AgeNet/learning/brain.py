import numpy as np
import tensorflow as tf
from collections import deque


# -------------------------------------------------------------------------------------------
class RLBrain:
    """
    Reinforcement Learning brain of an agent.
    Responsible ONLY for:
      - action selection
      - experience replay
      - training the policy network
    """
    # ---------------------------------------------------------------------------------------
    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 2,
        learning_rate: float = 1e-5,
        gamma: float = 0.98,
        batch_size: int = 20,
        model_path: str | None = None
    ):

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.batch_size = batch_size

        # Build model ---
        self.model     = self._build_model(model_path)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.loss_fn   = tf.keras.losses.MeanSquaredError()


        self.replay_memory = deque(maxlen=50)   # A bag for 50 recently viewed (state, action, reward, next_state)
        self.train_counter = 0                  # Just to set the train to run every few steps
        self.last_loss     = tf.constant(0.0)   # Just for print and plot loss per step


    # ---------------------------------------------------------------------------------------
    def _build_model(self, model_path: str=None):
        tf.keras.backend.clear_session()
        custom_activation = {'_custom_activation': tf.keras.layers.Activation(self._custom_activation)}
        # tf.keras.utils.get_custom_objects().update({ '_custom_activation': _custom_activation })

        if model_path is not None:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_activation, compile=False)
        else:
            inputs = tf.keras.layers.Input(shape=(self.state_dim,), name='input') 
            x = tf.keras.layers.Dense(32, activation='elu', name='dense_1')(inputs) 
            x = tf.keras.layers.Dense(32, activation='elu', name='dense_2')(x) 
            outputs = tf.keras.layers.Dense(self.action_dim, activation=self._custom_activation, name='output')(x) 
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        return model

    # ---------------------------------------------------------------------------------------
    def act(self, state: np.ndarray) -> int:
        # Choose action greedily
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        q_values = self.model({'input': state}).numpy()[0]
        return int(np.argmax(q_values))

    # ---------------------------------------------------------------------------------------
    def remember(self, state, action, reward, next_state):
        self.replay_memory.append( (state, action, reward, next_state) )

    # ---------------------------------------------------------------------------------------
    def train_per(self, steps_per_train: int=10):
        self.train_counter += 1
        if ( len(self.replay_memory) < self.batch_size or self.train_counter % steps_per_train != 0 ): return
        self._train()

    # ---------------------------------------------------------------------------------------
    def _train(self):
        indices = np.random.randint(len(self.replay_memory), size=self.batch_size)  # 32 random number between[0 - len(replay)]
        batch   = [self.replay_memory[index] for index in indices]                  # number in replay_memory[indices]

        states, actions, rewards, next_states = map( np.array, zip(*batch) )        # from replay_memory read these and save in

        # Stabilize rewards (legacy logic preserved)
        rewards = np.where( (-0.1 < rewards)&(rewards < 0.05), 0.0, np.round(rewards, 3) )

        next_Q     = self.model({'input': next_states})     # 32 predict of 2 actions
        max_next_Q = tf.reduce_max(next_Q, axis=1)          # choose higher probiblity of each actions (of each 32)
        target_Q   = rewards + self.gamma * max_next_Q      # Equation 18-5. Q-Learning algorithm
        target_Q   = tf.reshape(target_Q, (-1, 1))          # reshape to (32,1) beacuse of Q_values.shape
        
        mask = tf.one_hot(actions, self.action_dim)
        with tf.GradientTape() as tape:
            all_Q_values = self.model({'input': states})
            Q_values = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
            self.last_loss = tf.reduce_mean(self.loss_fn(target_Q, Q_values))
        grads = tape.gradient(self.last_loss, self.model.trainable_variables)


        if not any(np.isnan(g.numpy()).any() for g in grads if g is not None):
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    # ---------------------------------------------------------------------------------------
    def save_model(self, output: str):
        self.model.save(output)


    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _custom_activation(x):
        return 100 - tf.nn.elu(-tf.sqrt(tf.nn.softplus(x)) + 100)
