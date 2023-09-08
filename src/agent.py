import keras
from keras import layers
#from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.keras import utils
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from collections import deque


class ReplayExperience(deque):
  '''This will be used as the experience replay (don't ask questions why the words are flipped in the class name), it is a deque because deques are faster than arrays for appending'''
  def __init__(self, arr = None,maxlen = None):
    if arr != None:
      super(ReplayExperience, self).__init__(arr, maxlen = maxlen)
    else:
      super(ReplayExperience, self).__init__(maxlen = maxlen)
  
  def sample(self,amount):
    indicies = np.random.randint(0, len(self), amount)
    return [self[z] for z in indicies]


class Agent:
  '''This class is what is going to be interacting with the environment, it has the brains, it is a normal agent in RL'''
  
  # Optimizer we use
  optimizer = Adam(learning_rate=0.003)
  # What will be used to calculate discounted reward
  gamma = 0.95
  # The agent's experience replay
  experienceReplay = ReplayExperience(maxlen = 5000)
  
  def __init__(self, env):
    # Link the agent to the env
    self.env = env
  
  def define_model(self, input_dims, output_dims):
    '''Inputs dims will be a screenshot of the image, our current health, and our current energy (excluding our xp imma try)
       this function defines the model that will take in the inputs, do processing and output the rewards for the 5 possible actions'''
    
    if len(input_dims) == 2: # If we are not only passing in the gameData but also the vision data
      vision = layers.Input(shape=input_dims[0])
      gameData = layers.Input(shape= input_dims[1])
      vision_2_1 = layers.LeakyReLU()(layers.Conv2D(32, (5,5), strides=(2,2), activation=layers.LeakyReLU())(vision))
      
      gameData_2_2 = layers.Dense(16)(gameData)

      vision_3_1 = layers.LeakyReLU()(layers.Conv2D(16, (3,3), strides = (2,2))(vision_2_1))
      vision_4_1 = layers.Conv2D(16, (3,3), strides = (2,2))(vision_3_1)

      concatenate_4_1 = layers.Concatenate()([layers.Flatten()(vision_4_1), gameData_2_2])
      
      dense_5_1 = layers.Dense(16)(layers.LeakyReLU()(concatenate_4_1))
      dense_6_1 = layers.Dense(32)(layers.LeakyReLU()(dense_5_1))
      
      movementOutput = layers.Dense(output_dims[0], name="movementOutput")(dense_6_1)
      actionOutput = layers.Dense(output_dims[1], name = "actionOutput")(dense_6_1)

      self.model = keras.Model([vision, gameData], [movementOutput, actionOutput])
      self.target_model = keras.Model([vision, gameData], [movementOutput, actionOutput])
    else:

      vision = layers.Input(shape=input_dims)
      vision_2_1 = layers.Dropout(0.1)(layers.LeakyReLU()(layers.MaxPool2D((2,2), strides=(2,2))(layers.Conv2D(32, (5,5))(vision))))
      vision_3_1 = layers.Dropout(0.1)(layers.LeakyReLU()(layers.MaxPool2D((2,2), strides=(2,2))(layers.Conv2D(32, (3,3))(vision_2_1))))
      vision_4_1 = layers.Dropout(0.1)(layers.LeakyReLU()(layers.MaxPool2D((2,2), strides=(2,2))(layers.Conv2D(32, (3,3))(vision_3_1))))
      vision_2_1 = layers.Dropout(0.1)(layers.LeakyReLU()(layers.MaxPool2D((2,2), strides=(2,2))(layers.Conv2D(32, (5,5))(vision))))
      vision_3_1 = layers.Dropout(0.1)(layers.LeakyReLU()(layers.MaxPool2D((2,2), strides=(2,2))(layers.Conv2D(32, (3,3))(vision_2_1))))
      vision_4_1 = layers.Dropout(0.1)(layers.LeakyReLU()(layers.MaxPool2D((2,2), strides=(2,2))(layers.Conv2D(32, (3,3))(vision_3_1))))
      dense_5_1 = layers.Dense(64)(layers.LeakyReLU()(layers.Flatten()(vision_4_1)))

      movementOutput = layers.Dense(output_dims[0], name="movementOutput")(dense_5_1)
      actionOutput = layers.Dense(output_dims[1], name = "actionOutput")(dense_5_1)
      movementOutput = layers.Dense(output_dims[0], name="movementOutput")(dense_5_1)
      actionOutput = layers.Dense(output_dims[1], name = "actionOutput")(dense_5_1)

      self.model = keras.Model(vision, [movementOutput, actionOutput])
      self.target_model = keras.Model(vision, [movementOutput, actionOutput])
      
    self.model.compile("adam", keras.losses.mse)
    self.target_model.compile("adam", keras.losses.mse)

  def train(self, epochs, batch_size = 16, notGonnaBeData = True, data = None, verbose = 0, use_target = True):
    '''This will train the agent to try to make it get closer to the optimal solution, the system with notGonnaBeData
     is a botched fix so that the agent can be trained on outside data that isn't in its experience replay'''
    if notGonnaBeData:
      data = self.experienceReplay
      assert len(data) > 0
      np.random.shuffle(data)
      x_base, y_base, actions_base = self.processDataforTraining(False, data, use_target = use_target)
    else:
      np.random.shuffle(data)
      assert len(data) > 0
      x_base, y_base, actions_base = self.processDataforTraining(False, data, use_target = use_target)

    if len(data) <= 1:
      print("DUDE THERE IS NO DATA IN THE EXPERIENCE REPLAY, YOU NEED TO ADD SOME DATA TO THE EXPERIENCE REPLAY BEFORE YOU CAN TRAIN THE MODEL")
      return 999999
    
    loss_values = [0, 0]

    for z in range(epochs):
        indices = np.random.permutation(len(x_base))

        x = x_base[indices]
        y = [y_base[i][indices] for i in range(2)]
        actions = [actions_base[i][indices] for i in range(2)]

        for i in range(0, len(x), batch_size):
            movement_one_hot = tf.one_hot(actions[0][i:i + batch_size], self.env.action_space_size[0])
            action_one_hot = tf.one_hot(actions[1][i:i + batch_size], self.env.action_space_size[1])

            with tf.GradientTape() as tape:
                y_pred_movement = tf.multiply(self.model(x[i:i + batch_size], training=True)[0], movement_one_hot)
                y_actual_movement = tf.multiply(y[0][i:i + batch_size], movement_one_hot)
                loss_value_movement = keras.losses.MSE(y_actual_movement, y_pred_movement)

            loss_values[0] += np.sum(loss_value_movement.numpy())
            grads_movement = tape.gradient(loss_value_movement, self.model.trainable_variables)
            self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads_movement, self.model.trainable_variables) if grad is not None)

            with tf.GradientTape() as tape:
                y_pred_action = tf.multiply(self.model(x[i:i + batch_size], training=True)[1], action_one_hot)
                y_actual_action = tf.multiply(y[1][i:i + batch_size], action_one_hot)
                loss_value_action = keras.losses.MSE(y_actual_action, y_pred_action)

            loss_values[1] += np.sum(loss_value_action.numpy())
            grads_action = tape.gradient(loss_value_action, self.model.trainable_variables)
            self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads_action, self.model.trainable_variables) if grad is not None)
            if verbose > 0:
                print(f"\t\t\tLOSS: {loss_values}")

    return np.sum(loss_values) / len(x)
  
  
  def processDataforTraining(self, notGonnaBeData = True, data = None, use_target = True):
    '''This functions process the data that we get (in the form of state, action, reward, done, next_state), and output training data for the nn'''
    if data is None:
        x = np.array(self.experienceReplay, object).T.tolist()
    else:
        x = np.array(data, object).T.tolist()

    states = np.expand_dims(np.array(x[0], dtype=np.float16), -1)
    next_states = np.expand_dims(np.array(x[4], dtype=np.float16), -1)
    dones = (np.array(x[3], dtype=np.int8) - 1) * -1
    rewards = np.array(x[2], dtype=np.float16)

    actions = [np.array([z[i] for z in x[1]], dtype=np.int) for i in range(2)]

    if use_target:
        rewards_q = self.target_model.predict(states, workers=12, use_multiprocessing=True, verbose=-1)
        rewards_Q = self.target_model.predict(next_states, workers=12, use_multiprocessing=True, verbose=-1)
        rewards_Q = [np.max(rewards_Q[i], axis=1) for i in range(2)]
    else:
        rewards_q = [np.zeros(shape=(states.shape[0], 5)), np.zeros(shape=(states.shape[0], 2))]
        for i in range(2):
            rewards_q[i][np.arange(len(rewards)), actions[i]] = rewards
        return states, rewards_q, actions

    for i in range(2):
        rewards_q[i][np.arange(len(rewards)), actions[i]] = rewards + rewards_Q[i] * self.gamma * dones[:, 0]

    return states, rewards_q, actions
  
  def updateTargetModel(self):
    self.target_model.set_weights(self.model.get_weights())
  
  def predictRewardsForActions(self, state):
    # uncomment the following if the model is being fed the xp and health data
    # return self.model.predict(np.expand_dims(state[0],0), np.expand_dims(state[1:-1],0)])
    # if(state.shape[0] == 1):
    #   return self.model.predict(state, verbose=0)
    # if(state.shape[0] == 1):
    #   return self.model.predict(state, verbose=0)
    return self.model.predict(np.expand_dims(state,0), verbose=0)

if __name__ == "__main__":
  agent = Agent(2)
  agent.define_model([50, 50,1], [4, 2])
  agent.define_model([50, 50,1], [4, 2])
  print(agent.model.summary())

  # Make the model predict on a random input to test it
  # print(agent.model.predict([np.random.rand(1,68,135,1), np.random.rand(1,3)]))