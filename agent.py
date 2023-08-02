import keras
from keras import layers
from numba.core.decorators import jit
#from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.keras import utils
from keras.optimizers import adam_v2
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
  optimizer = adam_v2.Adam(learning_rate=0.003)
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
    vision = layers.Input(shape=input_dims[0])
    gameData = layers.Input(shape= input_dims[1])
    vision_2_1 = layers.LeakyReLU()(layers.Conv2D(32, (5,5), strides=(2,2), activation=layers.LeakyReLU())(vision))
    
    gameData_2_2 = layers.Dense(16)(gameData)

    vision_3_1 = layers.LeakyReLU()(layers.Conv2D(16, (3,3), strides = (2,2))(vision_2_1))
    vision_4_1 = layers.Conv2D(16, (3,3), strides = (2,2))(vision_3_1)

    concatenate_4_1 = layers.Concatenate()([layers.Flatten()(vision_4_1), gameData_2_2])
    
    dense_5_1 = layers.Dense(16)(layers.LeakyReLU()(concatenate_4_1))
    dense_6_1 = layers.Dense(32)(layers.LeakyReLU()(dense_5_1))
    
    # right now model is simple, one action per step
     
    movementOutput = layers.Dense(output_dims[0], name="movementOutput")(dense_6_1)
    actionOutput = layers.Dense(output_dims[1], name = "actionOutput")(dense_6_1)

    self.model = keras.Model([vision, gameData], [movementOutput, actionOutput])
    self.model.compile("adam", keras.losses.mse)
    self.target_model = keras.Model([vision, gameData], [movementOutput, actionOutput])
    self.target_model.compile("adam", keras.losses.mse)

  def train(self, epochs, batch_size = 16, notGonnaBeData = True, data = None, verbose = 0):
    '''This will train the agent to try to make it get closer to the optimal solution, the system with notGonnaBeData
     is a botched fix so that the agent can be trained on outside data that isn't in its experience replay'''
    if notGonnaBeData:
      data = self.experienceReplay
      np.random.shuffle(data)
      x, y, actions = self.processDataforTraining(False, data)
    else:
      np.random.shuffle(data)
      x, y, actions = self.processDataforTraining(False, data)

    # This will hold the value of the losses to be printed to the console so we can see the agents loss
    loss_values = [0,0]

    # This will repeat x amount of epochs
    for z in range(epochs):
      # Creates a list of indicies that is then shuffled which makes training better
      indicies =(np.arange(len(x)))
      np.random.shuffle(indicies)

      # Apply the shuffling to the data
      x = [np.array([x[0][z] for z in indicies]), np.array([x[1][z] for z in indicies])]
      y = [np.array([y[0][z] for z in indicies]), np.array([y[1][z] for z in indicies])]
      actions = [np.array([actions[0][z] for z in indicies]), np.array([actions[1][z] for z in indicies])]

      for i in range(0, len(x), batch_size):
        # This will go through the data in increments of batch_size

        # Creates a one hot encoding for the actions so that we can make a prediction on the state and then mask the values of everything besides action since that is the only reward value that we know
        movement_one_hot = tf.one_hot(actions[0], self.env.action_space_size[0])
        action_one_hot = tf.one_hot(actions[1], self.env.action_space_size[1])

        # This is going to calcualte the gradient for the movement output
        with tf.GradientTape() as tape:
          # Muliplies the predicted reward of the model on the batch by the one hot mask so that we only train on the action chosen, these are the predicted values
          y_pred_movement = tf.multiply(self.model([x[0][i:i+batch_size],x[1][i:i+batch_size]], training = True)[0], movement_one_hot[i:i+batch_size])
          # Multiplies the actual reward by the mask to get the actual values
          y_actual_movement = tf.multiply(y[0][i:i+batch_size], movement_one_hot[i:i+batch_size])
          # Figures out the losses (we're using mse because its kinda cool)
          loss_value_movement = keras.losses.MSE(y_actual_movement, y_pred_movement)
        # Appends the loss we got so that we can show the human (me :D) the loss of the model
        loss_values[0] += loss_value_movement
        # Gets the gradient (i dunno how this works)
        grads_movement = tape.gradient(loss_value_movement, self.model.trainable_variables)
        # Applies the gradient (i dunno how this works) and the wierd expression in the parentehsis just make it so that we're not applying gradients that don't exist so theres no warning (i think thats how it works, i dunno)
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads_movement, self.model.trainable_variables) if grad is not None)

        # This is the same thing as the movement gradients, so imma not explain it
        with tf.GradientTape() as tape:
          y_pred_action = tf.multiply(self.model([x[0][i:i+batch_size],x[1][i:i+batch_size]], training = True)[1], action_one_hot[i:i+batch_size])
          y_actual_action = tf.multiply(y[1][i:i+batch_size], action_one_hot[i:i+batch_size])
          loss_value_action = keras.losses.MSE(y_actual_action, y_pred_action)
        loss_values[1] += loss_value_action
        grads_action = tape.gradient(loss_value_action, self.model.trainable_variables)
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads_action, self.model.trainable_variables) if grad is not None)
      # if the verbose is greater than 0 then output the loss to the console
      if verbose > 0:
        print(f"\t\t\tLOSS: {np.sum(loss_values, 1) / len(loss_values)}")
    
    # returns the average loss values for the epochs we did
    return np.sum(loss_values, 1) / len(loss_values)
  
  
  def processDataforTraining(self, notGonnaBeData = True, data = None):
    '''This functions process the data that we get (in the form of state, action, reward, done, next_state), and output training data for the nn'''
    if notGonnaBeData:
      x = np.array(self.experienceReplay,object).transpose()
    else:
      x = np.array(data, object).transpose()
    states = [(np.array([np.array(z[0]) for z in x[0]])), (np.array([np.array(z[1:-1]) for z in x[0]]))]
    next_states = [(np.array([np.array(z[0]) for z in x[-1]])), (np.array([np.array(z[1:-1]) for z in x[-1]]))]
    dones = (x[3].astype(np.int8) -1) *-1
    x[1] = np.array(x[1])
    actions = [[z[0] for z in (x[1])], [z[1] for z in (x[1])]]
    rewards = x[2].astype(np.float16)
    rewards_q = self.target_model.predict(states)
    rewards_Q = self.target_model.predict(next_states)
    rewards_Q = [np.max(rewards_Q[0],1), np.max(rewards_Q[1],1) * 0]
    
    for i in range(len(rewards)):
      rewards_q[0][i][actions[0][i]] = rewards[i] + rewards_Q[0][i] * self.gamma * dones[i]
      rewards_q[1][i][actions[1][i]] = rewards[i] + rewards_Q[1][i] * self.gamma * dones[i]
        
    return states, rewards_q, actions
  
  
  def updateTargetModel(self):
    self.target_model.set_weights(self.model.get_weights())
  
  def predictRewardsForActions(self, state):
    return self.model.predict([np.expand_dims(state[0],0), np.expand_dims(state[1:-1],0)])

if __name__ == "__main__":
  agent = Agent(2)
  agent.define_model([[68, 135,1], 3], [6])
  print(agent.model.summary())