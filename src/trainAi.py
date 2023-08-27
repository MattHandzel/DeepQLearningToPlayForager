from PIL.Image import PERSPECTIVE
import tensorflow as tf
#from tensorflow.python.framework.ops import add_to_collections
from mapInput import *
from environment import *
from agent import *
from dataManager import Manager

import time, keras

class Trainer:
  def __init__(self, gym):
    self.gym = gym
    self.manager = self.gym.manager
    self.env = self.gym.env

  def train(self, epochs, epochsBeforeUpdating):
    print("CURRENTLY TRAINING MODEL...")
    s_time = time.time()
    for b in range(epochs // epochsBeforeUpdating):
      for i in range(epochsBeforeUpdating):
        try:
          data = (self.manager.loadRandomSample())
          print(f"\tTIME: {round((time.time() - s_time) * 10) / 10}\tLOSS {(b * epochsBeforeUpdating) + (i+1)} / {epochs}: {np.sum(self.gym.agent.train(16, False, data = data, verbose = 0))}")
        except Exception as e:
          print(e)
 
      print("UPDATING TARGET MODEL")
      self.gym.agent.updateTargetModel()
    print("SAVING MODELS...")
    self.gym.nEpisodes += 1
    self.gym.updateMetadata()
    self.gym.manager.saveSession(self.gym.agent, self.gym.nEpisodes)
  
  def trainOnAllData(self, epochs, epochsBeforeUpdating):
    print("GATHERING DATA...")
    data = np.concatenate(self.manager.loadAllData())
    np.random.shuffle(data)

    if data.shape[0] == 0:
      raise Exception("No data in the data folder")

    s_time = time.time()
    print("TRAINING MODEL...")
    for b in range(epochs // epochsBeforeUpdating):
      # try:
      losses = self.gym.agent.train(epochsBeforeUpdating, 32, False, data = self.improveQualityOfData(data[:len(data)//2]), verbose = 0)
      losses = self.gym.agent.train(epochsBeforeUpdating, 32, False, data = self.improveQualityOfData(data[len(data)//2:]), verbose = 0)
      print(f"\tTIME: {round((time.time() - s_time) * 10) / 10}\tLOSS {(b * epochsBeforeUpdating)} / {epochs}: {losses}")
      # except:
      #   print("ERROR")
      #   self.gym.agent.updateTargetModel()
    print("SAVING MODELS...")
    self.gym.nEpisodes += 1
    self.gym.updateMetadata()
    self.gym.manager.saveSession(self.gym.agent, self.gym.nEpisodes)

  def improveQualityOfData(self, data):
    '''By adding this noise it helps prevent overfitting'''
    i = 0
    indexes = list(np.arange(len(data)))

    while i < len(indexes):
        self.addNoise(data[i]) 
        self.changeBrightness(data[i], np.random.uniform(0.95, 1.05))
        self.applyColorFilter(data[i], np.random.uniform(0.90, 1.1),np.random.uniform(0.90, 1.1),np.random.uniform(0.90, 1.1))
        # plt.imshow(data[i][0][0])
        plt.show()
        i += 1
    return data

  
  def addNoise(self, data, noiseFactor = 0.1):
    '''Will be input a piece of data like: (state, action, reward, done, next_state)'''  
    data[0][0] = data[0][0] + (np.random.random(size=data[0][0].shape) - 0.5) * noiseFactor ** 2
    data[-1][0] = data[-1][0] + (np.random.random(size=data[-1][0].shape) - 0.5) * noiseFactor ** 2
  
  
  def changeBrightness(self, data, brightnessFactor):
    '''Will be input a piece of data like: (state, action, reward, done, next_state)'''
    data[0][0] = data[0][0] * (brightnessFactor)
    data[-1][0] = data[-1][0] * (brightnessFactor)
  
  
  def applyColorFilter(self, data, redFactor, greenFactor, blueFactor):
    return
    data[0][0] = np.dot(data[0][0][:], np.array([redFactor, greenFactor, blueFactor]))
    data[-1][0] *= [redFactor, greenFactor, blueFactor]

# if __name__ == "__main__":
  
  # from gym import *
  # gym = Gym(False)
  # trainer = Trainer(gym)
  
  # trainer.trainOnAllData(64, 4)