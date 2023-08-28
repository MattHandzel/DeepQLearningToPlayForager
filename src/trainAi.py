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

  def train(self, epochs, epochsBeforeUpdating, use_target=True):
    self.trainOnData(self.manager.loadRandomSample(), epochs=epochs, epochsBeforeUpdating=epochsBeforeUpdating, use_target=use_target)

  def trainOnData(self, original, epochs, epochsBeforeUpdating, use_target = True):
    data = np.copy(original)
    np.random.shuffle(data)

    s_time = time.time()
    print("TRAINING MODEL...")
    for b in range(epochs // epochsBeforeUpdating):
      # try:
      for i in range(epochsBeforeUpdating):
        print(f"\tTIME: {round((time.time() - s_time) * 10) / 10}\tLOSS {(b * epochsBeforeUpdating + i + 1)} / {epochs}: {np.sum(self.gym.agent.train(epochs = 1, batch_size=32, notGonnaBeData=False, data = self.improveQualityOfData(data), verbose = 0, use_target=use_target))}")
      # except:
      #   print("ERROR")
      #   self.gym.agent.updateTargetModel()
    print("SAVING MODELS...")
    self.gym.nEpisodes += 1
    self.gym.updateMetadata()
    self.gym.manager.saveSession(self.gym.agent, self.gym.nEpisodes)

  def trainOnAllData(self, epochs, epochsBeforeUpdating,use_target=True):
    print("GATHERING DATA...")
    self.trainOnData(np.concatenate(self.manager.loadAllData()), epochs=epochs, epochsBeforeUpdating=epochsBeforeUpdating, use_target=use_target)

  def improveQualityOfData(self, original):
    '''By adding this noise it helps prevent overfitting'''
    data = original.copy()
    i = 0
    if(data.shape == 3):
      data = np.expand_dims(data)
    indexes = list(np.arange(len(data)))

    while i < len(indexes):
        self.addNoise(data[i], 0.025) 
        self.changeBrightness(data[i], np.random.normal(1, scale=0.05))
        # self.applyColorFilter(data[i], np.random.uniform(0.90, 1.1),np.random.uniform(0.90, 1.1),np.random.uniform(0.90, 1.1))
        # plt.imshow(data[i][0][0])
        i += 1
    return data

  
  def addNoise(self, data, noiseFactor = 0.1):
    '''Will be input a piece of data like: (state, action, reward, done, next_state)'''  
    data[0] = data[0] + ((np.random.normal(0, noiseFactor, size=data[0].shape)))
    data[-1] = data[-1] + ((np.random.normal(0, noiseFactor, size=data[-1].shape)))
  
  
  def changeBrightness(self, data, brightnessFactor):
    '''Will be input a piece of data like: (state, action, reward, done, next_state)'''
    data[0] = data[0] * (brightnessFactor)
    data[-1] = data[-1] * (brightnessFactor)
  
  
  def applyColorFilter(self, data, redFactor, greenFactor, blueFactor):
    return
    data[0] = np.dot(data[0][:], np.array([redFactor, greenFactor, blueFactor]))
    data[-1] *= [redFactor, greenFactor, blueFactor]

if __name__ == "__main__":
  
  from gym import *
  gym = Gym(False)
  trainer = Trainer(gym)
  
  # all_data = gym.manager.loadAllData()
  # all_data = np.concatenate(all_data)


  # trainer.trainOnData(all_data,1, 1, False)

  data = gym.manager.loadRandomSample()

  plt.imshow(data[0][0])
  plt.show()
  data = trainer.improveQualityOfData(data)
  plt.imshow(data[0][0])
  plt.show()
