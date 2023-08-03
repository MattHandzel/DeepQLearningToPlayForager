import numpy as np
import pickle as pkl
import time, json, os, random
import matplotlib.pyplot as plt
from utils import * 
#from tensorflow.python.keras.mixed_precision import experimental
from agent import *
from environment import *

class Manager:
  
  folderpath = "./data/"
  modelPath = "./models/"
  metadataPath = "./metadata.json"
  
  env = Environment()

  def addData(self, data):
    '''Data will be in the format of: current_state, action, reward, next_state
       Each state will be (vision, hp, energy, xp)
       '''

    pkl.dump(data, open(self.folderpath + "data_" + str(round(time.time())) + ".p", "wb"))

  def loadData(self, path):
    return pkl.load(open(self.folderpath + path, "rb"))

  def addEpisode(self, data):
    pkl.dump(data, open(self.folderpath + "episode_" + str(round(time.time())) + ".p", "wb"))
  
  def addExperienceReplay(self, data):
    pkl.dump(data, open(self.folderpath + "experienceReplay_" + str(round(time.time())) + ".p", "wb"))

  def printMetadataOfEpisode(self, episodeData):
    episodeData = np.array(episodeData)
    episodeLength = len(episodeData[:,2])
    print(f"META DATA FOR EPISODE: {self.nEpisodes}")
    print(f"\tMISC:")
    print(f"\t\tNSTEPS: {episodeLength} ({episodeLength * (1/self.framerate)}) seconds")
    print("\tREWARDS:")
    print(f"\t\tTOTAL: {np.sum(episodeData[:,2])}")
    print(f"\t\tAVERAGE: {np.sum(episodeData[:,2])/episodeLength}")
    print(f"\t\tSTD:\t{np.std(episodeData[:,2])}")

  def loadMetadata(self):
    return json.load(open(self.metadataPath))
  
  def loadAllData(self):
    data = []
    for path in os.listdir(self.folderpath):
      data.append(pkl.load(open(self.folderpath + path, "rb")))
      
    if len(data) == 0:
      raise Exception("For some reason we can't get any data, does any data exist?")
    
    return data
  
  def loadRandomSample(self):
    path = random.choice(os.listdir(self.folderpath))
    return (pkl.load(open(self.folderpath + path, "rb")))

  def updateMetadata(self, metadata):
    json.dump(metadata, open(self.metadataPath, "w"), indent = 4)
  
  def seeHowGoodAgentIs(self, gym):
    data = self.loadRandomSample()
    movementData = []
    actionData = []
    for i in range(len(data)):
      current_state = data[i][0]
      predictions = gym.agent.predictRewardsForActions(current_state)
      #print("MOVEMENT:")
      movementData.append(predictions[0][0])
      actionData.append(predictions[1][0])
      #for i in range(len(predictions[0][0])):
      #  print(f"\t{gym.input.movementInputs[i]}:\t{round(predictions[0][0][i] * 1000) / 1000}")
      #print("ACTION:")
      #for i in range(len(predictions[1][0])):
      #  print(f"\t{gym.input.actionInputs[i]}:\t{round(predictions[1][0][i] * 1000) / 1000}")
      #print("\n")
    plt.plot(movementData)
    plt.show() 
    plt.plot(actionData)
    plt.show() 


  def loadPreviousSession(self, agent, nEpisodes = -1):

    if nEpisodes == -1:
      arr = os.listdir(self.modelPath)
      arr.sort()
    else:
      arr = [f"model_{formatNumbersWithZeros(nEpisodes, 4)}"]
    opt_weights = pkl.load(open(self.modelPath + arr[-1] + "/optimizer/weights.p","rb"))
    
    grad_vars = agent.model.trainable_weights
    # This need not be model.trainable_weights; it must be a correctly-ordered list of 
    # grad_vars corresponding to how you usually call the optimizer.
    
    optimizer = agent.optimizer
    
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    
    # Apply gradients which don't do nothing with Adam
    optimizer.apply_gradients(zip(zero_grads, grad_vars))
    
    # Set the weights of the optimizer
    optimizer.set_weights(opt_weights)

    # NOW set the trainable weights of the model
    model_weights = keras.models.load_model(self.modelPath + arr[-1]).weights
    agent.model.set_weights(model_weights)
  
  def saveSession(self, agent, nEpisodes):
    agent.model.save(f"./models/model_{formatNumbersWithZeros(nEpisodes, 4)}")
    os.mkdir(f"./models/model_{formatNumbersWithZeros(nEpisodes, 4)}/optimizer/")
    # pkl.dump(agent.optimizer., open(f"./models/model_{formatNumbersWithZeros(nEpisodes, 4)}/optimizer/weights.p", "wb"))

  def compresSizeOfdata(self):
    '''This function wil make the state image be in float16 instead of float32 which about halves the space the data takes up'''
    '''
    Data shape:
    (x, 5)
    - state:
      - (68, 135) (image)
      - (1,) health
      - (1,) energy
      - (1,) xp
    - action
    - reward
    - done
    - next_state
      - (68, 135) (image)
      - (1,) health
      - (1,) energy
      - (1,) xp
    '''
    for path in os.listdir(self.folderpath):
      data = pkl.load(open(self.folderpath + path, "rb"))
      for i in range(len(data)):
        data[i][0] = list(data[i][0])
        data[i][0][0] = np.array(data[i][0][0]).astype(np.float16)
        data[i][-1] = list(data[i][-1])
        data[i][-1][0] = np.array(data[i][-1][0]).astype(np.float16)

      pkl.dump(data, open(self.folderpath + path, "wb"))
    
  def modifyRewardValues(self):
    '''will remove how much xp agent has from the data, im hoping that this will remove the plateaus in when the reward per action is plotted vs time (where it is constant-ish then it jumps when the xp increases)
       by removing this im hoping to remove that, another thing is that it might be a problem with my reward system, imma check that out'''
    '''
    Data shape:
    (x, 5)
    - state:
      - (68, 135) (image)
      - (1,) health
      - (1,) energy
      - (1,) xp
    - action
    - reward
    - done
    - next_state
      - (68, 135) (image)
      - (1,) health
      - (1,) energy
      - (1,) xp
    '''
    for path in os.listdir(self.folderpath)[1:]:
      print("NOW DOING: " + path)
      data = pkl.load(open(self.folderpath + path, "rb"))
      
      # add the for loop
      for i in range(len(data)):
        if i % 100 == 0:
          print(f"PREVIOUS REWARD IS: {data[i][2]}, NEW REWARD IS {self.env.computeReward(data[i][0][1:], data[i][-1][1:])}") 
        data[i][2] = self.env.computeReward(data[i][0][1:], data[i][-1][1:])

      pkl.dump(data, open(self.folderpath + path, "wb"))
    

if __name__ == "__main__":
  manager = Manager()
  print(manager.loadRandomSample()[0][2])
  manager.modifyRewardValues()
  print(manager.loadRandomSample()[0][2])
  