from PIL.Image import PERSPECTIVE
#from tensorflow.python.framework.ops import add_to_collections
from mapInput import *
from environment import *
from agent import *
from utils import *
from dataManager import Manager

import time, keras, os

class Gym:
  input = Input()
  env = Environment()
  manager = Manager()
  
  action_space_size = [5, 2]

  data = []
  
  nEpisodes = 0
  nSteps = 0

  framerate = 5
  actionsArray = []
  
  trainTimes = 10

  explorationRate = 0.4

  def __init__(self, load = True):
    self.agent = Agent(self)
    # input dims = [[68, 135], 3]
    # output dims = 6
    self.agent.define_model([[68, 135, 1], 2], self.action_space_size)
    stats = self.manager.loadMetadata()
    self.nEpisodes = stats["nEpisodes"]
    self.nSteps = stats["nSteps"]
    if load:
      temp = self.trainTimes
      self.trainTimes = 1
      self.trainOnPrevData(1)
      self.trainTimes = temp
      self.manager.loadPreviousSession(self.agent, self.nEpisodes)
    self.input.run()
    
  
  def initallyTrain(self):
    self.agent.updateTargetModel()
    self.trainOnPrevData(500000)
    input("...")
  
  def trainOnPrevData(self,nSteps):
    totalSteps = 0
    while totalSteps < nSteps:
      print(f"\n{totalSteps}/{nSteps}")
      try:
        self.agent.experienceReplay = self.manager.loadRandomSample()
        self.train()
        print(len(self.agent.experienceReplay))
        totalSteps += len(self.agent.experienceReplay) * self.trainTimes
      except Exception as e:
        pass
    self.agent.experienceReplay = ReplayExperience(maxlen = 5000)

  def runHeuristic(self):
    while not self.input.terminated:
      self.heuristicEpisode()

  def run(self):
    while not self.input.terminated:
      self.episode()
      self.betweenEpisodes()
      print("SAVING MODELS...")
      self.manager.saveSession(self.agent, self.nEpisodes)
  
  def betweenEpisodes(self):
    pass

  def updateMetadata(self):
    self.manager.updateMetadata(self.getStats())
  
  def getStats(self):
    return {"nEpisodes" : self.nEpisodes,
            "nSteps" : self.nSteps}

  def episode(self):
    # check if training
    #   - if training:
    #       train
    # check if updating target model
    # update to history
    
    alive = True
    
    current_state = self.env.getDataFromGame()
    self.nEpisodes += 1

    while alive and not self.input.terminated:
      self.nSteps += 1
      # run env step (get action from model)
      if random.random() > self.explorationRate:
        predictions = self.agent.predictRewardsForActions(current_state)
        actions = [np.argmax(predictions[0]), np.argmax(predictions[1])]
      else:
        actions = [random.randint(0,self.action_space_size[0]-1),random.randint(0,self.action_space_size[1]-1)] 
      
      if self.input.movementInputs[actions[0]] != None:
        self.input.keyboard.press(self.input.movementInputs[actions[0]])
      if self.input.actionInputs[actions[1]] != None:
        self.input.keyboard.press(self.input.actionInputs[actions[1]])
      next_state = self.env.getDataFromGame()
      time.sleep(1/self.framerate) 

      if self.input.movementInputs[actions[0]] != None:
        self.input.keyboard.release(self.input.movementInputs[actions[0]])
      if self.input.actionInputs[actions[1]] != None:
        self.input.keyboard.release(self.input.actionInputs[actions[1]])
      
      # set data to experience replay
      self.agent.experienceReplay.append([current_state, actions, self.env.computeReward(current_state[1:], next_state[1:]), next_state[1] == 0,  next_state])
      
      ### REDUNDANT?
      '''if self.env.computeReward(current_state[1:], next_state[1:]) > 0.2:
        print(current_state[1:])
        print(next_state[1:])
        print("REWARD: " + str(self.env.computeReward(current_state[1:], next_state[1:])))
        print("\n")'''
      current_state = next_state
      if self.explorationRate > 0.1:
        self.explorationRate *= 0.9998
      if current_state[1] == 0:
        alive = False
        self.manager.addEpisode(np.array(self.agent.experienceReplay))
        self.printMetadataOfEpisode(self.agent.experienceReplay)
        self.agent.experienceReplay.clear()
        self.updateMetadata()
        self.resetEnv()
        print("KILLING")

  def showAiPredictions(self):
    current_state = self.env.getDataFromGame()
    predictions = self.agent.predictRewardsForActions(current_state)
    print("MOVEMENT:")
    for i in range(len(predictions[0][0])):
      print(f"\t{self.input.movementInputs[i]}:\t{round(predictions[0][0][i] * 1000) / 1000}")
    print("ACTION:")
    for i in range(len(predictions[1][0])):
      print(f"\t{self.input.actionInputs[i]}:\t{round(predictions[1][0][i] * 1000) / 1000}")
    print("\n")

  def heuristicEpisode(self):
    # check if training
    #   - if training:
    #       train
    # check if updating target model
    # update to history
    
    alive = True
    
    current_state = self.env.getDataFromGame()
    self.nEpisodes += 1

    while alive and not self.input.terminated:
      self.nSteps += 1
      # run env step (get action from model)
      predictions = self.agent.predictRewardsForActions(current_state)
      print("MOVEMENT:")
      for i in range(len(predictions[0][0])):
        print(f"\t{self.input.movementInputs[i]}:\t{round(predictions[0][0][i] * 1000) / 1000}")
      print("ACTION:")
      for i in range(len(predictions[1][0])):
        print(f"\t{self.input.actionInputs[i]}:\t{round(predictions[1][0][i] * 1000) / 1000}")
      actions = [np.argmax(predictions[0]), np.argmax(predictions[1])]

      if self.input.movementInputs[actions[0]] != None:
        self.input.keyboard.press(self.input.movementInputs[actions[0]])
      if self.input.actionInputs[actions[1]] != None:
        self.input.keyboard.press(self.input.actionInputs[actions[1]])
      next_state = self.env.getDataFromGame()
      time.sleep(1/self.framerate) 

      if self.input.movementInputs[actions[0]] != None:
        self.input.keyboard.release(self.input.movementInputs[actions[0]])
      if self.input.actionInputs[actions[1]] != None:
        self.input.keyboard.release(self.input.actionInputs[actions[1]])
      
      # set data to experience replay
      self.agent.experienceReplay.append([current_state, actions, self.env.computeReward(current_state[1:], next_state[1:]), next_state[1] == 0,  next_state])
      if self.env.computeReward(current_state[1:], next_state[1:]) > 0.2:
        print(current_state[1:])
        print(next_state[1:])
        print("REWARD: " + str(self.env.computeReward(current_state[1:], next_state[1:])))
        print("\n")
      current_state = next_state
      if current_state[1] == 0:
        alive = False
        self.manager.addEpisode(np.array(self.agent.experienceReplay))
        self.printMetadataOfEpisode(self.agent.experienceReplay)
        self.agent.experienceReplay.clear()
        self.updateMetadata()
        self.resetEnv()
        print("KILLING")
  def resetEnv(self):
    print("RESETING ENV")
    restarting = True
    delay = 0.15
    while restarting:
      while not self.env.checkIfMainScreen():
        self.trainOnPrevData(1000)
        self.input.keyboard.press('e')
      self.trainOnPrevData(50000)
      # press play button
      self.input.mouse.position = (1116, 425)
      time.sleep(delay)
      self.input.mouse.press(mouse.Button.left)
      time.sleep(delay)
      self.input.mouse.release(mouse.Button.left)
      time.sleep(delay)
      # Press delete button
      self.input.mouse.position = (1819, 399)
      time.sleep(delay)
      self.input.mouse.press(mouse.Button.left)
      time.sleep(delay)
      self.input.mouse.release(mouse.Button.left)
      time.sleep(delay)
      # NO POSITIOn
      #self.input.mouse.position = (1545, 551)
      # YES POSITIOn
      self.input.mouse.position = (1367, 546)
      time.sleep(delay)
      self.input.mouse.press(mouse.Button.left)
      time.sleep(delay)
      self.input.mouse.release(mouse.Button.left)
      time.sleep(delay)
      self.input.mouse.position = (1441, 518)
      time.sleep(delay)
      self.input.mouse.press(mouse.Button.left)
      time.sleep(delay)
      self.input.mouse.release(mouse.Button.left)
      # Classic mode
      self.input.mouse.position = (1154, 542)
      # Single island challenge
      # self.input.mouse.position = (1534, 539)
      time.sleep(delay)
      time.sleep(delay)
      self.input.mouse.press(mouse.Button.left)
      time.sleep(delay)
      self.input.mouse.release(mouse.Button.left)
      time.sleep(delay)
      self.input.mouse.position = (1453, 699)
      time.sleep(delay)
      self.input.mouse.press(mouse.Button.left)
      time.sleep(delay)
      self.input.mouse.release(mouse.Button.left)
      time.sleep(delay)
      time.sleep(2.5)
      self.input.mouse.press(mouse.Button.right)
      time.sleep(delay)
      self.input.mouse.release(mouse.Button.right)
      time.sleep(delay)
      self.input.mouse.position = (1256, 350)
      time.sleep(delay)
      self.input.mouse.press(mouse.Button.left)
      time.sleep(delay)
      self.input.mouse.release(mouse.Button.left)
      restarting = False
    print("ENV READY")

  def endSession(self):
    self.manager.addData(self.data)
  
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
    print(f"\t\tMIN: {np.min(episodeData[:,2])}\t\tMAX: {np.max(episodeData[:,2])}")
  
  def train(self):
    print("CURRENTLY TRAINING MODEL...")
    for i in range(self.trainTimes):
      print(f"LOSS {i+1} / {self.trainTimes}: {np.sum(self.agent.train(1))}")
    print("UPDATING TARGET MODEL")
    self.agent.updateTargetModel()

if __name__ == "__main__":
  gym = Gym()
  gym.run()