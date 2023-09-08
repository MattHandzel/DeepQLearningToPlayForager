from PIL.Image import PERSPECTIVE
#from tensorflow.python.framework.ops import add_to_collections
from mapInput import *
from environment import *
from agent import *
from utils import *
from dataManager import Manager
import time, keras, os

class Gym:
  env = Environment()
  manager = Manager()
  
  action_space_size = [5, 2]

  data = []
  
  nEpisodes = 0
  nSteps = 0

  framerate = 5
  actionsArray = []
  
  trainTimes = 10

  explorationRate = 0.9



  def __init__(self, load = True):
    self.agent = Agent(self)

    self.input = Input()
    # input dims = [[68, 135], 3]
    # output dims = 6
    input_vision_dimensions = [50, 50, 1]
    self.agent.define_model(input_vision_dimensions, self.action_space_size)
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
      self.agent.experienceReplay = self.manager.loadRandomSample()
      self.train()
      print(len(self.agent.experienceReplay))
      totalSteps += len(self.agent.experienceReplay) * self.trainTimes
    self.agent.experienceReplay = ReplayExperience(maxlen = 5000)

  def runHeuristic(self):
    while not self.input.terminated:
      self.heuristicEpisode()

  def run(self):
    if(self.trainer == None):
      raise Exception("YOU NEED TO SET THE TRAINER (gym.setTrainer(trainer))")
    print("Running...")
    print(self.input.terminated)
    while not self.input.terminated:
      print("Starting the episode..")
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
    
    start_steps = self.nSteps

    game_data = self.env.getDataFromGame()
    current_state = game_data[0]
    self.nEpisodes += 1

    while alive and not self.input.terminated:
      self.nSteps += 1

      if random.random() > self.explorationRate:
        predictions = self.agent.predictRewardsForActions(current_state)
        actions = [np.argmax(predictions[0]), np.argmax(predictions[1])]
      else:
        actions = [random.randint(0,self.action_space_size[0]-1),random.randint(0,self.action_space_size[1]-1)] 
      
      if self.input.movementInputs[actions[0]] != None:
        self.input.keyboard.press(self.input.movementInputs[actions[0]])
      if self.input.actionInputs[actions[1]] != None:
        self.input.keyboard.press(self.input.actionInputs[actions[1]])

      next_game_data = self.env.getDataFromGame()
      next_state = next_game_data[0]

      time.sleep(1/self.framerate)

      if self.input.movementInputs[actions[0]] != None:
        self.input.keyboard.release(self.input.movementInputs[actions[0]])
      if self.input.actionInputs[actions[1]] != None:
        self.input.keyboard.release(self.input.actionInputs[actions[1]])
      
      # set data to experience replay
      self.agent.experienceReplay.append([current_state, actions, self.env.computeReward(game_data[1:], next_game_data[1:]), next_state[1] == 0,  next_state])
      print(game_data[1:])
      game_data = next_game_data
      current_state = next_state
      if self.explorationRate > 0.5:
        self.explorationRate *= 0.999998
      
      if game_data[1] == 0:
        alive = False
        if self.nSteps - start_steps >= 10: # This is so that if the episode is really short we dont save it
          self.manager.addEpisode(np.array(self.agent.experienceReplay, dtype=object))
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

      if random.random() > self.explorationRate:
        predictions = self.agent.predictRewardsForActions(current_state)
        actions = [np.argmax(predictions[0]), np.argmax(predictions[1])]
      else:
        actions = [random.randint(0,self.action_space_size[0]-1),random.randint(0,self.action_space_size[1]-1)] 
      
      if self.input.movementInputs[actions[0]] != None:
        self.input.keyboard.press(self.input.movementInputs[actions[0]])
      if self.input.actionInputs[actions[1]] != None:
        self.input.keyboard.press(self.input.actionInputs[actions[1]])

      next_game_data = self.env.getDataFromGame()
      next_state = next_game_data[0]

      time.sleep(1/self.framerate) 

      if self.input.movementInputs[actions[0]] != None:
        self.input.keyboard.release(self.input.movementInputs[actions[0]])
      if self.input.actionInputs[actions[1]] != None:
        self.input.keyboard.release(self.input.actionInputs[actions[1]])
      
      # set data to experience replay
      self.agent.experienceReplay.append([current_state, actions, self.env.computeReward(game_data[1:], next_game_data[1:]), next_state[1] == 0,  next_state])
      
      current_state = next_state
      game_data = next_game_data
      if game_data[1] == 0: 
        alive = False
        if(self.nSteps > 5):
          self.manager.addEpisode(np.array(self.agent.experienceReplay, dtype=object))
        self.printMetadataOfEpisode(self.agent.experienceReplay)
        self.agent.experienceReplay.clear()
        self.updateMetadata()
        self.resetEnv()
        print("KILLING")

  def resetEnv(self):
    print("RESETING ENV")
    restarting = True
    delay = 0.25
    while restarting:
      while not self.env.checkIfMainScreen():
        self.train(1,1)
        self.input.keyboard.press('e')
      # self.trainOnPrevData(50000)
      # press play button
      y_offset = 30
      self.input.mouse.position = (1116, 425 + y_offset)
      time.sleep(delay)
      self.input.click(delay)
      time.sleep(delay)
      # Press delete button
      self.input.mouse.position = (1819, 399 + y_offset)
      time.sleep(delay)
      self.input.click(delay)
      time.sleep(delay)
      # NO POSITIOn
      #self.input.mouse.position = (1545, 551 + y_offset)
      # YES POSITIOn
      self.input.mouse.position = (1367, 546 + y_offset)
      time.sleep(delay)
      self.input.click(delay)
      time.sleep(delay * 2)
      self.input.mouse.position = (1441, 518 + y_offset)
      time.sleep(delay)
      self.input.click(delay)
      # Classic mode
      self.input.mouse.position = (1154, 542 + y_offset)
      # Single island challenge
      # self.input.mouse.position = (1534, 539 + y_offset)
      time.sleep(delay)
      self.input.click(delay)
      time.sleep(delay)
      self.input.mouse.position = (1453, 699 + y_offset)
      time.sleep(delay)
      self.input.click(delay)
      time.sleep(delay)
      time.sleep(2.5)
      self.input.mouse.press(mouse.Button.right)
      time.sleep(delay)
      self.input.mouse.release(mouse.Button.right)
      time.sleep(delay)
      self.input.mouse.position = (1256, 350 + y_offset)
      time.sleep(delay)
      self.input.click(delay)
      restarting = False
    print("ENV READY")
  
  def setTrainer(self, trainer):
    self.trainer = trainer

  def endSession(self):
    self.manager.addData(self.data)
  
  def printMetadataOfEpisode(self, episodeData):
    episodeData = np.array(episodeData, dtype=object)
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
      self.trainer.train(1, 1)
      # print(f"LOSS {i+1} / {self.trainTimes}: {np.sum(self.agent.train(epochs=1))}")
    print("UPDATING TARGET MODEL")
    self.agent.updateTargetModel()

if __name__ == "__main__":
  gym = Gym(False)

  from trainAi import Trainer
  trainer = Trainer(gym)
  gym.setTrainer(trainer)
  # try:
  gym.run()
  # except KeyboardInterrupt or Exception or OSError or ValueError as e:
  #   print(e)
  #   gym.input.stopListener()