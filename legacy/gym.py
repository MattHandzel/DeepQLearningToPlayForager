from PIL.Image import PERSPECTIVE
from mapInput import *
from environment import *
from src.dataManager import Manager
import time


'''
TODO: FIX THE PROBLEM WITH HEALTH FLUCTUATING, i.e computer thinks we are losing and gaining health when we're not

'''
class Gym:
  input = Input()
  env = Environment()
  manager = Manager()
  episodeData = []
  data = []
  
  nEpisodes = 0
  nSteps = 0

  framerate = 5

  def __init__(self):
    self.input.run()

  def run(self):
    while not self.input.terminated:
      self.episode()
      self.manager.addEpisode(self.episodeData)
      self.printMetadataOfEpisode()
      self.episodeData.clear()

  def episode(self):
    alive = True
    current_state = None
    self.nEpisodes += 1
    while alive and not self.input.terminated:
      self.nSteps += 1
      actions = self.input.getRandomActions()
      for action in actions:
        if action != None:
          self.input.keyboard.press(action)
      next_state = self.env.getDataFromGame()
      time.sleep(1/self.framerate) 
      for action in actions:
        if action != None:
          self.input.keyboard.release(action)
      if current_state != None:
        self.episodeData.append([current_state, [self.input.movementInputs.index(actions[0]), self.input.actionInputs.index(actions[1])], self.env.computeReward(current_state[1:], next_state[1:]),next_state])
        if self.env.computeReward(current_state[1:], next_state[1:]) > 0.2:
          print(current_state[1:])
          print(next_state[1:])
          print("REWARD: " + str(self.env.computeReward(current_state[1:], next_state[1:])))
          print("\n")
      current_state = next_state
      if current_state[1] == 0:
        alive = False
        self.resetEnv()
        
        print("KILLING")
  def resetEnv(self):
    print("RESETING ENV")
    restarting = True
    delay = 0.15
    while restarting:
      while not self.env.checkIfMainScreen():
        time.sleep(1)
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
  def train(self):
    self.manager.loadData("")

if __name__ == "__main__":
  gym = Gym()
  print("GYM INITIALIZED")
  gym.run()