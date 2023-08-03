from PIL import ImageGrab, Image
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl

class Environment:
  '''This class will be used as the environment, it will give the states of the agent and the rewards'''

  # Dimensions of the game screen 
  screenX0 = 960
  screenX1 = 1920
  screenY0 = 360
  screenY1 = 785
  
  heartYPosition = screenY0 + 10
  # Position that we will check for each heart
  heart1Position = (1019 - 44, heartYPosition)
  heart2Position = (1041 - 44, heartYPosition)
  heart3Position = (1064 - 44, heartYPosition)
  
  # The length of the heart (we will be checking in a 7x7 grid, this is used to prevent a bad reading)
  heartLength = 7
  
  ## Rewards for different actions:
  # A time punishment for each second, this prevents the agent from just standing still
  timePunishment = -0.01
  # A reward (+/-) if the agent gains/loses a heart
  healthReward = 0
  # A reward (+/-) if the agent gains/loses energy
  energyReward = 0.00
  # A reward if the agent gets xp, multiplied by the amount of xp it gains
  xpRewardMultiplier = 0.05
  
  # The color of the xp bar, this is used to determine how much xp the agent has
  xpBarColor = np.array([0.992, 0.969, 0.424])

  def __init__(self):
    # This loads the masks, masks will be used to determine things like, how much health that we have, and it will be doing that by compaing the difference of two images of the hearts, 
    # one from the game and one from the mask, this will then tell the "heartness of the heart" and we use that value to determine health, we do this for things like health, the main screen, if its a game over, etc.
    self.loadMasks()
    
  def run(self, framerate = 5):
    '''This will run the environment, this is used for testing purposes to see if the game is outputting the correct values, it gets a framerate which is how often we check for the values'''
    while True:
      state = self.getDataFromGame()
      # plt.imshow(state[0])
      # plt.show()
      print("The predicted stats are:")
      print(f"Health:\t{state[1]}\tEnergy:\t{state[2]}\tXp:\t{state[3]}\n")
      # print(state[1:])
      time.sleep(1/framerate)

  def getGameScreen(self):
    '''This returns the game screen so we can get pixel values from it'''
    return np.array(ImageGrab.grab(bbox=(self.screenX0,self.screenY0,self.screenX1,self.screenY1)), dtype = np.float32) * (1/255)
  
  def getDataFromGame(self):
    '''This function gets data from the game, it uses other functions to get things like health, energy, and xp'''
    gameScreen = self.getGameScreen()
    aiVision = self.grayscale(gameScreen)[50:390, 125:800][::5,::5].astype(np.float16)
    return [aiVision, self.getHealthData(gameScreen), self.getEnergyData(gameScreen), self.getXpData(gameScreen)]
  
  def computeReward(self, previousState, currentState):
    '''Computes how much reward the agent should get based upon the reward values and how much of that thing it gains/loses (ex: mining a flower gains less xp then mining a rock)'''
    reward = self.timePunishment
    if currentState[0] - previousState[0] < 0: # If health now is less than health before
      reward += self.healthReward * (currentState[0] - previousState[0]) 
    if currentState[1] - previousState[1] < 0: # If energy now is less than energy before
      reward += self.energyReward * (currentState[1] - previousState[1])
    if currentState[2] - previousState[2] > 0: # If xp now is greater than xp before
      reward += (currentState[2] - previousState[2]) * self.xpRewardMultiplier

    return reward
    
  def getHealthData(self, gameScreen):
    '''Inputs the game screen, and it returns how much health the agent has as an int from 0-3'''
    # It gets the relative positions of the hearts, this is because we get the location of the hearts by using mouse.position, which are two numbers from like 0-1920, and they show where 
    # the mouse is in relation to the screen, but the array we send is just the game screen, so we have to compensate for that difference
    relativeHeart1Position = [self.heart1Position[0] - self.screenX0, self.heart1Position[1] - self.screenY0][::-1]
    relativeHeart2Position = [self.heart2Position[0] - self.screenX0, self.heart2Position[1] - self.screenY0][::-1]
    relativeHeart3Position = [self.heart3Position[0] - self.screenX0, self.heart3Position[1] - self.screenY0][::-1]
    
    # finds the hearts from the game screen
    heart1 = gameScreen[relativeHeart1Position[0] - self.heartLength: relativeHeart1Position[0] + self.heartLength, relativeHeart1Position[1] - self.heartLength : relativeHeart1Position[1] + self.heartLength]
    heart2 = gameScreen[relativeHeart2Position[0] - self.heartLength: relativeHeart2Position[0] + self.heartLength, relativeHeart2Position[1] - self.heartLength : relativeHeart2Position[1] + self.heartLength]
    heart3 = gameScreen[relativeHeart3Position[0] - self.heartLength: relativeHeart3Position[0] + self.heartLength, relativeHeart3Position[1] - self.heartLength : relativeHeart3Position[1] + self.heartLength]
    
    # Makes an array of hearts
    hearts = [heart1, heart2, heart3]
    
    # Make arrays of heartness and deadness of each heart screenshot
    heartnesses = []
    deadnesses = []

    # For each heart in the hearts, we compare the hearts to an alive heart mask and a dead heart mask, we find the difference squared of the screenshot and the mask
    for i in range(len(hearts)):
      heartnesses.append(np.sum((hearts[i] - self.heartMasks[i]) ** 2) / (3 * self.heartLength ** 2))
      deadnesses.append(np.sum((hearts[i] - self.lostHeartMasks[i]) ** 2) / (3 * self.heartLength ** 2))
    
    # This is where we calculate the amount of hearts we have, we compare which value is lower (because lower difference between the images means that they are more similar),
    # if the heartness at index i is higher, we get a true value, and we convert that to a 1
    totalHearts = 0
    for i in range(len(heartnesses)):
      totalHearts += int(heartnesses[i] < deadnesses[i])
    
    return totalHearts
  
  def getXpData(self, image):
    '''Inputs the game screen and returns a number which is how much xp the agent has at the moment'''

    # We get the dimensions of the xp bar, and then we just get one line, the amount of xp we have is the length of the xp bar where the color is self.xpBarColor (which is a light yellow)
    xpBar = image[8:9,244:716].transpose([1,0,2])
    
    # This is where we compare the difference to an arbitrary value, I chose 0.2 because i got a good result with a squared difference of 0.2
    totalXp = 0
    for pixel in xpBar:
      totalXp += int((np.sum((pixel - self.xpBarColor) ** 2)) < 0.2)
        
    return totalXp
  
  def getEnergyData(self, image):
    '''Inputs the game screen and it returns the amount of energy that the player has'''
    # We get the dimensions of the energy bar, and then we just get one line, the amount of energy we have is sum of the sum of the pixels and if they are above 0.3, 
    # this is because the energy bar's color when there is energy is a green, and when it is emptied, it is close to a black. 
    energyBar = image[28:29, 6:46][0]
    totalEnergy = 0

    # This is where that summination comes in
    for pixel in energyBar:
      if sum(pixel) > 0.3:
        totalEnergy += 1
    return totalEnergy

  def loadMasks(self):
    '''This functions loads the masks which wil be used later to identify images in the game scren'''
    self.heartMasks = pkl.load(open("./masks/heartMasks.p", "rb"))
    self.lostHeartMasks = pkl.load(open("./masks/deadHeartMasks.p","rb"))
    self.gameOverMask = pkl.load(open("./masks/gameOverMask.p", "rb"))
    self.mainScreenMask = pkl.load(open("./masks/mainScreenMask.p", "rb"))

  def grayscale(self, image):
    '''this function takes in image and returns it to be grayscaled, this is useful because it greatly reduces the dimensionality of the problem, 
       and reduces the size of the model by a big factor, allowing it to train faster and allowing us to make the model bigger'''
    # The numbers in order to get the image in grayscale were taken from stack overflow when someone asked how to grayscale an image 
    return np.sum(image * np.array([[[0.2989, 0.587, 0.114]]]), axis = 2)
    
  def calibrateHeartness(self, nHeartsAlive):
    '''Input is how many hearts are alive (inputted by human), it then updates the heart masks. The purpose of this function is to get the average of the heart image over a length of time, we average this to create the heart masks,
       as well as the dead heart masks. Averaging this is necessary because the heart increases and decreases in size (it's the animation that the
       heart plays while it is "beating"), which causes the program to report that we are gaining and losing health, averaging this for the alive 
       heart masks (they are just called heart masks), and dead heart masks solves this problem'''
    
    # Initialize the arrays which will hold the images that will be averaged
    self.heart1Alive = []
    self.heart1Dead = []
    self.heart2Alive = []
    self.heart2Dead = []
    self.heart3Alive = []
    self.heart3Dead = []
    s_time = time.time()
    
    # Spend three seconds calibrating
    while time.time() - s_time < 3:
      # Gets the game screen and it outputs what the function currents thinks is how much health we have (for testing pr)
      picture = self.getGameScreen()
      hearts = self.getHealthData(picture) # this function originally returned just the three images of the hearts
      if nHeartsAlive == 1:
        self.heart1Alive.append(hearts[0])
        self.heart2Dead.append(hearts[1])
        self.heart3Dead.append(hearts[2])
      elif nHeartsAlive == 2:
        self.heart1Alive.append(hearts[0])
        self.heart2Alive.append(hearts[1])
        self.heart3Dead.append(hearts[2])
      elif nHeartsAlive == 3:
        self.heart1Alive.append(hearts[0])
        self.heart2Alive.append(hearts[1])
        self.heart3Alive.append(hearts[2])
      time.sleep(1/20)
    
    # After averaging the amount of hearts we wanted to average, we save it, this is not done programmatically
    averageHeart1 = np.sum(np.array(self.heart1Alive), axis = 0) / len(self.heart1Alive)
    plt.imshow(averageHeart1)
    plt.show()
    if input("save?").upper() == "Y":
      pkl.dump([averageHeart1], open("./masks/firstLostHeartMask.p", "wb"))
    
  def checkIfMainScreen(self):
    '''Checks if we are at the main screen yet, if we are then return true, by using the main screen mask'''
    return (np.sum((self.mainScreenMask - self.getGameScreen()[80:140,30:600]) ** 2) / (self.mainScreenMask.shape[0] * self.mainScreenMask.shape[1] * self.mainScreenMask.shape[2])) < 0.01
  
  def setMainScreenMask(self):
    pkl.dump(self.getGameScreen()[80:140,30:600], open("./masks/mainScreenMask.p", "wb"))

  def checkIfGameOver(self):
    '''checks if it is a game over, if so, return true, by using the game over mask'''
    gameOver = self.getGameScreen()[130:150,350:450]
    return (np.sum((gameOver - self.gameOverMask) ** 2) / (gameOver.shape[0] * gameOver.shape[1] * gameOver.shape[2])) < 0.01


if __name__ == "__main__":
  # This is to test to see if the environment outputs the correct values
  env = Environment()
  # env.run(framerate=1)
  # env.setMainScreenMask()
  print(env.checkIfMainScreen())
