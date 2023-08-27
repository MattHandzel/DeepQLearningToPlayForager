from pynput import keyboard, mouse
import pynput
from pynput.keyboard import Key, KeyCode
import time, random

### REDUNDANT?
'''class Mouse(mouse.Controller):
  def __init__(self, positionOffset):
    super(Mouse, self).__init__()
    self.positionOffset = positionOffset

  def setPosition(self, position):
    self.position = [position[0] + self.positionOffset[0], position[1] + self.positionOffset[1]]'''

class Input:
  '''This class will be used for the program to be able to control the keyboard and mouse so that it can interact with the game'''
  ### REDUNDANT?
  #inputHistory = []

  # Mouse and keyboard controllers
  keyboard = keyboard.Controller()
  mouse = pynput.mouse.Controller()
  
  # This will contain if keys are being pressed or not, this could be used in the future to record player input and feed it to the model
  keyStates = []

  # this is used in place of an enum, it will be sent to a setKeyState function to determine if the key is pressed or unpressed (released)
  pressed = "pressed"
  unpressed = "unpressed"
  
  # The possible keys pressed that relate to movement (they are separated because the ai has two outputs, one for movement another for action, which allows the ai to move while mining)
  movementInputs = [Key.right, Key.left, Key.up, Key.down, None]
  
  # The possible keys that relate to actions (i.e things that the player does not relating to movement), in the future, I might allow the agent to be able to eat food
  actionInputs = [KeyCode.from_char("e"), None]

  # these are all possible inputs, this is used when encoding actions for the model
  allInputs = [Key.right, Key.left, Key.up, Key.down, KeyCode.from_char("e"), None]

  # This is the key that will be used to mine ("e" on the keyboard)
  mineKey = KeyCode.from_char("e")
  # This is where the mouse will be placed on teh screen in order to mine
  minePosition = [1860, 590]
  
  # This is to see if the program has been terminated in order to stop the keyboard listener
  terminated = False

  def run(self):
    '''This will run the program, start the listeners and start time'''
    self.s_time = time.time()
    self.listener = keyboard.Listener(
        on_press=self.on_press,
        on_release=self.on_release)
    self.listener.start()
  
  def getTime(self):
    '''Gets the current time, in use if recording input from a human'''
    return time.time() - self.s_time

  def on_press(self, key):
    '''This function is called whenever a key is pressed on the keyboard'''

    # Checks if the key is a movement key or if it is a mining key
    if key in self.movementInputs or key == self.mineKey: # TODO:: Change this to be if key in self.allInputs (check to see if you need to exclude the None)
      ### REDUNDANT?
      '''if key not in self.keyStates:
        self.inputHistory.append([self.getTime(),key, True])'''

      # Checks if the key is the mining key
      if key == self.mineKey:
        # Sets the mouse position and presses the left button
        self.mouse.position = self.minePosition
        self.mouse.press(mouse.Button.left)
  
      # if the key is not in key states then set the key state to "pressed"
      if key not in self.keyStates:
        self.setKeyState(key, self.pressed)
      
  def on_release(self, key):
    '''This function is called whenever a key is released'''
    if (key in self.movementInputs or key == self.mineKey) and key in self.keyStates:
      ### REDUNDANT?
      '''self.inputHistory.append([self.getTime(),key, False])'''

      # Sets the key in self.keyStates to unpressed
      self.setKeyState(key, self.unpressed)
      
      # If we are releasing a mining key then stop mining
      if key == self.mineKey:
        self.mouse.release(mouse.Button.left)
      
      # if the key is no longer in key states then release the left button mouse
      

      ### REDUNDANT? Why the check are we releasing the mouse
      '''
      if not key in self.keyStates:
        self.mouse.release(mouse.Button.left)'''

    # If the key is escape, then stop the program
    elif key == Key.esc or key == Key.caps_lock:
      print("Stopping the listener...")
      self.terminated = True
      self.stopListener()
    
  def click(self, delay):
    self.mouse.press(mouse.Button.left)
    time.sleep(delay)
    self.mouse.release(mouse.Button.left)
  
  def setKeyState(self, key, state):
    '''Takes in a key and what state to set it to in the self.keyStates array'''
    if state == "pressed":
      self.keyStates.append(key)
    elif state == "unpressed":
      self.keyStates.remove(key)
  
  def stopListener(self):
    '''Stops the listener'''
    self.listener.stop()

  def getRandomActions(self):
    '''Returns a random choice of movement and action, this is used when the model is exploring'''
    return [random.choice(self.movementInputs), random.choice(self.actionInputs)]

if __name__ == "__main__":
  # Testing the input class
  player = Input() 
  player.run()
  while True:
    print(player.mouse.position)
    time.sleep(0.1)
