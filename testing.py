from dataManager import Manager
import matplotlib.pyplot as plt
import numpy as np
from gym import *
gym = Gym()
manager = Manager()
data = np.array(manager.loadData("episode_1636835948.p"))
print(data.shape)
print(gym.agent.model.predict([np.expand_dims(np.expand_dims(data[0][0][0],-1),0), np.expand_dims(data[0][0][1:],0)]))
gym.agent.experienceReplay = data
losses = []
states, _, _ = gym.agent.processDataforTraining()
rewards = np.array(data)[:,2]
preds = gym.agent.model.predict(states)
plt.plot(rewards, label = "true")
plt.plot(np.max(preds[0], 1), label = "pred")
plt.show()