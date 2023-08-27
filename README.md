# Playing Forager with ML (Deep Q Learning)

This is a project I made in the summer of 2021 because I was addicted to a game called forager, which looks like:

Basically, the player can mine resources and has to fight against monsters similar to other games.
The motivation behind this project is that a lot of the mechanics in the game require a lot of *mindless* activity, where the player collects resources and the only stimuating thing being the occasional monster that pops out, so why not make an AI that can do this for us?

Here is how it works: (which is REALLY inefficient, but hey, I wanted it work on my local save file)

Because I had the game on my phone I used scrcpy to mirror my phone screen onto my computer, and I used computer vision to read the game state from the pixels on screen. I have an **environment** class that manages all of this:

![7ukj0n](https://github.com/MattHandzel/PlayingForager/assets/39449480/f0402d6a-92ac-4474-9d36-439a50d272dd)

The notable states for this project is the agent's health, current experience points, as well as their energy. When these values change then the environment can detect this and let the agent know.

The agent also sees a black-and-white area surrounding it as the input:

![image](https://github.com/MattHandzel/PlayingForager/assets/39449480/0992fab5-0f79-49df-b2a9-71121997339b)

From that, it outputs two vectors, the first one being the movement vector and the agent predicts which direction (right, left, up, down, none) will maximize its reward function. The second output is where the agent predicts which action (mining or no mining) will maximize its reward function as well. The agent selects the action with the highest reward.

With the current state of the project, the agent gets penalized 0.01 reward for every timestep and it gains reward whenever it mines a resource (the better the resource the more reward) this is tracked through the experience points the agent.
