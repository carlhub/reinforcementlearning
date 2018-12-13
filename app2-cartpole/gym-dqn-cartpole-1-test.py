
import random
import gym
import numpy as np
from gym import envs
# import tensorflow
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('MsPacman-v0')
env = gym.make('Copy-v0')
env = gym.make('CartPole-v0')
env = gym.make('CarRacing-v0')
env = gym.make('Assault-v0')
env = gym.make('MsPacman-v0')
env =gym.make('MontezumaRevenge-v0')#has v4
env = gym.make('DemonAttack-ram-v0')
#env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')

#Globals
EPISODES=5000

"""
env.reset() # loads up environemnt
for _ in range(1000)
        env.render() #displays a gui window
        env.step(env.action_space.sample()) #adjust current env by inputting an action
"""
# def wrapper_monitor():
#       print("wrapper:")
# def tester2():
#       for i in range(0,10): #
#               print(" ")
#               print("Episode: ",i)
#               print(" ")
#               env.reset()
#               j=0
#               #print("The Registry:  ",envs.registry.all())
#               print("Action Space size=env.action_space.n = ",env.action_space.n )
#               print("State_Size= ")
#               print(env.observation_space.shape[0])
#               # return #break
#               while True: #
#                       j=j+1
#                       print("Timestampe {}".format(j) )       #env.render()
#                       #env.render()
#                       action=env.action_space.sample()        # get action from action space
#                       obs,reward,done,info=env.step(action)   # New state after action in old state
#                       #print("Observation: ",obs)
#                       if done:
#                               print("DONE after {} timesteps".format(j+0))
#                               break



# Game Driver
#
class DQNAgent:
        def __init__(self,state_size,action_size):
                #print("test:DQNAgent")
                self.state_size=state_size
                self.action_size=action_size
                self.memory=deque(maxlen=2000)
                self.gamma=0.95
                self.epsilon=1.0
                self.epsilon_min=0.01
                self.epsilon_decay=0.995
                self.learning_rate=0.001
                self.model=self._build_model()

        def _build_model(self):
                #NN
                model=Sequential()
                model.add(Dense(24,input_dim=self.state_size,activation='relu'))
                model.add(Dense(24,activation='relu'))
                model.add(Dense(self.action_size, activation='linear'))
                model.compile(loss='mse',
                                                optimizer=Adam(lr=self.learning_rate))
                return model
        def remember(self,state,action,reward,next_state,done):
                #print('')
                self.memory.append((state,action,reward,next_state,done))

        def act(self,state):
                if np.random.rand() <= self.epsilon:
                        return random.randrange(self.action_size)
                act_values=self.model.predict(state)
                return np.argmax(act_values[0])#returns
        def replay(self,batch_size):
                minibatch=random.sample(self.memory,batch_size)
                for state,action,reward,next_state,done in minibatch:
                        target=reward
                        if not done:
                                target=(reward + self.gamma*
                                                np.amax(self.model.predict(next_state)[0]))
                        target_f=self.model.predict(state)
                        target_f[0][action]=target
                        self.model.fit(state,target_f,epochs=1,verbose=0)
                if self.epsilon > self.epsilon_min:
                        self.epsilon = self.epsilon* self.epsilon_decay
        def load(self,name):
                self.model.load_weights(name)
        def save(self,name):
                self.model.save_weights(name)


def game_driver():
        # episode=10

        # ENV From Above
        #env=gym.make('CartPole-v0')
        action_size=env.action_space.n
        state_size=env.observation_space.shape[0]
        # print("observ_space = ",env.observation_space)

        agent=DQNAgent(state_size,action_size)
        agent.load("./save/cartpole-dqn.h5")

        done=False
        batch_size=32

        for e in range(0,EPISODES): ## Episodes
                state=env.reset()
                state=np.reshape(state,[1,state_size])#  [1,4])#[1,state_size]#from [4,1]to[1,4]

                for time_t in range(0,500): # Timestamps within each Episodes
                        env.render() #Graphical Display
                        action=agent.act(state)
                        next_state,reward, done, _ = env.step(action)
                        #print("Reward_current-a: ",reward)#next_state,reward, done, _ = env.step(action)
                        reward=reward if not done else -10
                        #print("Reward_current-b: ",reward)#next_state,reward, done, _ = env.step(action)
                        next_state = np.reshape(next_state, [1,state_size])#  [1,4])#[1,state_size]
                        #remember prev
                        agent.remember(state,action,reward,next_state,done)
                        state=next_state
                        #done
                        if done:
                                #print
                                # print("Episode: {}/{}, score: {}".format(e,EPISODES,time_t))
                                print("episode: {}/{}, score: {}, e: {}"
                                                .format(e, EPISODES, time_t, agent.epsilon))#e: {:.2}

                                break
                        if len(agent.memory)>batch_size:
                                agent.replay(batch_size)
                        # if e % 10 == 0:
                        #       agent.save("./save/cartpole-dqn.h5")
# Game Driver

if __name__ =='__main__':
        # tester2()
        game_driver()

