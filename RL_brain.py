import numpy as np 
import pandas as pd 

class QLearningTable:
	def __init__(self,POP_SIZE,learning_rate=0.01,reward_decay=0.9,e_greedy=0.6):
		#initailize 
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		#make an array according to the pop size
		actions = np.array(range(POP_SIZE))
		self.actions = actions

		#create a q_table like
		'''
			q_table:

			  aciton 0 1 2 3 ... 99
			state
			'start'  0 0 0 0 ... 0
		'''
		self.q_table = pd.DataFrame(columns=actions,dtype=np.float64)
		self.q_table = self.q_table.append(
			pd.Series(
				[0.0]*POP_SIZE,
				index = self.q_table.columns,
				name = 0,
				))

	#choose next action according to greedy skills
	def choose_action(self):
		if np.random.uniform() < self.epsilon:
			#choose the biggest of 'start' row
			#random choose in case that the result is not only
			current_choice = self.q_table.loc[0,:]
			pool = (current_choice == np.max(current_choice))
			action = np.random.choice(current_choice[pool].index)
		else:
			action = np.random.choice(self.actions)
		#return a index of pop_size
		return action
	
	def learn(self,action,reward):
		try:
			q_predict = self.q_table.loc[0,action]
		except:
			action = str(action)
			q_predict = self.q_table.loc[0,action]
		q_target = reward
		self.q_table.loc[0,action] += self.lr * (q_target - q_predict)

	#让之前的经验变得有用,如果要保存的话种群也要保存起来
	def readPastInfor(self,num):
		name = 'agent' + str(num) + '.csv' 
		test = pd.read_csv(name)
		self.q_table = test

	def writeInfor(self,num):
		name = 'agent' + str(num) + '.csv'
		self.q_table.to_csv(name,sep=",",header=True,index=False)
