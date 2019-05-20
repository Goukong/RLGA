import numpy as np 
import cv2
import targetTacing as TT
import RL_brain
class RLGA:
	"""docstring for RLGA"""
	def __init__(self,DNA_SIZE = 6,POP_SIZE = 100):
		self.dna_size = DNA_SIZE
		self.pop_size = POP_SIZE
		self.min_mse = 1e10
		self.min_param = []
		self.base = 0
		self.max_found = -1
		self.last_found = 0
		self.max_reward = -9999

	def changeTrainSet(self):
		self.base += 10
		self.min_mse = 1e10
		self.max_found = -1
		self.min_param = []
		self.max_reward = -9999

	def createQTable(self):
		#First we devide the dna into 6 parts
		#and create 6 q-table with 100 action choice 
		t1 = RL_brain.QLearningTable(self.pop_size)
		t2 = RL_brain.QLearningTable(self.pop_size)
		t3 = RL_brain.QLearningTable(self.pop_size)
		t4 = RL_brain.QLearningTable(self.pop_size)
		t5 = RL_brain.QLearningTable(self.pop_size)
		t6 = RL_brain.QLearningTable(self.pop_size)
		agents =[t1,t2,t3,t4,t5,t6]

		return agents

	def createPopulation(self):
		#create a module for DNA
		dna_mod = np.empty((self.pop_size,self.dna_size))
		#each DNA has its own bound
		for i in range(self.pop_size):
			#qualityLevel [0.01,0.1]
			dna_mod[i][0] = np.random.rand()/10
		
			#mindistance  [0,100] 
			dna_mod[i][1] = np.random.randint(0,100,1)
		
			#winSize  (2,100]
			dna_mod[i][2] = np.random.randint(3,100,1)
		
			#maxlevel,COUNT [0,100]
			dna_mod[i][3] = np.random.randint(0,100,1)
			dna_mod[i][4] = np.random.randint(0,100,1)
		
			#EPS [0,1]
			dna_mod[i][5] = np.random.rand()
		self.pop = dna_mod
		return dna_mod

	def getfitness(self,child):
		#translate the DNA into params
		feature_params = dict(
			maxCorners = 1000,
			qualityLevel = child[0],
			minDistance = child[1],
			blockSize = 3
			)
		lk_params = dict(
			winSize = (int(child[2]),int(child[2])),
			maxLevel = int(child[3]),
			criteria = (
				cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
				int(child[4]),child[5])
			)
		#!!!!get MSE
		mse,found = TT.targetTrace(feature_params,lk_params,self.base) 
		return mse,found

	def readPastExperience(self,agents):
		self.pop = np.fromfile('pop.bin',dtype = np.float64)
		self.pop = self.pop.reshape(self.pop_size,self.dna_size)
		for i,agent in enumerate(agents):
			agent.readPastInfor(i)
			#if i == 0:
				#print(agent.q_table)
		'''
			读取种群信息
			读取Q表信息
		'''
	def writeCurrentInfor(self,agents):
		self.pop.tofile("pop.bin")
		for i,agent in enumerate(agents):
			agent.writeInfor(i)
			#if i == 0:
				#print(agent.q_table)


	#Simulate and update GA 
	def RL(self,agents):
		#create mods to store dna and choosed pop
		child = []
		state = []
		for i,agent in enumerate(agents):
			idx = agent.choose_action()
			state.append(idx)
			#print(self.pop.shape)
			idx = int(idx)
			parent = self.pop[idx]
			child.append(parent[i])

		#get reward
		'''
			得到适应度，并且拿到最适合的参数
			计算出奖励值
		'''
		mse,found = self.getfitness(child)
		reward = 0

		'''#mse_reward
								if self.min_mse == 1e10:
									self.min_mse = mse
									self.last_mse = mse
									mse_reward = 0
								elif mse == 0:
									mse_reward = 0
								else:
									if mse < self.min_mse:
										self.min_mse = mse 
										#self.min_param = child
									mse_reward = self.last_mse - mse
									self.last_mse = mse
						
								#point_found_reward
								if self.max_found == -1:
									self.max_found = found
									self.last_found = found
									found_reward = 0
								elif mse == 0:
									found_reward = 0
								else:
									if found > self.max_found:
										self.max_found = found
									found_reward = found - self.last_found
									self.last_found = found'''
		if self.min_mse == 1e10:
			self.min_mse = mse 
			self.last_mse = mse 
			mse_reward = 0
		elif mse == 0 or mse == 1e9:
			mse_reward = -10
		else:
			mse_reward = self.last_mse - mse 
			self.last_mse = mse 

		if self.max_found == -1:
			self.max_found = found 
			self.last_found = found
			found_reward = found
		else:
			found_reward = found - self.last_found
			self.last_found = found
									
		reward = mse_reward * 0.8 + found_reward * 1.2

		if reward > self.max_reward:
			self.max_reward = reward
			self.min_param = child
		print(reward)

		#train the q_table
		for i,agent in enumerate(agents):
			agent.learn(state[i],reward)

		return agents
		