import numpy as np 
import cv2
import targetTacing as TT
import RL_brain
class RLGA:
	"""docstring for RLGA"""
	def __init__(self,DNA_SIZE = 6,POP_SIZE = 100):
		self.dna_size = DNA_SIZE
		self.pop_size = POP_SIZE
		self.max_found = 0
		self.max_param = []
		self.last_reward = 0

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
		result = TT.targetTrace(feature_params,lk_params) 
		return result

	def readPastExperience():
		'''
			读取种群信息
			读取Q表信息
		'''
		return

	#Simulate and update GA 
	def RL(self,pop,agents):
		#create mods to store dna and choosed pop
		child = []
		state = []
		for i,agent in enumerate(agents):
			idx = agent.choose_action()
			state.append(idx)
			parent = pop[idx]
			child.append(parent[i])

		#get reward
		'''
			得到适应度，并且拿到最适合的参数
			计算出奖励值
		'''
		fitness = self.getfitness(child)
		if fitness > self.max_found:
			self.max_found = fitness
			self.max_param = child
		reward = fitness - self.last_reward

		#train the q_table
		for i,agent in enumerate(agents):
			agent.learn(state[i],reward)

		return agents
		