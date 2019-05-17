import RLGA
def main():
	rlga = RLGA.RLGA()
	agents = rlga.createQTable()
	pop = rlga.createPopulation()
	firstTime = False
	if not firstTime:
		rlga.readPastExperience(agents)
		#print(rlga.pop)
	for i in range(10):
		for _ in range(100):
			agents = rlga.RL(agents)
		print('min_mse:',rlga.min_mse)
		print('min_param:',rlga.min_param)
		rlga.changeTrainSet()
	rlga.writeCurrentInfor(agents)	


if __name__ == '__main__':
	main()