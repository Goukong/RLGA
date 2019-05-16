import RLGA
def main():
	rlga = RLGA.RLGA()
	agents = rlga.createQTable()
	pop = rlga.createPopulation()
	for _ in range(100):
		agents = rlga.RL(pop,agents)
		print('max_found:',rlga.max_found)
		print('max_param:',rlga.max_param)



if __name__ == '__main__':
	main()