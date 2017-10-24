#############################################################
# This is an example of using ADMM method to solve a consensus problem
# This application uses cvxpy solver for convex optimization
# Peer-to-peer communications are implemented using the Process module
# Author: Junyao Guo
#############################################################

from cvxpy import *
import numpy as np 
import scipy.sparse as sps
from multiprocessing import Process, Pipe

def buildCommpipes(pipeID):
	pipeStart = []
	pipeEnd = []
	for i in range(len(pipeID)):
		start, end = Pipe()
		pipeStart += [start]
		pipeEnd += [end]
	return pipeStart, pipeEnd

def runWorker(xlocal, pipeID, pipeStart, pipeEnd, workerID):
	nvar = 4
	rho = 1
	epsilon = 0.001
	# formulate subproblem
	xbar = Parameter(nvar, value=np.zeros(nvar))
	u = Parameter(nvar, value=np.zeros(nvar))
	x = Variable(nvar,1)
	f = square(norm(x - xlocal)) + (rho/2)*sum_squares(x - xbar + u)
	prox = Problem(Minimize(f))
	# get the local connected pipes
	pipeLocal = [] #The comm end to use for each destLocal
	destLocal = []
	for i in range(len(pipeID)):
		if pipeID[i,1] == workerID:
			pipeLocal += [pipeStart[i]]
			destLocal += pipeID[i,2]
		if pipeID[i,2] == workerID:
			pipeLocal += [pipeEnd[i]]
			destLocal += pipeID[i,1]
	nPipe = len(pipeLocal)
	N = nPipe + 1  #number of neighbors
	k = 0 #iteration counter
	convg = False
	prox.solve()
	#create local table to record the received message
	msgrcv = np.tile(np.array(x.value.transpose()),(nPipe,1)).transpose() 
	# local ADMM loop.
	while not convg:
		k += 1
		# print(k)
		prox.solve()
		# print(x.value)
		# print(msgrcv)
		for i in range(nPipe):
			msg = x.value #the message sent to different dest could be different
			pipeLocal[i].send(msg)
			# print("Message sent from worker %d at %d -th iteration is" % (workerID,k))
			# print(msg)
			if pipeLocal[i].poll(0.0001): #if received some message
				msgrcv[:,i] = pipeLocal[i].recv().transpose()
				# print("Message received in worker %d's pipe at %d -th iteration is" % (workerID,k))
				# print(msgrcv[:,i])
		# print((np.sum(msgrcv,axis = 1) + x.value.transpose())/N)
		# print("all message received by worker %d at %d -th iteration is" %(workerID,k))
		# print(msgrcv)
		xbar.value = (np.sum(msgrcv,axis = 1) + x.value.transpose()).transpose()/N
		u.value += x.value - xbar.value
		# print(np.linalg.norm(xbar.value - x.value))
		if np.linalg.norm(xbar.value - x.value) < epsilon and k > 5:
		# if k == 10000:
			convg = True
	if convg:
		print("Worker %d converged at %d -th iteration! The result is" % (workerID,k) )
		print(x.value)

if __name__ == '__main__':
	nvar = 4
	N = 3 #number of workers
	# create adjacency matrix A for small N
	A = sps.csr_matrix([[0,1,1],[1,0,1],[1,1,0]]) 
	# transform the adjacency matrix to upper triagular to avoid duplicate pipes 
	Atr = sps.triu(A)
	# create adjacency matrix A for large N
	# row = np.array([0,0,8,9,7,7,5,6,4,4,1,3,2,4,7,1,5,3,6])
	# col = np.array([8,9,7,7,5,6,4,4,1,3,2,2,4,7,0,5,8,6,9])
	# data = np.ones(row.size)
	# Atr = sps.csr_matrix((data, (row, col)), shape=(N, N))

	r,c,v = sps.find(Atr)
	# store pipeID, startNode, endNode 
	pipeID = np.array([np.arange(r.size),r,c]).transpose()
	pipeStart, pipeEnd = buildCommpipes(pipeID)

	# generate initial x's
	np.random.seed(0)
	a = np.random.randn(nvar, N)
	print("initial values of x's are \n", a)
	# start the processes
	procs = []
	for i in range(N):
		procs += [Process(target = runWorker, args = (a[:,i],pipeID, pipeStart,pipeEnd,i))]
		procs[-1].start()
