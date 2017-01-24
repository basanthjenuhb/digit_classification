from sklearn import datasets
import matplotlib.pyplot as plt , numpy as np

np.random.seed(1)

def sigmoid(z, derivative=False):
	if derivative == True:
		return sigmoid(z) * sigmoid(1 - z)
	return 1.0 / (1.0 + np.exp(-z))

def costFunction(x, y, m, W1, W2, b1, b2):
	z1 = np.dot(x, W1.T) + b1
	x1 = sigmoid(z1)
	z2 = np.dot(x1, W2.T) + b2
	x2 = sigmoid(z2)
	error = x2 - y
	cost = 0.5 / m * sum(error ** 2)
	return z1, x1, z2, x2, error, cost

#Loading the datasets
digits = datasets.load_digits()

#Preprocessing
x, Y = (digits.data > 0).astype(np.int32,copy=False), np.array(digits.target)
y = np.zeros((len(x),10))
y[np.arange(len(Y)),Y] = 1

#Network variables
hidden, m = 128, x.shape[0]
W1, W2 = 2 * np.random.randn(hidden,x.shape[1]) - 1 , 2 * np.random.randn(y.shape[1],hidden) - 1
b1 , b2 = 2 * np.random.randn(1,hidden) - 1 , 2 * np.random.randn(1,10) - 1
epochs , alpha , costs = 10000 , 0.02 , []

#training
for i in range(epochs):
	z1 , x1 , z2 , x2 , error , cost = costFunction(x, y, m, W1, W2, b1, b2)
	print "Epoch",i,"Error:",sum(cost)
	delta2 = error * sigmoid(z2, derivative=True)
	dw2 = np.dot(delta2.T, x1)
	costs.append(sum(cost))

	delta1 = np.dot(delta2,W2) * sigmoid(z1, derivative=True)
	dw1 = np.dot(delta1.T,x)

	W1, W2 = W1 - alpha * dw1, W2 - alpha * dw2
	b1 , b2 = b1 - alpha * sum(delta1), b2 - alpha * sum(delta2)

#Evaluation
res = np.argmax(costFunction(x, y, m, W1, W2, b1, b2)[3],axis=1)
count = 0.0
for i in range(len(Y)):
	if res[i] == Y[i]:count += 1
print count ,"/",x.shape[0],count/x.shape[0] * 100 ,"%"
plt.plot([j for j in range(len(costs))],costs)
plt.xlabel('Iterations -->')
plt.ylabel('Error -->')
plt.show()
