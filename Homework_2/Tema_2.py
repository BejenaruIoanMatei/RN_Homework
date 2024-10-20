import ssl
import numpy as np
from torchvision.datasets import MNIST

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    return np.array(mnist_data), np.array(mnist_labels)

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

# pentru a normaliza datele
train_X = train_X / 255.0
test_X = test_X / 255.0

# One-hot encoding for labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels] # np.eye()[] folosesc valorile din labels
    # pentru a indexa matricea I 

train_Y_one_hot = one_hot_encode(train_Y)
test_Y_one_hot = one_hot_encode(test_Y)

input_size = 784  
output_size = 10  


np.random.seed(42)
W = np.random.randn(input_size, output_size) * 0.01  
b = np.zeros((output_size,)) 


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def forward(X):
    z = np.dot(X, W) + b 
    return softmax(z)


def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m  

def backward(X, y_true, y_pred):
    m = y_true.shape[0]
    grad_z = y_pred - y_true  
    grad_W = np.dot(X.T, grad_z) / m  
    grad_b = np.sum(grad_z, axis=0) / m  
    return grad_W, grad_b


def train(X, y, epochs=500, batch_size=100, learning_rate=0.01):
    global W, b
    m = X.shape[0]
    for epoch in range(epochs):
        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            y_pred = forward(X_batch)
            
            loss = compute_loss(y_batch, y_pred)
            
            grad_W, grad_b = backward(X_batch, y_batch, y_pred)
            
            W -= learning_rate * grad_W
            b -= learning_rate * grad_b

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

def predict(X):
    y_pred = forward(X)
    return np.argmax(y_pred, axis=1)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

initial_predictions = predict(test_X)
initial_accuracy = accuracy(test_Y, initial_predictions)
print(f'Initial accuracy: {initial_accuracy * 100:.2f}%')

train(train_X, train_Y_one_hot, epochs=100, batch_size=100, learning_rate=0.01)

final_predictions = predict(test_X)
final_accuracy = accuracy(test_Y, final_predictions)
print(f'Final accuracy: {final_accuracy * 100:.2f}%')
