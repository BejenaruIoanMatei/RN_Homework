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
train_X, test_X = train_X / 255.0, test_X / 255.0

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

train_Y_one_hot, test_Y_one_hot = one_hot_encode(train_Y), one_hot_encode(test_Y)

input_size, hidden_size, output_size = 784, 100, 10

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((hidden_size,))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((output_size,))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

def backward(X, y_true, z1, a1, a2):
    m = y_true.shape[0]
    dz2 = a2 - y_true
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0) / m
    
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0) / m
    
    return dW1, db1, dW2, db2

epochs, batch_size, learning_rate = 100, 50, 0.01

def train(X, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate):
    global W1, b1, W2, b2
    m = X.shape[0]
    for epoch in range(epochs):
        for i in range(0, m, batch_size):
            X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
            z1, a1, z2, a2 = forward(X_batch)
            loss = compute_loss(y_batch, a2)
            
            dW1, db1, dW2, db2 = backward(X_batch, y_batch, z1, a1, a2)
            
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        if epoch % 10 == 0:
            train_loss, train_accuracy = evaluate(X, y)
            test_loss, test_accuracy = evaluate(test_X, test_Y_one_hot)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

def evaluate(X, y):
    _, _, _, y_pred = forward(X)
    loss = compute_loss(y, y_pred)
    accuracy_val = accuracy(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
    return loss, accuracy_val

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

train(train_X, train_Y_one_hot)

final_train_loss, final_train_accuracy = evaluate(train_X, train_Y_one_hot)
final_test_loss, final_test_accuracy = evaluate(test_X, test_Y_one_hot)
print(f'Final Train Accuracy: {final_train_accuracy * 100:.2f}%')
print(f'Final Test Accuracy: {final_test_accuracy * 100:.2f}%')
