## In continuarea temei am modificat codul astfel incat algoritmul sa afiseze eticheta reala (ce numar este in poza)
#si pe langa asta, predictia modelului 

import ssl
import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt # ce vom folosi pentru afisarea imaginilor


## datasetul cu imagini reprez cifre
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

# partea de one hot encode 
def one_hot_encode(labels, num_classes=10): # avem 10 cifre 0..9
    return np.eye(num_classes)[labels] # np.eye()[] folosesc valorile din labels
    # pentru a indexa matricea I 
    ## reprezentam cifrele sub forma de vectori de lungime 10

train_Y_one_hot = one_hot_encode(train_Y)
test_Y_one_hot = one_hot_encode(test_Y)

input_size = 784 # fiecare imagine are 784 de inputuri 
output_size = 10 # oricare din cele 10 cifre 


np.random.seed(42)
W = np.random.randn(input_size, output_size) * 0.01 # greutatile random
b = np.zeros((output_size,)) # biasurile cu 0 

## functia de activare, transf vectorii de valori reale in probabilitati
def softmax(z): ## z matricea de intrare
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # scadem max pentru stabilitate numerica
    return exp_z / exp_z.sum(axis=1, keepdims=True) # exp asigura ca valorile mari cresc pe masura si cele mici scad

# partea de propagare 
def forward(X):
    z = np.dot(X, W) + b 
    return softmax(z)

## calc pierderea medie din batch, se masoara cat de bine/prost a prezis modelul
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m  

# calc cum trebuie sa ajustam parametrii (weights n biases) in functie de erorile obtinute 
def backward(X, y_true, y_pred):
    m = y_true.shape[0]
    grad_z = y_pred - y_true  
    grad_W = np.dot(X.T, grad_z) / m  
    grad_b = np.sum(grad_z, axis=0) / m  # derivata pierderii fata de bias
    return grad_W, grad_b

# partea de antrenare cu gradient descent
def train(X, y, epochs=100, batch_size=100, learning_rate=0.01):
    global W, b
    m = X.shape[0] # numarul de exemple din setul de date de antrenare X
    for epoch in range(epochs): # o epoca = o trecere completa prin tot setul de date de antrenament
        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            y_pred = forward(X_batch) # predictiile pentru lotul curent
            
            loss = compute_loss(y_batch, y_pred)
            
            grad_W, grad_b = backward(X_batch, y_batch, y_pred) # calc gradientelor
            
            W -= learning_rate * grad_W
            b -= learning_rate * grad_b

        if epoch % 10 == 0: ## afisarea pierderii la fiecare 10 epoch
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

def predict(X):
    y_pred = forward(X)
    return np.argmax(y_pred, axis=1)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

## pentru afisarea imaginii impreuna cu predictie
def show_image_and_predict(index):
    image = test_X[index].reshape(28, 28)  
    true_label = test_Y[index]  
    prediction = predict(test_X[index:index+1])[0]  
    
    # matplotlivb
    plt.imshow(image, cmap='gray')
    plt.title(f'Eticheta reala este: {true_label}, Predictia modelului: {prediction}')
    plt.axis('off')
    plt.show()


initial_predictions = predict(test_X)
initial_accuracy = accuracy(test_Y, initial_predictions)
print(f'Initial accuracy: {initial_accuracy * 100:.2f}%')


train(train_X, train_Y_one_hot, epochs=100, batch_size=100, learning_rate=0.01)

final_predictions = predict(test_X)
final_accuracy = accuracy(test_Y, final_predictions)
print(f'Final accuracy: {final_accuracy * 100:.2f}%')

# imaginea si predictia, trebuie inlocuit indexul manual 
show_image_and_predict(4)  
