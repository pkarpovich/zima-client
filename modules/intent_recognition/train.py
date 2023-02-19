import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import IntentRecognitionDataset
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet


def prepare_data(intents):
    all_words = []
    tags = []
    xy = []
    # loop through each sentence in our intents patterns
    for intent in intents:
        print(intent)
        tag = intent['intent']
        # add to tag list
        tags.append(tag)
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = tokenize(pattern)
            # add to our words list
            all_words.extend(w)
            # add to xy pair
            xy.append((w, tag))

    # stem and lower each word
    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    # remove duplicates and sort
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # create training data
    x_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        # X: bag of words for each pattern_sentence
        bag = bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)
        # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train, all_words, tags


def train_model(train_loader, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(words)
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'final loss: {loss.item():.6f}')

    return model


with open('../../data/intents.json', 'r') as f:
    intentsData = json.load(f)

x_train, y_train, all_words, tags = prepare_data(intentsData['intents'])

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

dataset = IntentRecognitionDataset(x_train, y_train)
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

model = train_model(train_loader, num_epochs, learning_rate)

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
