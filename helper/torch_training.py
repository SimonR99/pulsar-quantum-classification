import torch
import time
from tqdm import tqdm
import pennylane.numpy as np
from .dataset import sub_select_dataset, torch_convertion, generate_batches


def train_network_batch(model,optimizer,criterion,x_train,y_train,x_test,y_test,num_epochs,train_losses,test_losses, batch_size=10):
    train_loader = generate_batches(x_train, y_train, batch_size=batch_size)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            #clear out the gradients from the last step loss.backward()
            optimizer.zero_grad()
            
            #forward feed
            outputs = model(inputs)

            #calculate the loss
            loss = criterion(outputs, labels.float())

            #backward propagation: calculate gradients
            loss.backward()

            #update the weights
            optimizer.step()

            running_loss += loss.item()
        
        # Record the average training loss
        train_losses[epoch] = running_loss / len(train_loader)

        # Evaluation on the test set
        model.eval()
        with torch.no_grad():
            output_test = model(x_test)
            loss_test = criterion(output_test, y_test.float())
            test_losses[epoch] = loss_test.item()


def run_torch_sequence(model_parameters, model, x_train, y_train, x_test, y_test, metrics, batch_size=1):
    for i in tqdm(range(model_parameters.num_runs)):
        num_epochs = model_parameters.max_num_epochs
        train_losses = np.zeros(num_epochs)
        test_losses  = np.zeros(num_epochs)

        learning_rate = 0.01
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

        balanced_x_train, balanced_y_train = sub_select_dataset(x_train, y_train, model_parameters.training_samples, balanced=True)
        sub_selected_x_test, sub_selected_y_test = sub_select_dataset(x_test, y_test, model_parameters.testing_samples)

        balanced_x_train, sub_selected_x_test, balanced_y_train, sub_selected_y_test = torch_convertion(balanced_x_train, sub_selected_x_test, balanced_y_train, sub_selected_y_test)

        start_training_time = time.time()
        train_network_batch(model,optimizer,criterion, balanced_x_train,balanced_y_train, sub_selected_x_test,sub_selected_y_test,num_epochs,train_losses,test_losses, batch_size=batch_size)
        end_training_time = time.time()

        training_duration = end_training_time - start_training_time

        # Predict the test set
        start_testing_time = time.time()
        output_test = model(sub_selected_x_test)
        end_testing_time = time.time()

        testing_duration = end_testing_time - start_testing_time

        predicted_test = (output_test > 0.5).float()

        # Calculate the scores
        metrics.append_score(sub_selected_y_test, predicted_test, training_duration, testing_duration)