import torch
import time
from tqdm import tqdm
import pennylane.numpy as np
from .dataset import sub_select_dataset, torch_convertion, generate_batches
import matplotlib.pyplot as plt

def train_network_batch(model,optimizer,criterion,x_train,y_train,x_test,y_test,num_epochs,train_losses,test_losses, batch_size, early_stopping_threshold, patience):
    train_loader = generate_batches(x_train, y_train, batch_size=batch_size)
    patience_counter = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = criterion(outputs, labels.float())

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_losses[epoch] = running_loss / len(train_loader) 

        # Evaluation on the test set
        model.eval()
        with torch.no_grad():
            output_test = model(x_test)
            loss_test = criterion(output_test, y_test.float())
            test_losses[epoch] = loss_test.item()

        if early_stopping_threshold is not None:

            if loss_test < best_loss - early_stopping_threshold:
                best_loss = loss_test
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch+1}. Validation loss has not improved for {patience} epochs.")
                break

def run_torch_sequence(model_parameters, model_creator, x_train, y_train, x_test, y_test, metrics, batch_size=1, plot = True, early_stopping_threshold=None, patience=5):
    total_test_losses = []
    total_train_losses = []

    for i in tqdm(range(model_parameters.num_runs)):
        num_epochs = model_parameters.max_num_epochs
        train_losses = np.full(num_epochs, np.nan)
        test_losses  = np.full(num_epochs, np.nan)

        learning_rate = 0.01
        model = model_creator()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

        balanced_x_train, balanced_y_train = sub_select_dataset(x_train, y_train, model_parameters.training_samples, balanced=True)
        sub_selected_x_test, sub_selected_y_test = sub_select_dataset(x_test, y_test, model_parameters.testing_samples)

        balanced_x_train, sub_selected_x_test, balanced_y_train, sub_selected_y_test = torch_convertion(balanced_x_train, sub_selected_x_test, balanced_y_train, sub_selected_y_test)

        start_training_time = time.time()
        train_network_batch(model,optimizer,criterion, balanced_x_train,balanced_y_train, sub_selected_x_test,sub_selected_y_test,num_epochs,train_losses,test_losses, batch_size=batch_size, early_stopping_threshold=early_stopping_threshold, patience=patience)
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

        total_test_losses.append(test_losses)
        total_train_losses.append(train_losses)
        
    if plot:
        # plot all the training losses in red and all the test losses in blue
        avg_train_losses = np.mean(total_train_losses, axis=0)
        std_train_losses = np.std(total_train_losses, axis=0)
        avg_test_losses = np.mean(total_test_losses, axis=0)
        std_test_losses = np.std(total_test_losses, axis=0)
        
        # Plotting
        plt.figure(figsize=(10, 5))
        epochs = np.arange(1, len(avg_train_losses) + 1)
        
        # Plot mean training and testing losses
        plt.plot(epochs, avg_train_losses, 'r-', label='Average Training Loss')
        plt.plot(epochs, avg_test_losses, 'b-', label='Average Testing Loss')
        
        # Fill between for standard deviation
        plt.fill_between(epochs, avg_train_losses - std_train_losses, avg_train_losses + std_train_losses, color='red', alpha=0.3, label='Training Loss Std Dev')
        plt.fill_between(epochs, avg_test_losses - std_test_losses, avg_test_losses + std_test_losses, color='blue', alpha=0.3, label='Testing Loss Std Dev')
        
        # Additional plot formatting
        plt.title('Training and Testing Losses Over Epochs with Standard Deviation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()