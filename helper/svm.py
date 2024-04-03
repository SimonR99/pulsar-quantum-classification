import time
from .dataset import sub_select_dataset, torch_convertion, generate_batches
from tqdm import tqdm

def run_svm_sequence(model_parameters, model, x_train, y_train, x_test, y_test, metrics):
    for i in tqdm(range(model_parameters.num_runs)):
        balanced_x_train, balance_y_train = sub_select_dataset(x_train, y_train, model_parameters.training_samples, balanced=True)
        balanced_x_test, balance_y_test = sub_select_dataset(x_test, y_test, model_parameters.testing_samples)

        # Train the model and make predictions
        start_training_time = time.time()
        model.fit(balanced_x_train, balance_y_train)
        end_training_time = time.time()

        training_duration = end_training_time - start_training_time

        start_testing_time = time.time()
        y_pred = model.predict(balanced_x_test)
        end_testing_time = time.time()

        test_duration = end_testing_time - start_testing_time

        # Store the scores
        metrics.append_score(balance_y_test, y_pred, training_duration, test_duration)