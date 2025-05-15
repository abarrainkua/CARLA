## We study the relation between algorithmic fairness 
## and fairness in algorithmic recourse
## the relations between the fairness metrics from both areas

import numpy as np
import pandas as pd
from carla.models.catalog import  MLModelCatalog
from carla.data.catalog import OnlineCatalog, CsvCatalog, DataCatalog
from carla.recourse_methods import GrowingSpheres, ActionableRecourse
from carla import RecourseMethod
from carla.models.negative_instances import predict_negative_instances
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import sys
from sklearn.utils import resample
import torch.nn as nn
import torch


def train_social_burden(dataset='adult', sens_attr='race', lr=0.001, epochs=10, batch_size=256, 
                        hidden_sizes=[128, 128], activation_name="relu", verbose=True, pretrain_epochs=2, 
                        recourse_method="gs", recourse_hyperparam={}, random_state=42):
    
    dataset_path =  dataset
    df_orig = pd.read_csv(f"{dataset_path}.csv")

    if dataset == "adult":
        
        continuous = ["age", "fnlwgt", "education-num", "capital-gain", "hours-per-week", "capital-loss"]
        categorical = ["marital-status", "native-country", "occupation", "race", "relationship", "sex", "workclass"]
        immutable = ["age", "sex", "race"]
        y_var = "income"
        
        if sens_attr == "race":
            s_var = "race_White"
        elif sens_attr == "sex":
            s_var = "sex_Male"
        elif sens_attr == "age":
            s_var = "age_bin"
    
    # Get part for training and store the other part for dyamic simulation
    df_train, df_test = train_test_split(df_orig, test_size=0.3, random_state=random_state)
    if sens_attr == "age":
        s_train = (df_train["age"] > 30).astype(int).to_numpy()
        s_test = (df_test["age"] > 30).astype(int).to_numpy()         
        

    # Save train into dataframe 
    df_train.to_csv(f"{dataset_path}_train.csv", index=False)

    dataset_train = CsvCatalog(file_path=f"{dataset_path}_train.csv",
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target='income')

    # Map from string to PyTorch activation class
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU
    }
    
    if activation_name.lower() not in activation_map:
        raise ValueError(f"Unsupported activation: {activation_name}. Supported: {list(activation_map.keys())}")
    
    activation = activation_map[activation_name.lower()]
    
    # Map from string to recourse method
    recourse_map = {
        "gs": GrowingSpheres,
        "ar": ActionableRecourse
    }
    
    if recourse_method.lower() not in recourse_map:
        raise ValueError(f"Unsupported recourse method: {recourse_method}. Supported: {list(recourse_map.keys())}")
    
    recourse_model_obj = recourse_map[recourse_method.lower()]
    
    ml_model = MLModelCatalog(
        dataset_train, 
        model_type="ann", 
        load_online=False, 
        backend="pytorch"
    )
    
    training_params = {"lr": lr, "epochs": pretrain_epochs, "batch_size": batch_size, 
                       "hidden_size": hidden_sizes}
    
    # Pre-train model
    ml_model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"]
    )
    
    
    # Now start the iterative process to optimize wrt to social burden
    
    # The loss function of the model
    loss_learner = nn.CrossEntropyLoss(reduction="none")
    
    # Get the underlying model
    learner = ml_model._model 
    
    # The optimizer of the learner
    optimizer_learner = torch.optim.SGD(learner.parameters(), lr=lr)
    
    for epoch in range(pretrain_epochs, epochs, 1):
        
        # Define the batches of the epoch
        dataset_len = len(dataset_train.df)
        batch_permu = np.random.permutation(dataset_len)
        batch_indices = np.array_split(batch_permu, np.ceil(dataset_len / batch_size))
        
        for current_batch in batch_indices:
            
            # Create dataframe with current batch
            train_batch_df = dataset_train.df.iloc[current_batch]
            
            X_batch = train_batch_df.drop(columns=[y_var]).to_numpy()
            y_batch = train_batch_df[y_var].to_numpy()
            s_batch = train_batch_df[s_var].to_numpy()
            
            X_batch = torch.from_numpy(X_batch).to(torch.float32)
            y_batch = torch.from_numpy(preprocessing.LabelEncoder().fit_transform(y_batch))
            s_batch = torch.from_numpy(preprocessing.LabelEncoder().fit_transform(s_batch))
                    
            # Initialize the recourse method
            recourse_m = recourse_model_obj(ml_model, recourse_hyperparam)
            
            #Get predictions for test instances for current model
            y_pred_scores = ml_model.predict(train_batch_df)

            # Binarize predictions
            y_pred_bin = (y_pred_scores > .5).astype(int).reshape(1,-1)[0] 
            
            # Get instances that will be subject to recourse
            factuals = train_batch_df[y_pred_bin == 0]
            
            # Get counterfactuals
            counterfactuals = recourse_m.get_counterfactuals(factuals)
            
            # Create dataframe with new representations, after recourse    
            train_batch_new = train_batch_df.copy()
            factual_indices = factuals.index  # index of factuals
            cf_columns = counterfactuals.columns
            train_batch_new.loc[factual_indices, cf_columns] = counterfactuals.values  # replace by counterfactuals
            
            # Convert dataframes into numpy array
            old_test_array = train_batch_df.to_numpy()
            new_test_array = train_batch_new.to_numpy()
            
            # Compute recourse costs for each instance in the batch
            recourse_costs = np.linalg.norm(new_test_array - old_test_array, axis=1)
            
            # Create a DataFrame with recourse information
            recourse_df = pd.DataFrame({
                "group": train_batch_df[s_var].to_numpy(),
                "y_true": train_batch_df[y_var].to_numpy(),
                "cost": recourse_costs,
                "burden": np.where(train_batch_df[y_var].to_numpy() == 0, 0, recourse_costs)
            })

            
            # Update instance weight based on burden
            social_burden_tensor = torch.tensor(recourse_df["burden"].to_numpy(), dtype=torch.float32)
            total_burden = social_burden_tensor.sum()
            instance_weights = 1 + len(current_batch) * (social_burden_tensor / total_burden)
            
            # Get the underlying model
            learner = ml_model._model 
            
            # Get learner loss value
            loss_value_learner = loss_learner(learner(X_batch), y_batch)
            weighted_loss_learner = loss_value_learner * instance_weights
            weighted_loss_learner = torch.mean(weighted_loss_learner)

            # Gradient step    
            optimizer_learner.zero_grad()
            weighted_loss_learner.backward()
            optimizer_learner.step()

            # Update ML model
            ml_model._model = learner
            
            print("ML model updated")
        
        if verbose:
            
            print(f"epoch={epoch} loss={weighted_loss_learner}")
    
    
    return learner
    
    





# dataset = sys.argv[1]
dataset = "adult"
# sens_attr = sys.argv[2]
sens_attr = "race" # "age", "race", "sex"


my_trained_model = train_social_burden(dataset='adult', sens_attr='race', lr=0.001, epochs=10, batch_size=256, 
                                       hidden_sizes=[128, 128], activation_name="relu", verbose=True, pretrain_epochs=2, 
                                       recourse_method="gs", recourse_hyperparam={}, random_state=42)