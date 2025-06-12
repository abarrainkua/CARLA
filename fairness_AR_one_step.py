## We study the relation between algorithmic fairness 
## and fairness in algorithmic recourse
## the relations between the fairness metrics from both areas

import numpy as np
import pandas as pd
from carla.models.catalog import  MLModelCatalog
from carla.data.catalog import OnlineCatalog, CsvCatalog, DataCatalog
from carla.recourse_methods import GrowingSpheres, ActionableRecourse, CCHVAE
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


def train_social_burden(dataset='adult', file_path=None, sens_attr='race', s_vals = None, lr=0.001, epochs=10, batch_size=256, 
                        hidden_sizes=[128, 128], activation_name="relu", verbose=True, pretrain_epochs=2, training_metrics=False,
                        recourse_method="GS", weighing_strategy="individual", recourse_hyperparam={}, 
                        results_file=None, random_state=42):
    
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

    dataset_train = CsvCatalog(file_path=file_path,
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
        "GS": GrowingSpheres,
        "AR": ActionableRecourse,
        "CCHVAE": CCHVAE
    }
    
    if recourse_method.upper() not in recourse_map:
        raise ValueError(f"Unsupported recourse method: {recourse_method}. Supported: {list(recourse_map.keys())}")
    
    recourse_model_obj = recourse_map[recourse_method.upper()]
    
    ml_model = MLModelCatalog(
        dataset_train, 
        model_type="ann", 
        load_online=False, 
        backend="pytorch"
    )
    
    training_params = {"lr": lr, "epochs": 1, "batch_size": batch_size, 
                       "hidden_size": hidden_sizes}
    
    ml_model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"]
    )
    
    if recourse_method == "CCHVAE":
        recourse_hyperparam = {
        "data_name": f"{dataset}_{sens_attr}",
        "n_search_samples": 100,
        "p_norm": 1,
        "step": 0.1,
        "max_iter": 1000,
        "clamp": True,
        "binary_cat_features": True,
        "vae_params": {
            # "layers": [len(ml_model.feature_input_order), 256, 128, 8],
            "layers": [(len(continuous) + len(categorical) - len(immutable)), 256, 128, 8],
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 128,
        },
    }
    
    
    # Now start the iterative process to optimize wrt to social burden
    
    # The loss function of the model
    loss_learner = nn.CrossEntropyLoss(reduction="none")
    
    # Get the underlying model
    learner = ml_model._model
    
    # The optimizer of the learner
    optimizer_learner = torch.optim.SGD(learner.parameters(), lr=lr)
    
    # To store the performance metrics
    metrics_log = []

    
    for epoch in range(epochs):
        
        # Define the batches of the epoch
        dataset_len = len(dataset_train.df)
        batch_permu = np.random.permutation(dataset_len)
        batch_indices = np.array_split(batch_permu, np.ceil(dataset_len / batch_size))
        
        for current_batch in batch_indices:
            
            instance_weights = torch.ones(len(current_batch), requires_grad=True)
            
            # Create dataframe with current batch
            train_batch_df = dataset_train.df.iloc[current_batch]
            
            X_batch = train_batch_df.drop(columns=[y_var]).to_numpy()
            y_batch = train_batch_df[y_var].to_numpy()
            if sens_attr == "age":
                s_batch = s_vals[current_batch]
            else:
                s_batch = train_batch_df[s_var].astype(int).to_numpy()
            
            X_batch = torch.from_numpy(X_batch).to(torch.float32)
            y_batch = torch.from_numpy(preprocessing.LabelEncoder().fit_transform(y_batch))
            s_batch = torch.from_numpy(preprocessing.LabelEncoder().fit_transform(s_batch))
            
            if epoch > pretrain_epochs:
                    
                # Initialize the recourse method
                recourse_m = recourse_model_obj(ml_model, recourse_hyperparam)
                
                #Get predictions for test instances for current model
                y_pred_scores = ml_model.predict(train_batch_df)

                # Binarize predictions
                y_pred_bin = (y_pred_scores > .5).astype(int).reshape(1,-1)[0] 
                
                # Get instances that will be subject to recourse
                factuals = train_batch_df[y_pred_bin == 0]
                
                # Get counterfactuals
                print("getting counterfactuals")
                counterfactuals = recourse_m.get_counterfactuals(factuals)
                print("finished getting counterfactuals")
                
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
                    "group": s_batch.cpu().detach().numpy(),
                    "y_true": train_batch_df[y_var].to_numpy(),
                    "cost": recourse_costs,
                    "burden": np.where(train_batch_df[y_var].to_numpy() == 0, 0, recourse_costs)
                })

                
                # Update instance weight based on burden
                social_burden_tensor = torch.tensor(recourse_df["burden"].to_numpy(), dtype=torch.float32)
                total_burden = social_burden_tensor.sum()
                
                # if weighing_strategy == "individual":
                
                instance_weights = 1 + len(current_batch) * 0.2 * (social_burden_tensor / total_burden)
                
                # elif weighing_strategy == "group":
                #     # Total burden per group
                #     group_burden = recourse_df.groupby("group")["burden"].sum()

                #     # Each group's proportion of total burden
                #     burden_proportions = group_burden / group_burden.sum()

                #     # Map each instance's group to its burden proportion
                #     instance_group = recourse_df["group"]
                #     instance_group_proportion = instance_group.map(burden_proportions)

                #     # Step 4: Compute weights (e.g., scaled by number of instances)
                #     instance_weights = torch.tensor(instance_group_proportion.to_numpy(),
                #         dtype=torch.float32
                #     )

                
                # print(instance_weights)
            
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
        
        # If training_metrics = True, Get performance stats for the model at this point of the training
        
        with torch.no_grad():
            # Full training data
            train_df = dataset_train.df.copy()
            X_train_full = train_df.drop(columns=[y_var]).to_numpy()
            y_train_full = train_df[y_var].to_numpy()
            if sens_attr == "age":
                s_train_full = s_vals
            else:
                s_train_full = train_df[s_var].astype(int).to_numpy()

            # Get model predictions
            X_train_tensor = torch.from_numpy(X_train_full).to(torch.float32)
            # y_pred_scores = ml_model._model(X_train_tensor).detach().numpy().reshape(-1)
            y_pred_scores = ml_model._model(X_train_tensor).detach().numpy()[:,1]
            # print(f"the length of y_pred_scores is {len(y_pred_scores)}")
            y_pred_bin = (y_pred_scores > 0.5).astype(int)

        # Accuracy
        y_encoded = preprocessing.LabelEncoder().fit_transform(y_train_full)
        train_accuracy = np.mean(y_pred_bin == y_encoded)

        # Group-wise accuracy
        # Convert ground truth and sensitive group to numpy arrays
        y_true = y_encoded 
        s_groups = s_train_full

        # Initialize dictionaries for group-wise metrics
        acc_by_group = {}
        tpr_by_group = {}
        fpr_by_group = {}
        ar_by_group = {}
        
        
        for group_val in np.unique(s_groups):
            idx = s_groups == group_val
            y_true_group = y_true[idx]
            y_pred_group = y_pred_bin[idx]

            acc_by_group[f"acc_group_{group_val}"] = np.mean(y_pred_group == y_true_group)

            # TPR: TP / (TP + FN)
            positives = y_true_group == 1
            tpr = np.sum((y_pred_group == 1) & positives) / (np.sum(positives) + 1e-8)
            tpr_by_group[f"tpr_group_{group_val}"] = tpr

            # FPR: FP / (FP + TN)
            negatives = y_true_group == 0
            fpr = np.sum((y_pred_group == 1) & negatives) / (np.sum(negatives) + 1e-8)
            fpr_by_group[f"fpr_group_{group_val}"] = fpr
            
            # AR: (FP + TP) / N_instances
            ar = np.sum((y_pred_group == 1)) / len(y_pred_group)
            ar_by_group[f"ar_group_{group_val}"] = ar

        # Recompute counterfactuals to evaluate burden
        factuals = train_df[y_pred_bin == 0]
        recourse_m = recourse_model_obj(ml_model, recourse_hyperparam)
        print("getting counterfactuals")
        counterfactuals = recourse_m.get_counterfactuals(factuals)
        print("finished getting counterfactuals")

        train_new_df = train_df.copy()
        factual_indices = factuals.index
        train_new_df.loc[factual_indices, counterfactuals.columns] = counterfactuals.values

        old_array = train_df.to_numpy()
        new_array = train_new_df.to_numpy()
        recourse_costs = np.linalg.norm(new_array - old_array, axis=1)

        recourse_info = pd.DataFrame({
            "group": s_train_full,
            "y_true": y_train_full,
            "cost": recourse_costs,
            "burden": np.where(np.array(y_train_full) == 0, 0, recourse_costs)
        })
        
        cost_by_group = recourse_info.groupby("group")["cost"].mean().to_dict()
        cost_gap = np.abs(np.diff(list(cost_by_group.values()))[0]) if len(cost_by_group) == 2 else np.nan

        burden_by_group = recourse_info.groupby("group")["burden"].mean().to_dict()
        burden_gap = np.abs(np.diff(list(burden_by_group.values()))[0]) if len(burden_by_group) == 2 else np.nan

        # Build dictionary for this epoch
        epoch_metrics = {
            "epoch": epoch,
            "accuracy": train_accuracy,
            "burden_gap": burden_gap,
            "cost_gap": cost_gap
        }
        epoch_metrics.update(acc_by_group)
        epoch_metrics.update(tpr_by_group)
        epoch_metrics.update(fpr_by_group)
        epoch_metrics.update(ar_by_group)
        epoch_metrics.update({f"burden_group_{k}": v for k, v in burden_by_group.items()})
        epoch_metrics.update({f"cost_group_{k}": v for k, v in cost_by_group.items()})


        # Append to log
        metrics_log.append(epoch_metrics)

        print(f"Epoch {epoch} - Accuracy: {train_accuracy:.4f}, Burden Gap: {burden_gap:.4f}")
        
        if verbose:
            
            print(f"epoch={epoch} loss={weighted_loss_learner}")
            
        metrics_df = pd.DataFrame(metrics_log)
        
        metrics_df.to_csv(results_file, index=False)
        
    return learner, metrics_df, ml_model


# TODO: write optimization equal cost 

def train_equal_cost(dataset='adult', file_path=None, sens_attr='race', lr=0.001, epochs=10, batch_size=256, 
                        hidden_sizes=[128, 128], activation_name="relu", verbose=True, pretrain_epochs=2, 
                        recourse_method="gs", weighing_strategy="individual", recourse_hyperparam={}, 
                        results_file=None, random_state=42):
    
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

    dataset_train = CsvCatalog(file_path=file_path,
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
    
    training_params = {"lr": lr, "epochs": 1, "batch_size": batch_size, 
                       "hidden_size": hidden_sizes}
    
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
    
    # To store the performance metrics
    metrics_log = []

    
    for epoch in range(epochs):
        
        # Define the batches of the epoch
        dataset_len = len(dataset_train.df)
        batch_permu = np.random.permutation(dataset_len)
        batch_indices = np.array_split(batch_permu, np.ceil(dataset_len / batch_size))
        
        for current_batch in batch_indices:
            
            instance_weights = torch.ones(len(current_batch))
            
            # Create dataframe with current batch
            train_batch_df = dataset_train.df.iloc[current_batch]
            
            X_batch = train_batch_df.drop(columns=[y_var]).to_numpy()
            y_batch = train_batch_df[y_var].to_numpy()
            s_batch = train_batch_df[s_var].to_numpy()
            
            X_batch = torch.from_numpy(X_batch).to(torch.float32)
            y_batch = torch.from_numpy(preprocessing.LabelEncoder().fit_transform(y_batch))
            s_batch = torch.from_numpy(preprocessing.LabelEncoder().fit_transform(s_batch))
            
            if epoch > pretrain_epochs:
                    
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
                    "y_pred": y_pred_bin,
                    "cost": recourse_costs,
                    "burden": np.where(train_batch_df[y_var].to_numpy() == 0, 0, recourse_costs)
                })

                
                # Update instance weight based on cost
                social_burden_tensor = torch.tensor(recourse_df["burden"].to_numpy(), dtype=torch.float32)
                total_burden = social_burden_tensor.sum()
                
                cost_tensor = torch.tensor(recourse_df["cost"].to_numpy(), dtype=torch.float32)
                total_cost = cost_tensor.sum()
                
                # Find the minimum cost for the instances with a negative prediction
                instances_with_y_pred_0 = recourse_df[recourse_df['y_pred'] == 0]
                if not instances_with_y_pred_0.empty:
                    smallest_cost_for_y_pred_0 = instances_with_y_pred_0['cost'].min()
                else:
                    print("There are no instances where y_pred = 0 in the DataFrame.")
                    
                # Identify where cost_tensor is not zero (i.e., instances that have a positive prediction)
                non_zero_cost_mask = cost_tensor != 0

                # Perform the division only for instances with negative prediction
                instance_weights[non_zero_cost_mask] = smallest_cost_for_y_pred_0 / cost_tensor[non_zero_cost_mask]
                
                
                # # Average cost per group
                # group_cost = recourse_df.groupby("group")["cost"].mean()
                
                # print(group_cost)
                
                
                # # group_weight = group_cost.sum() / group_cost
                # group_weight = 10.0 / group_cost
                # # group_weight = 5 * group_cost / group_cost.sum()
                
                # print(group_weight)

                # # Map each instance's group to its weight
                # instance_group = recourse_df["group"]
                # instance_weights = instance_group.map(group_weight)
                
                # # Step 4: Compute weights
                # instance_weights = torch.tensor(instance_weights.to_numpy(),
                #     dtype=torch.float32
                # )
                
                print(instance_weights)
            
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
        
        # Get performance stats for the model at this point of the training
        
        with torch.no_grad():
            # Full training data
            train_df = dataset_train.df.copy()
            X_train_full = train_df.drop(columns=[y_var]).to_numpy()
            y_train_full = train_df[y_var].to_numpy()
            s_train_full = train_df[s_var].to_numpy()

            # Get model predictions
            X_train_tensor = torch.from_numpy(X_train_full).to(torch.float32)
            # y_pred_scores = ml_model._model(X_train_tensor).detach().numpy().reshape(-1)
            y_pred_scores = ml_model._model(X_train_tensor).detach().numpy()[:,1]
            # print(f"the length of y_pred_scores is {len(y_pred_scores)}")
            y_pred_bin = (y_pred_scores > 0.5).astype(int)

            # Accuracy
            y_encoded = preprocessing.LabelEncoder().fit_transform(y_train_full)
            train_accuracy = np.mean(y_pred_bin == y_encoded)

            # Group-wise accuracy
            # Convert ground truth and sensitive group to numpy arrays
            y_true = y_encoded 
            s_groups = s_train_full

            # Initialize dictionaries for group-wise metrics
            acc_by_group = {}
            tpr_by_group = {}
            fpr_by_group = {}
            ar_by_group = {}
            for group_val in np.unique(s_groups):
                idx = s_groups == group_val
                y_true_group = y_true[idx]
                y_pred_group = y_pred_bin[idx]

                acc_by_group[f"acc_group_{group_val}"] = np.mean(y_pred_group == y_true_group)

                # TPR: TP / (TP + FN)
                positives = y_true_group == 1
                tpr = np.sum((y_pred_group == 1) & positives) / (np.sum(positives) + 1e-8)
                tpr_by_group[f"tpr_group_{group_val}"] = tpr

                # FPR: FP / (FP + TN)
                negatives = y_true_group == 0
                fpr = np.sum((y_pred_group == 1) & negatives) / (np.sum(negatives) + 1e-8)
                fpr_by_group[f"fpr_group_{group_val}"] = fpr
                
                # AR: (FP + TP) / N_instances
                ar = np.sum((y_pred_group == 1)) / len(y_pred_group)
                ar_by_group[f"ar_group_{group_val}"] = ar

            # Recompute counterfactuals to evaluate burden
            factuals = train_df[y_pred_bin == 0]
            recourse_m = recourse_model_obj(ml_model, recourse_hyperparam)
            counterfactuals = recourse_m.get_counterfactuals(factuals)

            train_new_df = train_df.copy()
            factual_indices = factuals.index
            train_new_df.loc[factual_indices, counterfactuals.columns] = counterfactuals.values

            old_array = train_df.to_numpy()
            new_array = train_new_df.to_numpy()
            recourse_costs = np.linalg.norm(new_array - old_array, axis=1)

            recourse_info = pd.DataFrame({
                "group": s_train_full,
                "y_true": y_train_full,
                "cost": recourse_costs,
                "burden": np.where(np.array(y_train_full) == 0, 0, recourse_costs)
            })
            
            cost_by_group = recourse_info.groupby("group")["cost"].mean().to_dict()
            cost_gap = np.abs(np.diff(list(cost_by_group.values()))[0]) if len(cost_by_group) == 2 else np.nan

            burden_by_group = recourse_info.groupby("group")["burden"].mean().to_dict()
            burden_gap = np.abs(np.diff(list(burden_by_group.values()))[0]) if len(burden_by_group) == 2 else np.nan

            # Build dictionary for this epoch
            epoch_metrics = {
                "epoch": epoch,
                "accuracy": train_accuracy,
                "burden_gap": burden_gap,
                "cost_gap": cost_gap
            }
            epoch_metrics.update(acc_by_group)
            epoch_metrics.update(tpr_by_group)
            epoch_metrics.update(fpr_by_group)
            epoch_metrics.update(ar_by_group)
            epoch_metrics.update({f"burden_group_{k}": v for k, v in burden_by_group.items()})
            epoch_metrics.update({f"cost_group_{k}": v for k, v in cost_by_group.items()})


            # Append to log
            metrics_log.append(epoch_metrics)

            print(f"Epoch {epoch} - Accuracy: {train_accuracy:.4f}, Burden Gap: {burden_gap:.4f}")
        
        if verbose:
            
            print(f"epoch={epoch} loss={weighted_loss_learner}")
            
        metrics_df = pd.DataFrame(metrics_log)
        print(metrics_df)
        
        metrics_df.to_csv("training_metrics_log.csv", index=False)
        
    return learner, metrics_df, ml_model
    
    

def test_recourse(dataset=None, file_path=None, sens_attr='race', s_vals=None, ml_model=None,
                        recourse_method="gs", recourse_hyperparam={}, 
                        results_file=None, random_state=42):
    
    
    '''Test the final classifier. '''
    
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
            
    dataset_test = CsvCatalog(file_path=file_path,
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target='income')
            
    # Map from string to recourse method
    recourse_map = {
        "gs": GrowingSpheres,
        "ar": ActionableRecourse
    }
    
    if recourse_method.lower() not in recourse_map:
        raise ValueError(f"Unsupported recourse method: {recourse_method}. Supported: {list(recourse_map.keys())}")
    
    recourse_model_obj = recourse_map[recourse_method.lower()]
    
    # Classify test instances
    
    with torch.no_grad():
            # Full training data
            test_df = dataset_test.df.copy()
            X_test_full = test_df.drop(columns=[y_var]).to_numpy()
            y_test_full = test_df[y_var].to_numpy()
            if sens_attr == "age":
                s_test_full = s_vals
            else:
                s_test_full = test_df[s_var].astype(int).to_numpy()

            # Get model predictions
            X_test_tensor = torch.from_numpy(X_test_full).to(torch.float32)
            y_pred_scores = ml_model._model(X_test_tensor).detach().numpy()[:,1]
            y_pred_bin = (y_pred_scores > 0.5).astype(int)

            # Accuracy
            y_encoded = preprocessing.LabelEncoder().fit_transform(y_test_full)
            test_accuracy = np.mean(y_pred_bin == y_encoded)

            # Group-wise accuracy
            # Convert ground truth and sensitive group to numpy arrays
            y_true = y_encoded 
            s_groups = s_test_full

            # Initialize dictionaries for group-wise metrics
            acc_by_group = {}
            tpr_by_group = {}
            fpr_by_group = {}
            ar_by_group = {}
            for group_val in np.unique(s_groups):
                idx = s_groups == group_val
                y_true_group = y_true[idx]
                y_pred_group = y_pred_bin[idx]

                acc_by_group[f"acc_group_{group_val}"] = np.mean(y_pred_group == y_true_group)

                # TPR: TP / (TP + FN)
                positives = y_true_group == 1
                tpr = np.sum((y_pred_group == 1) & positives) / (np.sum(positives) + 1e-8)
                tpr_by_group[f"tpr_group_{group_val}"] = tpr

                # FPR: FP / (FP + TN)
                negatives = y_true_group == 0
                fpr = np.sum((y_pred_group == 1) & negatives) / (np.sum(negatives) + 1e-8)
                fpr_by_group[f"fpr_group_{group_val}"] = fpr
                
                # AR: (FP + TP) / N_instances
                ar = np.sum((y_pred_group == 1)) / len(y_pred_group)
                ar_by_group[f"ar_group_{group_val}"] = ar

            # Recompute counterfactuals to evaluate burden
            factuals = test_df[y_pred_bin == 0]
            recourse_m = recourse_model_obj(ml_model, recourse_hyperparam)
            counterfactuals = recourse_m.get_counterfactuals(factuals)

            test_new_df = test_df.copy()
            factual_indices = factuals.index
            test_new_df.loc[factual_indices, counterfactuals.columns] = counterfactuals.values

            old_array = test_df.to_numpy()
            new_array = test_new_df.to_numpy()
            recourse_costs = np.linalg.norm(new_array - old_array, axis=1)

            recourse_info = pd.DataFrame({
                "group": s_test_full,
                "y_true": y_test_full,
                "cost": recourse_costs,
                "burden": np.where(np.array(y_test_full) == 0, 0, recourse_costs)
            })
            
            cost_by_group_raw = recourse_info.groupby("group")["cost"].mean().to_dict()
            cost_by_group = {f"cost_group_{group}": cost for group, cost in cost_by_group_raw.items()}
            cost_gap = np.abs(np.diff(list(cost_by_group.values()))[0]) if len(cost_by_group) == 2 else np.nan

            burden_by_group_raw = recourse_info.groupby("group")["burden"].mean().to_dict()
            burden_by_group = {f"burden_group_{group}": burden for group, burden in burden_by_group_raw.items()}
            burden_gap = np.abs(np.diff(list(burden_by_group.values()))[0]) if len(burden_by_group) == 2 else np.nan
            
            
    # Initialize result dictionary
    results_dict = {
        "overall_accuracy": {"all": test_accuracy},
    }

    # Add group-wise accuracies
    for k, v in acc_by_group.items():
        group = k.replace("acc_group_", "")
        results_dict.setdefault("group_accuracy", {})[group] = v

    # Add group-wise TPR
    for k, v in tpr_by_group.items():
        group = k.replace("tpr_group_", "")
        results_dict.setdefault("group_tpr", {})[group] = v
        
    # Add group-wise AR
    for k, v in ar_by_group.items():
        group = k.replace("ar_group_", "")
        results_dict.setdefault("group_ar", {})[group] = v
        
    for k, v in cost_by_group.items():
        group = k.replace("cost_group_", "")
        results_dict.setdefault("group_cost", {})[group] = v
    
    for k, v in burden_by_group.items():
        group = k.replace("burden_group_", "")
        results_dict.setdefault("group_burden", {})[group] = v

    # Convert to DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient="index").sort_index()
    
    results_df.to_csv(results_file, index=False)
    
    return results_df 
    
    


# -- Dataset parameters --
# dataset = sys.argv[1]
dataset = "adult"
# sens_attr = sys.argv[2]
sens_attr = "race" # "age", "race", "sex"
dataset_path =  dataset
df_orig = pd.read_csv(f"{dataset_path}.csv")


# -- Training parameters -- 
pretrain_epoch = 2
total_epoch = 5
batch_size = 256
learning_rate = 0.001
hidden_sizes = [128, 128]
activation_function = 'relu'

# -- Define recourse method --
recourse_method = 'CCHVAE'

if recourse_method == 'GS':
    recourse_hyperparam = {}
elif recourse_method == 'AR':
    recourse_hyperparam = {"fs_size": 5, "binary_cat_features": True, "discretize": True}
elif recourse_method == "CCHVAE":
    recourse_hyperparam = {} # they are defined inside the train function 
    # TODO: Fix this 

# Define random state
random_state = 42

# Get part for training and store the other part for dyamic simulation
df_train, df_test = train_test_split(df_orig, test_size=0.95, random_state=random_state)
# to make a smaller test set:
df_test, _ = train_test_split(df_test, test_size=0.95, random_state=random_state)
if sens_attr == "age":
    s_train = (df_train["age"] > 30).astype(int).to_numpy()
    s_test = (df_test["age"] > 30).astype(int).to_numpy()
else:
    s_train = df_train[sens_attr].to_numpy()
    s_test = df_test[sens_attr].to_numpy()
    
# The strategy for fairness in recourse
fair_strategy = "minimax_burden" # "eq_cost", "minimax_burden"    
    
# Save train into dataframe 
df_train.to_csv(f"{dataset_path}_train.csv", index=False)
df_test.to_csv(f"{dataset_path}_test.csv", index=False)

train_file_path = f"{dataset_path}_train.csv"
train_results_file_path = f"results_num/{dataset_path}_{recourse_method}_{fair_strategy}_pretrain{pretrain_epoch}_total{total_epoch}_sens{sens_attr}_rs{random_state}_training_metrics_log.csv"

test_file_path = f"{dataset_path}_test.csv"
test_results_file_path = f"results_num/{dataset_path}_{recourse_method}_{fair_strategy}_pretrain{pretrain_epoch}_total{total_epoch}_sens{sens_attr}_rs{random_state}_test_metrics_log.csv"


if fair_strategy == "minimax_burden":
    my_trained_model, train_metrics, ml_model = train_social_burden(dataset=dataset, file_path=train_file_path, sens_attr=sens_attr, s_vals=s_train,
                                        lr=learning_rate, epochs=total_epoch, batch_size=batch_size, 
                                        hidden_sizes=hidden_sizes, activation_name=activation_function, verbose=True, pretrain_epochs=pretrain_epoch, 
                                        recourse_method=recourse_method, weighing_strategy="individual", recourse_hyperparam=recourse_hyperparam, 
                                        results_file=train_results_file_path, random_state=random_state)
    
elif fair_strategy == "eq_cost":
    my_trained_model, train_metrics, ml_model = train_equal_cost(dataset='adult', file_path=train_file_path, sens_attr='race', lr=0.001, epochs=total_epoch, 
                                                                 batch_size=256, hidden_sizes=[128, 128], activation_name="relu", verbose=True, pretrain_epochs=pretrain_epoch, 
                                                                 recourse_method="gs", weighing_strategy="individual", recourse_hyperparam={}, 
                                                                 results_file=train_results_file_path, random_state=42)


print(train_metrics)

# Plot the metrics

plt.figure(figsize=(16, 16))

# Subplot 1: Accuracy
plt.subplot(5, 1, 1)
plt.plot(train_metrics["epoch"], train_metrics["accuracy"], label="Overall Accuracy", color="black", linewidth=2)
plt.plot(train_metrics["epoch"], train_metrics["acc_group_0"], label="Group 0 Accuracy", linestyle="--")
plt.plot(train_metrics["epoch"], train_metrics["acc_group_1"], label="Group 1 Accuracy", linestyle="--")
plt.axvline(pretrain_epoch, color='gray', linestyle=':', linewidth=2, label="Pretraining End")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Subplot 2: TPR and FPR per group
plt.subplot(5, 1, 2)
plt.plot(train_metrics["epoch"], train_metrics["tpr_group_0"], label="TPR Group 0", linestyle="-", color="blue")
plt.plot(train_metrics["epoch"], train_metrics["tpr_group_1"], label="TPR Group 1", linestyle="-", color="cyan")
plt.plot(train_metrics["epoch"], train_metrics["fpr_group_0"], label="FPR Group 0", linestyle="--", color="red")
plt.plot(train_metrics["epoch"], train_metrics["fpr_group_1"], label="FPR Group 1", linestyle="--", color="orange")
plt.axvline(pretrain_epoch, color='gray', linestyle=':', linewidth=2, label="Pretraining End")
plt.title("TPR and FPR per Group")
plt.xlabel("Epoch")
plt.ylabel("Rate")
plt.legend()
plt.grid(True)

# Subplot 3: Social Burden
plt.subplot(5, 1, 3)
plt.plot(train_metrics["epoch"], train_metrics["burden_group_0"], label="Burden Group 0", color="green")
plt.plot(train_metrics["epoch"], train_metrics["burden_group_1"], label="Burden Group 1", color="purple")
plt.axvline(pretrain_epoch, color='gray', linestyle=':', linewidth=2, label="Pretraining End")
plt.title("Social Burden per Group")
plt.xlabel("Epoch")
plt.ylabel("Burden")
plt.legend()
plt.grid(True)

# Subplot 2: TPR and FPR per group
plt.subplot(5, 1, 4)
plt.plot(train_metrics["epoch"], train_metrics["ar_group_0"], label="AR Group 0", linestyle="-", color="blue")
plt.plot(train_metrics["epoch"], train_metrics["ar_group_1"], label="AR Group 1", linestyle="-", color="cyan")
plt.axvline(pretrain_epoch, color='gray', linestyle=':', linewidth=2, label="Pretraining End")
plt.title("AR per Group")
plt.xlabel("Epoch")
plt.ylabel("Rate")
plt.legend()
plt.grid(True)

# Subplot 3: Social Burden
plt.subplot(5, 1, 5)
plt.plot(train_metrics["epoch"], train_metrics["cost_group_0"], label="Cost Group 0", color="green")
plt.plot(train_metrics["epoch"], train_metrics["cost_group_1"], label="Cost Group 1", color="purple")
plt.axvline(pretrain_epoch, color='gray', linestyle=':', linewidth=2, label="Pretraining End")
plt.title("Cost per Group")
plt.xlabel("Epoch")
plt.ylabel("Burden")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(f"result_figures/{fair_strategy}_{recourse_method}_pretrain{pretrain_epoch}_total{total_epoch}_sens{sens_attr}_rs{random_state}_training_metrics_plot.png", dpi=300, bbox_inches='tight')
# plt.show()


# Get test metrics

test_metrics = test_recourse(dataset="adult", file_path=test_file_path, sens_attr='race', s_vals=s_test, ml_model=ml_model,
                        recourse_method="gs", recourse_hyperparam={}, 
                        results_file=test_results_file_path, random_state=42)

print(test_metrics)


