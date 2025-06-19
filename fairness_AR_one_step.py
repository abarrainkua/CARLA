## We study the relation between algorithmic fairness 
## and fairness in algorithmic recourse
## the relations between the fairness metrics from both areas

import numpy as np
import pandas as pd
from carla.models.catalog import  MLModelCatalog
from carla.data.catalog import OnlineCatalog, CsvCatalog, DataCatalog
from carla.recourse_methods import GrowingSpheres, ActionableRecourse, CCHVAE, Wachter, Face
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
import itertools
import argparse


def train_social_burden(dataset='adult', file_path=None, sens_attr=['race'], lr=0.001, epochs=10, batch_size=256, 
                        hidden_sizes=[128, 128], activation_name="relu", verbose=True, pretrain_epochs=2, save_training_metrics=True,
                        n_inst_eval_train_metrics=100, recourse_method="GS", recourse_hyperparam={}, 
                        results_file=None, random_state=42):
    
    if dataset == "adult":
        
        continuous = ["age", "fnlwgt", "education-num", "capital-gain", "hours-per-week", "capital-loss"]
        categorical = ["marital-status", "native-country", "occupation", "race", "relationship", "sex", "workclass"]
        immutable = ["age", "sex", "race"]
        y_var = "income"
        
        # The sensitive attribute mapping
        mapping = {
            "race": "race_White",
            "sex": "sex_Male",
            "age": "age_bin"
        }
        
        # Create the corresponding s_var list
        s_var = [mapping[attr] for attr in sens_attr]
    
        # if sens_attr == "race":
        #     s_var = "race_White"
        # elif sens_attr == "sex":
        #     s_var = "sex_Male"
        # elif sens_attr == "age":
        #     s_var = "age_bin"

    dataset_train = CsvCatalog(file_path=file_path,
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target='income')
    
    # Create the s_vals matrix with the values of the sensitive attribute for all the characterizations
    data_no_preprocess = pd.read_csv(file_path) # necessary to do this for the attribute age (to have orginal without processing)
    s_vals_list = []
    for attr, var in zip(sens_attr, s_var):
        if attr == "age":
            # Apply binarization for age directly from raw data
            s_col = (data_no_preprocess["age"] > 30).astype(int).to_numpy()
        else:
            # Use the processed column from the dataset
            s_col = dataset_train.df[var].to_numpy()
        
        s_vals_list.append(s_col)

    # Stack the columns horizontally to get a 2D matrix
    s_vals = np.column_stack(s_vals_list)
    
    # --- Augment s_vals with all the possible intersectional groups ---
    
    # Step 1: Build DataFrame with s_var column names
    df_sens = pd.DataFrame(s_vals, columns=s_var)

    # Step 2: Start with original columns
    augmented_df = df_sens.copy()

    # Step 3: Add intersectional combinations
    for r in range(2, len(s_var) + 1):
        for combo in itertools.combinations(s_var, r):
            combo_name = "_&_".join(combo)
            group_ids, _ = pd.factorize(list(zip(*(df_sens[col] for col in combo))))
            augmented_df[combo_name] = group_ids
            
    # Convert again to matrix
    s_vals = augmented_df.to_numpy()
    
    # Get all the sensitive groupings
    s_column_names = augmented_df.columns
    
    # -------------------------------------------------------------------


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
        "CCHVAE": CCHVAE,
        "WT": Wachter,
        "FACE": Face
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
            s_batch = s_vals[current_batch]
            # if sens_attr == "age":
            #     s_batch = s_vals[current_batch]
            # else:
            #     s_batch = train_batch_df[s_var].astype(int).to_numpy()
            
            X_batch = torch.from_numpy(X_batch).to(torch.float32)
            y_batch = torch.from_numpy(preprocessing.LabelEncoder().fit_transform(y_batch))
            s_batch = torch.from_numpy(s_batch).to(torch.int32)
            # s_batch = torch.from_numpy(preprocessing.LabelEncoder().fit_transform(s_batch))
            
            
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
                
                # # Create a DataFrame with recourse information
                # recourse_df = pd.DataFrame({
                #     "group": s_batch.cpu().detach().numpy(),
                #     "y_true": train_batch_df[y_var].to_numpy(),
                #     "cost": recourse_costs,
                #     "burden": np.where(train_batch_df[y_var].to_numpy() == 0, 0, recourse_costs)
                # })
            

                # Start building the full recourse DataFrame from the sensitive information
                recourse_df = pd.DataFrame(
                    s_batch.cpu().detach().numpy(),  # sensitive attribute values
                    columns=s_column_names  # both one-dimensional and intersectional names
                )

                # Add outcome and cost information
                recourse_df["y_true"] = train_batch_df[y_var].to_numpy()
                recourse_df["cost"] = recourse_costs
                recourse_df["burden"] = np.where(recourse_df["y_true"] == 0, 0, recourse_costs)

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
        
        # If save_training_metrics = True, Get performance stats for the model at this point of the training
        
        if save_training_metrics:
        
            with torch.no_grad():
                # Full training data
                train_df = dataset_train.df.copy()
                # Get a subsample of n_inst_eval_train_metrics instances
                subset_train_df = train_df.sample(n=n_inst_eval_train_metrics, random_state=random_state+99*epoch)
                # Save which instances have been selected
                subset_indices = subset_train_df.index
                X_train_full = subset_train_df.drop(columns=[y_var]).to_numpy()
                y_train_full = subset_train_df[y_var].to_numpy()
                s_train_full = s_vals[subset_indices]
                # if sens_attr == "age":
                #     s_train_full = s_vals[subset_indices]
                # else:
                #     s_train_full = subset_train_df[s_var].astype(int).to_numpy()

                # Get model predictions
                X_train_tensor = torch.from_numpy(X_train_full).to(torch.float32)
                y_pred_scores = ml_model._model(X_train_tensor).detach().numpy()[:,1]
                y_pred_bin = (y_pred_scores > 0.5).astype(int)

            # Accuracy
            y_encoded = preprocessing.LabelEncoder().fit_transform(y_train_full)
            train_accuracy = np.mean(y_pred_bin == y_encoded)

            # Convert ground truth and sensitive group to numpy arrays
            y_true = y_encoded 
            s_groups = s_train_full

            # Initialize dictionaries to store metrics
            acc_by_group = {}
            tpr_by_group = {}
            fpr_by_group = {}
            ar_by_group = {}

            for i, col_name in enumerate(s_column_names):
                s_col = s_groups[:, i]  # extract the i-th sensitive attribute column

                for group_val in np.unique(s_col):
                    idx = s_col == group_val
                    y_true_group = y_true[idx]
                    y_pred_group = y_pred_bin[idx]

                    # Build key using actual values for each attribute
                    if "_&_" in col_name:
                        attrs = col_name.split("_&_")
                        example_idx = np.where(idx)[0][0]  # get one matching row index
                        full_label_parts = []

                        for attr in attrs:
                            attr_index = s_column_names.get_loc(attr)
                            attr_val = s_groups[example_idx, attr_index]
                            full_label_parts.append(f"{attr}_{attr_val}")
                        key_prefix = "_&_".join(full_label_parts)
                    else:
                        key_prefix = f"{col_name}_{group_val}"

                    # Compute and store metrics using the full label
                    acc_by_group[f"acc_{key_prefix}"] = np.mean(y_pred_group == y_true_group)

                    positives = y_true_group == 1
                    tpr = np.sum((y_pred_group == 1) & positives) / (np.sum(positives) + 1e-8)
                    tpr_by_group[f"tpr_{key_prefix}"] = tpr

                    negatives = y_true_group == 0
                    fpr = np.sum((y_pred_group == 1) & negatives) / (np.sum(negatives) + 1e-8)
                    fpr_by_group[f"fpr_{key_prefix}"] = fpr

                    ar = np.mean(y_pred_group == 1)
                    ar_by_group[f"ar_{key_prefix}"] = ar
                    
            # Recompute counterfactuals to evaluate burden
            factuals = subset_train_df[y_pred_bin == 0]
            recourse_m = recourse_model_obj(ml_model, recourse_hyperparam)
            print("getting counterfactuals")
            counterfactuals = recourse_m.get_counterfactuals(factuals)
            print("finished getting counterfactuals")

            train_new_df = subset_train_df.copy()
            factual_indices = factuals.index
            train_new_df.loc[factual_indices, counterfactuals.columns] = counterfactuals.values

            old_array = subset_train_df.to_numpy()
            new_array = train_new_df.to_numpy()
            recourse_costs = np.linalg.norm(new_array - old_array, axis=1)
            
            # Start building the full recourse DataFrame from the sensitive information
            recourse_info = pd.DataFrame(
                s_train_full,  # sensitive attribute values
                columns=s_column_names  # both one-dimensional and intersectional names
            )

            # Add outcome and cost information
            recourse_info["y_true"] = y_train_full
            recourse_info["cost"] = recourse_costs
            recourse_info["burden"] = np.where(np.array(y_train_full) == 0, 0, recourse_costs)

            # recourse_info = pd.DataFrame({
            #     "group": s_train_full,
            #     "y_true": y_train_full,
            #     "cost": recourse_costs,
            #     "burden": np.where(np.array(y_train_full) == 0, 0, recourse_costs)
            # })
            
            cost_by_group_all = {}
            burden_by_group_all = {}
            cost_gap_all = {}
            burden_gap_all = {}

            for col in s_column_names:
                # Compute group-wise means for cost and burden
                cost_by_group = recourse_info.groupby(col)["cost"].mean().to_dict()
                burden_by_group = recourse_info.groupby(col)["burden"].mean().to_dict()

                # If the column is intersectional, decode the values
                if "_&_" in col:
                    # Parse original attribute names
                    attrs = col.split("_&_")

                    # Decode each unique group into readable attribute-value pairs
                    for group_id, cost in cost_by_group.items():
                        group_mask = recourse_info[col] == group_id
                        example_row = recourse_info.loc[group_mask].iloc[0]  # any representative row

                        # Create a name like race_White_0.0&_sex_Male_1.0
                        full_label_parts = []
                        for attr in attrs:
                            val = example_row[attr]
                            full_label_parts.append(f"{attr}_{val}")
                        full_label = "_&_".join(full_label_parts)

                        cost_by_group_all[f"cost_{full_label}_group_{group_id}"] = cost
                        burden_by_group_all[f"burden_{full_label}_group_{group_id}"] = burden_by_group[group_id]
                else:
                    # Single-attribute case
                    for group_id, cost in cost_by_group.items():
                        full_label = f"{col}_{group_id}"
                        cost_by_group_all[f"cost_{full_label}_group_{group_id}"] = cost
                        burden_by_group_all[f"burden_{full_label}_group_{group_id}"] = burden_by_group[group_id]

                # Compute gap (max - min)
                cost_vals = list(cost_by_group.values())
                burden_vals = list(burden_by_group.values())
                cost_gap_all[f"cost_gap_{col}"] = np.max(cost_vals) - np.min(cost_vals) if cost_vals else np.nan
                burden_gap_all[f"burden_gap_{col}"] = np.max(burden_vals) - np.min(burden_vals) if burden_vals else np.nan


            # Build dictionary for this epoch
            epoch_metrics = {
                "epoch": epoch,
                "accuracy": train_accuracy
            }
            epoch_metrics.update(cost_gap_all)
            epoch_metrics.update(burden_gap_all)
            epoch_metrics.update(acc_by_group)
            epoch_metrics.update(tpr_by_group)
            epoch_metrics.update(fpr_by_group)
            epoch_metrics.update(ar_by_group)
            epoch_metrics.update(burden_by_group_all)
            epoch_metrics.update(cost_by_group_all)
            
            # cost_by_group = recourse_info.groupby("group")["cost"].mean().to_dict()
            # cost_gap = np.abs(np.diff(list(cost_by_group.values()))[0]) if len(cost_by_group) == 2 else np.nan

            # burden_by_group = recourse_info.groupby("group")["burden"].mean().to_dict()
            # burden_gap = np.abs(np.diff(list(burden_by_group.values()))[0]) if len(burden_by_group) == 2 else np.nan

            # # Build dictionary for this epoch
            # epoch_metrics = {
            #     "epoch": epoch,
            #     "accuracy": train_accuracy,
            #     "burden_gap": burden_gap,
            #     "cost_gap": cost_gap
            # }
            # epoch_metrics.update(acc_by_group)
            # epoch_metrics.update(tpr_by_group)
            # epoch_metrics.update(fpr_by_group)
            # epoch_metrics.update(ar_by_group)
            # epoch_metrics.update({f"burden_group_{k}": v for k, v in burden_by_group.items()})
            # epoch_metrics.update({f"cost_group_{k}": v for k, v in cost_by_group.items()})


            # Append to log
            metrics_log.append(epoch_metrics)

            # print(f"Epoch {epoch} - Accuracy: {train_accuracy:.4f}, Burden Gap: {burden_gap_all:.4f}")
                
            metrics_df = pd.DataFrame(metrics_log)
            metrics_df.to_csv(results_file, index=False)
            
        if verbose:
                print(f"epoch={epoch} loss={weighted_loss_learner}")
        
    return learner, metrics_df, ml_model
    

def test_recourse(dataset=None, file_path=None, sens_attr='race', ml_model=None,
                        recourse_method="gs", recourse_hyperparam={}, 
                        results_file=None, random_state=42):
    
    
    '''Test the final classifier. '''
    
    if dataset == "adult":
        
        continuous = ["age", "fnlwgt", "education-num", "capital-gain", "hours-per-week", "capital-loss"]
        categorical = ["marital-status", "native-country", "occupation", "race", "relationship", "sex", "workclass"]
        immutable = ["age", "sex", "race"]
        y_var = "income"
        
        # The sensitive attribute mapping
        mapping = {
            "race": "race_White",
            "sex": "sex_Male",
            "age": "age_bin"
        }
        
        # Create the corresponding s_var list
        s_var = [mapping[attr] for attr in sens_attr]
    
            
    dataset_test = CsvCatalog(file_path=file_path,
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target='income')
    
    # Create the s_vals matrix with the values of the sensitive attribute for all the characterizations
    data_no_preprocess = pd.read_csv(file_path) # necessary to do this for the attribute age (to have orginal without processing)
    s_vals_list = []
    for attr, var in zip(sens_attr, s_var):
        if attr == "age":
            # Apply binarization for age directly from raw data
            s_col = (data_no_preprocess["age"] > 30).astype(int).to_numpy()
        else:
            # Use the processed column from the dataset
            s_col = dataset_test.df[var].to_numpy()
        
        s_vals_list.append(s_col)

    # Stack the columns horizontally to get a 2D matrix
    s_vals = np.column_stack(s_vals_list)
    
    # --- Augment s_vals with all the possible intersectional groups ---
    
    # Step 1: Build DataFrame with s_var column names
    df_sens = pd.DataFrame(s_vals, columns=s_var)

    # Start with original columns
    augmented_df = df_sens.copy()

    # Add intersectional combinations
    for r in range(2, len(s_var) + 1):
        for combo in itertools.combinations(s_var, r):
            combo_name = "_&_".join(combo)
            group_ids, _ = pd.factorize(list(zip(*(df_sens[col] for col in combo))))
            augmented_df[combo_name] = group_ids
            
    # Convert again to matrix
    s_vals = augmented_df.to_numpy()
    
    # Get all the sensitive groupings
    s_column_names = augmented_df.columns
            
    # Map from string to recourse method
    recourse_map = {
        "GS": GrowingSpheres,
        "AR": ActionableRecourse,
        "CCHVAE": CCHVAE,
        "WT": Wachter,
        "FACE": Face
    }
    
    if recourse_method.upper() not in recourse_map:
        raise ValueError(f"Unsupported recourse method: {recourse_method}. Supported: {list(recourse_map.keys())}")
    
    recourse_model_obj = recourse_map[recourse_method.upper()]
    
    # Classify test instances
    
    with torch.no_grad():
            # Full training data
            test_df = dataset_test.df.copy()
            X_test_full = test_df.drop(columns=[y_var]).to_numpy()
            y_test_full = test_df[y_var].to_numpy()
            s_test_full = s_vals
            
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

    # Initialize dictionaries to store metrics
    acc_by_group = {}
    tpr_by_group = {}
    fpr_by_group = {}
    ar_by_group = {}

    # Loop through each sensitive attribute column
    for i, col_name in enumerate(s_column_names):
        s_col = s_groups[:, i]  # extract the i-th sensitive attribute column

        for group_val in np.unique(s_col):
            idx = s_col == group_val
            y_true_group = y_true[idx]
            y_pred_group = y_pred_bin[idx]

            # Build key using actual values for each attribute
            if "_&_" in col_name:
                attrs = col_name.split("_&_")
                example_idx = np.where(idx)[0][0]  # get one matching row index
                full_label_parts = []

                for attr in attrs:
                    attr_index = s_column_names.get_loc(attr)
                    attr_val = s_groups[example_idx, attr_index]
                    full_label_parts.append(f"{attr}_{attr_val}")
                key_prefix = "_&_".join(full_label_parts)
            else:
                key_prefix = f"{col_name}_{group_val}"

            # Compute and store metrics using the full label
            acc_by_group[f"acc_{key_prefix}"] = np.mean(y_pred_group == y_true_group)

            positives = y_true_group == 1
            tpr = np.sum((y_pred_group == 1) & positives) / (np.sum(positives) + 1e-8)
            tpr_by_group[f"tpr_{key_prefix}"] = tpr

            negatives = y_true_group == 0
            fpr = np.sum((y_pred_group == 1) & negatives) / (np.sum(negatives) + 1e-8)
            fpr_by_group[f"fpr_{key_prefix}"] = fpr

            ar = np.mean(y_pred_group == 1)
            ar_by_group[f"ar_{key_prefix}"] = ar
    
    
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
    
    # Start building the full recourse DataFrame from the sensitive information
    recourse_info = pd.DataFrame(
        s_test_full,  # sensitive attribute values
        columns=s_column_names  # both one-dimensional and intersectional names
    )

    # Add outcome and cost information
    recourse_info["y_true"] = y_test_full
    recourse_info["cost"] = recourse_costs
    recourse_info["burden"] = np.where(np.array(y_test_full) == 0, 0, recourse_costs)
    
    cost_by_group_all = {}
    burden_by_group_all = {}
    cost_gap_all = {}
    burden_gap_all = {}

    for col in s_column_names:
        # Compute group-wise means for cost and burden
        cost_by_group = recourse_info.groupby(col)["cost"].mean().to_dict()
        burden_by_group = recourse_info.groupby(col)["burden"].mean().to_dict()

        # If the column is intersectional, decode the values
        if "_&_" in col:
            # Parse original attribute names
            attrs = col.split("_&_")

            # Decode each unique group into readable attribute-value pairs
            for group_id, cost in cost_by_group.items():
                group_mask = recourse_info[col] == group_id
                example_row = recourse_info.loc[group_mask].iloc[0]  # any representative row

                # Create a name like race_White_0.0&_sex_Male_1.0
                full_label_parts = []
                for attr in attrs:
                    val = example_row[attr]
                    full_label_parts.append(f"{attr}_{val}")
                full_label = "_&_".join(full_label_parts)

                cost_by_group_all[f"cost_{full_label}_group_{group_id}"] = cost
                burden_by_group_all[f"burden_{full_label}_group_{group_id}"] = burden_by_group[group_id]
        else:
            # Single-attribute case
            for group_id, cost in cost_by_group.items():
                full_label = f"{col}_{group_id}"
                cost_by_group_all[f"cost_{full_label}_group_{group_id}"] = cost
                burden_by_group_all[f"burden_{full_label}_group_{group_id}"] = burden_by_group[group_id]

        # Compute gap (max - min)
        cost_vals = list(cost_by_group.values())
        burden_vals = list(burden_by_group.values())
        cost_gap_all[f"cost_gap_{col}"] = np.max(cost_vals) - np.min(cost_vals) if cost_vals else np.nan
        burden_gap_all[f"burden_gap_{col}"] = np.max(burden_vals) - np.min(burden_vals) if burden_vals else np.nan
            
            
    ## Initialize result dictionary
    results_dict = {
        "overall_accuracy": {"all": test_accuracy},
    }
    
    # Helper function to extract attr name + group value
    def parse_group_key(key, prefix):
        """
        Extracts the full attribute string after the metric prefix.
        For example:
        key = "acc_race_White_0.0&_sex_Male_1.0", prefix = "acc_"
        returns: "race_White_0.0&_sex_Male_1.0"
        """
        if not key.startswith(prefix):
            raise ValueError(f"Key '{key}' does not start with expected prefix '{prefix}'")
        return key[len(prefix):]  # just strip the prefix and return the rest


    # Generic updater
    def update_nested_dict(metric_name, group_dict, prefix):
        for k, v in group_dict.items():
            group_label = parse_group_key(k, prefix)  # e.g., "race_White_0.0&_sex_Male_1.0"
            results_dict.setdefault(metric_name, {})[group_label] = v
            
    # Group-wise accuracy, TPR, AR
    update_nested_dict("group_accuracy", acc_by_group, prefix="acc_")
    update_nested_dict("group_tpr", tpr_by_group, prefix="tpr_")
    update_nested_dict("group_ar", ar_by_group, prefix="ar_")

    # Cost and burden
    update_nested_dict("group_cost", cost_by_group_all, prefix="cost_")
    update_nested_dict("group_burden", burden_by_group_all, prefix="burden_")

    # Optional: Flatten results_dict for DataFrame conversion
    results_df = pd.json_normalize(results_dict, sep="/").T
    results_df.columns = ["value"]
    results_df.index.name = "metric/group"
    results_df = results_df.sort_index()
    
    results_df.to_csv(results_file, index=False)
    
    return results_df 

def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dataset-name", type=str, default="adult", help="The name of the dataset")
    parser.add_argument("--sens-attr", default=["race", "sex", "age"], nargs='?', help="The sensitive attribute(s)")
    parser.add_argument("--test-size", type=float, default=0.3, help="The proportion of points for test.")
    parser.add_argument("--pre-epoch", type=int, default=2, help="The number of pre-train epochs (warm-up).")
    parser.add_argument("--total-epoch", type=int, default=5, help="The number of total epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training the classifier")
    parser.add_argument("--n-eval-train-metrics", type=int, default=1000, help="The number of instances in which to evaluate the training metrics.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training the classifier")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function for the NN")
    parser.add_argument("--train-recourse-method", type=str, default="GS", help="The method for recourse at training.")
    parser.add_argument("--test-recourse-method", type=str, default=None, help="The method for recourse at deployment.")
    parser.add_argument("--hidden-sizes", default=[128, 128], nargs='?', help="The size(s) of the hidden layer(s).")
    parser.add_argument("--fair_strategy", type=str, default="burden", help="The fairness strategy.")

    # fmt: on

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    

    
    # Load dataset
    df_orig = pd.read_csv(f"{args.dataset_name}.csv")

    # --- Training recourse method ---

    if args.train_recourse_method == 'GS':
        recourse_hyperparam_train = {}
    elif args.train_recourse_method == 'AR':
        recourse_hyperparam_train = {"fs_size": 5, "binary_cat_features": True, "discretize": True}
    elif args.train_recourse_method == "CCHVAE":
        recourse_hyperparam_train = {} # they are defined inside the train function 
        # TODO: Fix this 
    elif args.train_recourse_method == "WT":
        recourse_hyperparam_train = {"loss_type": "MSE", "y_target": [1.0], "binary_cat_features": True}
    elif args.train_recourse_method == "FACE":
        recourse_hyperparam_train = {"mode": "knn", "fraction": 0.1}
    
    
    # --- Test recourse method ---
    
    if args.test_recourse_method is None:
        args.test_recourse_method = args.train_recourse_method
        recourse_hyperparam_test = recourse_hyperparam_train
    else:
        if args.test_recourse_method == 'GS':
            recourse_hyperparam_test = {}
        elif args.test_recourse_method == 'AR':
            recourse_hyperparam_test = {"fs_size": 5, "binary_cat_features": True, "discretize": True}
        elif args.test_recourse_method == "CCHVAE":
            recourse_hyperparam_test = {} # they are defined inside the train function 
            # TODO: Fix this 
        elif args.test_recourse_method == "WT":
            recourse_hyperparam_test = {"loss_type": "MSE", "y_target": [1.0], "binary_cat_features": True}
        elif args.test_recourse_method == "FACE":
            recourse_hyperparam_test = {"mode": "knn", "fraction": 0.1}
        



    # Get part for training and store the other part for dyamic simulation
    df_train, df_test = train_test_split(df_orig, test_size=0.95, random_state=args.random_state)
    # # to make a smaller test set:
    df_test, _ = train_test_split(df_test, test_size=0.95, random_state=args.random_state)
        
    # Save train into dataframe 
    df_train.to_csv(f"{args.dataset_name}_train.csv", index=False)
    df_test.to_csv(f"{args.dataset_name}_test.csv", index=False)

    train_file_path = f"{args.dataset_name}_train.csv"
    train_results_file_path = f"results_num/{args.dataset_name}_{args.train_recourse_method}_{args.fair_strategy}_pretrain{args.pre_epoch}_total{args.total_epoch}_sens{args.sens_attr}_rs{args.random_state}_training_metrics_log.csv"

    test_file_path = f"{args.dataset_name}_test.csv"
    test_results_file_path = f"results_num/{args.dataset_name}_train{args.train_recourse_method}_test{args.test_recourse_method}_{args.fair_strategy}_pretrain{args.pre_epoch}_total{args.total_epoch}_sens{args.sens_attr}_rs{args.random_state}_test_metrics_log.csv"


    if args.fair_strategy == "burden":
        my_trained_model, train_metrics, ml_model = train_social_burden(dataset=args.dataset_name, file_path=train_file_path, sens_attr=args.sens_attr,
                                            lr=args.learning_rate, epochs=args.total_epoch, batch_size=args.batch_size, n_inst_eval_train_metrics=args.n_eval_train_metrics,
                                            hidden_sizes=args.hidden_sizes, activation_name=args.activation, verbose=True, pretrain_epochs=args.pre_epoch, 
                                            recourse_method=args.train_recourse_method, recourse_hyperparam=recourse_hyperparam_train, 
                                            results_file=train_results_file_path, random_state=args.random_state)


    print(train_metrics)

    # Get test metrics

    test_metrics = test_recourse(dataset=args.dataset_name, file_path=test_file_path, sens_attr=args.sens_attr, ml_model=ml_model,
                            recourse_method=args.test_recourse_method, recourse_hyperparam=recourse_hyperparam_test, 
                            results_file=test_results_file_path, random_state=args.random_state)

    print(test_metrics)

# Plot the metrics

# plt.figure(figsize=(16, 16))

# # Subplot 1: Accuracy
# plt.subplot(5, 1, 1)
# plt.plot(train_metrics["epoch"], train_metrics["accuracy"], label="Overall Accuracy", color="black", linewidth=2)
# plt.plot(train_metrics["epoch"], train_metrics["acc_group_0"], label="Group 0 Accuracy", linestyle="--")
# plt.plot(train_metrics["epoch"], train_metrics["acc_group_1"], label="Group 1 Accuracy", linestyle="--")
# plt.axvline(pretrain_epoch, color='gray', linestyle=':', linewidth=2, label="Pretraining End")
# plt.title("Training Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)

# # Subplot 2: TPR and FPR per group
# plt.subplot(5, 1, 2)
# plt.plot(train_metrics["epoch"], train_metrics["tpr_group_0"], label="TPR Group 0", linestyle="-", color="blue")
# plt.plot(train_metrics["epoch"], train_metrics["tpr_group_1"], label="TPR Group 1", linestyle="-", color="cyan")
# plt.plot(train_metrics["epoch"], train_metrics["fpr_group_0"], label="FPR Group 0", linestyle="--", color="red")
# plt.plot(train_metrics["epoch"], train_metrics["fpr_group_1"], label="FPR Group 1", linestyle="--", color="orange")
# plt.axvline(pretrain_epoch, color='gray', linestyle=':', linewidth=2, label="Pretraining End")
# plt.title("TPR and FPR per Group")
# plt.xlabel("Epoch")
# plt.ylabel("Rate")
# plt.legend()
# plt.grid(True)

# # Subplot 3: Social Burden
# plt.subplot(5, 1, 3)
# plt.plot(train_metrics["epoch"], train_metrics["burden_group_0"], label="Burden Group 0", color="green")
# plt.plot(train_metrics["epoch"], train_metrics["burden_group_1"], label="Burden Group 1", color="purple")
# plt.axvline(pretrain_epoch, color='gray', linestyle=':', linewidth=2, label="Pretraining End")
# plt.title("Social Burden per Group")
# plt.xlabel("Epoch")
# plt.ylabel("Burden")
# plt.legend()
# plt.grid(True)

# # Subplot 2: TPR and FPR per group
# plt.subplot(5, 1, 4)
# plt.plot(train_metrics["epoch"], train_metrics["ar_group_0"], label="AR Group 0", linestyle="-", color="blue")
# plt.plot(train_metrics["epoch"], train_metrics["ar_group_1"], label="AR Group 1", linestyle="-", color="cyan")
# plt.axvline(pretrain_epoch, color='gray', linestyle=':', linewidth=2, label="Pretraining End")
# plt.title("AR per Group")
# plt.xlabel("Epoch")
# plt.ylabel("Rate")
# plt.legend()
# plt.grid(True)

# # Subplot 3: Social Burden
# plt.subplot(5, 1, 5)
# plt.plot(train_metrics["epoch"], train_metrics["cost_group_0"], label="Cost Group 0", color="green")
# plt.plot(train_metrics["epoch"], train_metrics["cost_group_1"], label="Cost Group 1", color="purple")
# plt.axvline(pretrain_epoch, color='gray', linestyle=':', linewidth=2, label="Pretraining End")
# plt.title("Cost per Group")
# plt.xlabel("Epoch")
# plt.ylabel("Burden")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()

# plt.savefig(f"result_figures/{fair_strategy}_{recourse_method}_pretrain{pretrain_epoch}_total{total_epoch}_sens{sens_attr}_rs{random_state}_training_metrics_plot.png", dpi=300, bbox_inches='tight')
# plt.show()


# Get test metrics




