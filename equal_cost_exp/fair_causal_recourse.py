import loadModel
import loadData
import loadSCM

fair_nodes = ['x1', 'x2', 'x3']

clf, X_test, y_test = loadModel.loadModelForDataset('mlp', 'adult', scm_class = None, num_train_samples = int(1e5), fair_nodes = fair_nodes, fair_kernel_type = 'rbf', experiment_folder_name = None)

print(X_test)
print(type(X_test))

print(y_test)
print(type(y_test))


# Run the Fair Recourse Experiment

sensitive_attribute_nodes = fair_nodes

scm_obj = loadSCM.loadSCM(args, experiment_folder_name)
dataset_obj = loadData.loadDataset(args, experiment_folder_name)

for node in sensitive_attribute_nodes:
    assert \
      set(np.unique(np.array(dataset_obj.data_frame_kurz[node]))) == set(np.array((-1,1))), \
      f'Sensitive attribute must be +1/-1 .'


# TODO: find the sensitive attribute get the test metrics on cost, burden, accuracy, etc. and make plots