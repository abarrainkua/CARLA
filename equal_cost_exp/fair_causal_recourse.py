import loadModel

clf, X_test, y_test = loadModel.loadModelForDataset('mlp', 'adult', scm_class = None, num_train_samples = int(1e5), fair_nodes = ['x1', 'x2', 'x3'], fair_kernel_type = 'rbf', experiment_folder_name = None)

print(X_test)
print(type(X_test))

print(y_test)
print(type(y_test))