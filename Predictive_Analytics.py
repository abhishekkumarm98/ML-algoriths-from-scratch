# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
"""
Predicitve_Analytics.py
"""


def Accuracy(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    return (y_true == y_pred).sum() / y_true.shape[0]


def Recall(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    confusion_matrix = ConfusionMatrix(y_true=y_true, y_pred=y_pred)
    correct = confusion_matrix.diagonal()
    true_examples_per_class = confusion_matrix.sum(axis=0)
    # Preventing divide by zero errors in the below line.
    true_examples_per_class[true_examples_per_class == 0] = 1
    recalls = correct / true_examples_per_class
    macro_recall = recalls.mean()
    return macro_recall


def Precision(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    confusion_matrix = ConfusionMatrix(y_true=y_true, y_pred=y_pred)
    correct = confusion_matrix.diagonal()
    predicted_per_class = confusion_matrix.sum(axis=1)
    # Preventing divide by zero errors in the below line.
    predicted_per_class[predicted_per_class == 0] = 1
    precisions = correct / predicted_per_class
    macro_precision = precisions.mean()
    return macro_precision


def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    ec_dist = 0
    for i, j in enumerate(Clusters):
        c, records = Clusters[i]
        ec_dist += np.sum(np.square(records - c))

    return float(ec_dist)


def ConfusionMatrix(y_true, y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: numpy.ndarray
    """
    # We assume that classes start at zero. Therefore we need to add 1 to our maximum value calculations.
    num_classes = max(y_true.max() + 1, y_pred.max() + 1)
    indices = num_classes * y_pred
    indices = y_true + indices
    results_array = np.zeros((num_classes ** 2,))
    indices, counts = np.unique(indices, return_counts=True)
    for index, count in zip(indices, counts):
        results_array[index] = count
    return results_array.reshape((num_classes, num_classes))


def KNN(X_train, X_test, Y_train, K):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    :type K: int

    @author Jacob Ekstrum
    :rtype: numpy.ndarray
    """

    # We leverage broadcasting to find the difference between all train/test pairs.
    assert len(X_train.shape) == 2, "Excepted an n by m matrix for training data."
    assert X_train.shape[1] == X_test.shape[1], "Training and testing data had different feature counts."
    x_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    x_test = X_test.reshape((1, X_test.shape[0], X_test.shape[1]))

    # We leverage the kernel trick here.
    # Note that we want to compute euclidean distance, that is, d[i, j] = (sum_over_f((trn[i] - tst[j]) ^ 2) ^ 1/2
    # Instead of computing pairwise differences, we can use the kernel trick by reducing the above to:
    # d[i, j] = (sum_over_f(trn[i] ^ 2 + tst[j] ^ 2 - 2*trn[i]*tst[j])) ^ 1/2
    # Because of the linearity of summation, we push the summation over the feature axis further into the formula.
    # Thus, we only need to compute the three quantities below:
    train_squared = (x_train ** 2).sum(axis=-1)
    test_squared = (x_test ** 2).sum(axis=-1)
    train_test = np.matmul(X_train, X_test.T)

    # The euclidean distances between all train and test examples are therefore listed below.
    difference_matrix = (train_squared + test_squared - 2 * train_test) ** 0.5

    # Our distance matrix has values representing distances, rows representing training examples,
    # and columns representing testing examples.
    # Each testing example in the k-nn case is then classified based on the indices of the distances.
    # We use numpy's argsort to find the indices in the training array of the smallest distances.
    nearest_neighbors = np.argsort(difference_matrix, axis=0)[:K, :]

    # We then index into the training labels in order to find out what the results should be.
    results = Y_train.flatten()[nearest_neighbors]

    # The result is the mode of our results. This is calculated by duplicating the voting classifier code from below.
    all_results = results

    # Voting matrix has rows representing indices and columns representing classes.
    voting_matrix = np.zeros((all_results.shape[1], all_results.max() - all_results.min() + 1), dtype=int)
    for class_index in range(all_results.min(), all_results.max() + 1):
        for neighbor_results in results:
            # For each neighbor, we find the values where the classifier predicted this class.
            # That counts as one vote and is added to the matrix.
            voting_matrix[:, class_index - all_results.min()] = voting_matrix[:, class_index - all_results.min()] + (
                    neighbor_results == class_index
            ).astype(int)

    # We can now simply find the results of the k-nn classifier by finding the argmax.
    results = voting_matrix.argmax(axis=-1)
    # Lastly, because our indices start at <min_value>, we need to add the min value back to the array
    # in order to have the classes align with the classes that were originally presented to us.
    results += all_results.min()

    return results


def RandomForest(X_train, Y_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    X_train = np.array(X_train)
    y_train = np.array(Y_train)
    X_test = np.array(X_test)

    # Encoding y_train set
    cls = np.unique(y_train)
    y_mod_train = np.zeros(y_train.shape[0])
    cls_mod = dict(zip(cls, list(range(len(cls)))))
    for i in cls:
        idx = np.where(y_train == i)
        y_mod_train[idx] = cls_mod[i]

    # It helps to calculate gini impurity of a node.

    def gini_impurity(records):

        target_var, target_counts = np.unique(records[:, -1], return_counts=True)
        probability = target_counts / target_counts.sum()

        return 1 - sum(np.square(probability))

    # I have used cumulative gini or or kind of gain using gini impurity to determine best split condition on an attribute among other attributes.
    # Here I have computed gain considering child impurity and their records and have ignored parent impurity because it will be
    # constant for all attributes.
    # It returns the best feature and its index by calculating gain, which helps to find the best split.

    def gain(records, splits):
        gini = np.inf
        for idx in splits:
            for val in splits[idx]:
                records_left, records_right = dataSplitting_along_a_Feature(records, idx, val)
                prob_records_left = records_left.shape[0] / ((records_right.shape[0] + records_left.shape[0]))
                prob_records_right = records_right.shape[0] / ((records_right.shape[0] + records_left.shape[0]))
                delta = (prob_records_left * gini_impurity(records_left)) + (
                            prob_records_right * gini_impurity(records_right))

                if delta <= gini:
                    gini = delta
                    idx_best, val_best = idx, val

        return idx_best, val_best

    # It checks whether distributuion is mixed or not.
    def isMixing(records):
        if np.unique(records[:, -1]).shape[0] == 1:
            return True
        else:
            return False

    # It returns a class to which a record belongs to.
    def getClass(records):
        target_var, target_counts = np.unique(records[:, -1], return_counts=True)
        return target_var[target_counts.argmax()]

    # Splitting of attributes are done by sorting as we have continuous feature points and then split
    # positions are taken as the midpoint of two adjacent values.

    def getSplits(records, numOfFeatures):

        if (records.shape[1] - 1) >= numOfFeatures:
            featureIndexes = np.random.randint(0, records.shape[1] - 1, numOfFeatures)

        splits = {}
        for idx in featureIndexes:
            sorted_records = np.sort(records[:, idx])
            splits[idx] = ((sorted_records + np.append(sorted_records[1:], 0)) / 2)[:-1]

        return splits

    # It splits the records on the basis of condition like greater than or less than the specified value.
    def dataSplitting_along_a_Feature(records, feature_idx, feature_val):
        records_left = records[records[:, feature_idx] > feature_val]
        records_right = records[records[:, feature_idx] <= feature_val]
        return records_left, records_right

    # Decision Tree(dt)
    def dt(dataset, count=0, min_samples=2, max_depth=5, numOfFeatures=None):

        if (isMixing(dataset)) or (dataset.shape[0] < min_samples) or (count == max_depth):
            target_var = getClass(dataset)
            return target_var

        else:
            count += 1
            splits = getSplits(dataset, numOfFeatures)
            idx_best, val_best = gain(dataset, splits)
            dataset_left, dataset_right = dataSplitting_along_a_Feature(dataset, idx_best, val_best)

            # It checks whether purity is achieved or not.
            if (dataset_left.shape[0] == 0) or (dataset_right.shape[0] == 0):
                target_var = getClass(dataset)
                return target_var

            # Finding the condition or question.
            condition = "{} <= {}".format(idx_best, val_best)

            # Making instances of sub-tree
            sub_tree = {condition: []}

            # It fetches answers true and false based on conditions.
            trueAnswer = dt(dataset_right, count, min_samples, max_depth, numOfFeatures)
            falseAnswer = dt(dataset_left, count, min_samples, max_depth, numOfFeatures)

            if trueAnswer == falseAnswer:
                sub_tree = trueAnswer
            else:
                sub_tree[condition].append(trueAnswer)
                sub_tree[condition].append(falseAnswer)

            return sub_tree

    # Random Forest(rf)

    def rf(training_set, numOftrees, rswr, numOfFeatures, max_depth):
        # Trees in forest (tif)
        tif = []
        for i in range(numOftrees):
            # Random Sampling With replacement(rswr) or bootstrap
            rswr_indexes = np.random.randint(low=0, high=training_set.shape[0], size=rswr)
            tree = dt(training_set[rswr_indexes], max_depth=max_depth, numOfFeatures=numOfFeatures)
            tif.append(tree)

        return tif

    def bagging(test_set, rf_obj):
        predictions = {}
        for i in range(len(rf_obj)):
            feature_name = "tree_{}".format(i)
            test_set = pd.DataFrame(test_set)

            def compute(instance, tree):

                condition = list(tree.keys())[0]
                feature_name, comparison_operator, value = condition.split(" ")
                if comparison_operator == "<=":
                    if instance[int(feature_name)] <= float(value):
                        answer = tree[condition][0]
                    else:
                        answer = tree[condition][1]

                if not isinstance(answer, dict):
                    return answer

                else:
                    leftover_tree = answer
                    return compute(instance, leftover_tree)

            predictions[feature_name] = test_set.apply(compute, args=(rf_obj[i],), axis=1)

        rf_predictions = pd.DataFrame(predictions).mode(axis=1)[0]

        return rf_predictions

    rf_obj = rf(np.concatenate([X_train, y_mod_train.reshape(-1, 1)], axis=1), numOftrees=4, rswr=800,
                numOfFeatures=int(np.sqrt(X_train.shape[1])), max_depth=8)
    predictions = bagging(X_test, rf_obj)
    return predictions.values


def PCA(X_train, N):
    """
    :type X_train: numpy.ndarray
    :type N: int

    @author Jacob Ekstrum
    :rtype: numpy.ndarray
    """
    column_means = X_train.mean(axis=0, keepdims=True)
    normalized_data = X_train - column_means
    covariance_matrix = np.cov(normalized_data.T)
    # For our purposes, we don't need V from the decomposition. We thus discard it.
    mapping_matrix, scaling_factors, _ = np.linalg.svd(covariance_matrix, full_matrices=True)
    # While the scaling values and mapping matrix aren't *exactly* eigenvalues and eigenvectors,
    # for our purposes they're a close substitute. We rename them to make the later computations more interpretable.
    eigenvalues = scaling_factors
    eigenvectors = mapping_matrix
    column_vector_indices = np.argsort(eigenvalues)[::-1][:N]
    reduced_dataset = np.matmul(normalized_data, eigenvectors[:, column_vector_indices])
    return reduced_dataset


def Kmeans(X_train, N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    X_train = np.array(X_train)
    centroids = np.random.randint(0, X_train.shape[0], N)
    c_old = X_train[centroids].astype(float)

    arr_ = np.zeros(X_train.shape[0])
    # Iterating centroids
    for i in c_old:
        arr_ = np.c_[arr_, np.sum(np.square(X_train - i), axis=1)]
    idx_to_c = np.argmin(arr_[:, 1:], axis=1)  # Ignoring first column because it contains zeros
    data = np.c_[X_train, idx_to_c]

    # Group by on labels using numpy
    c_new = np.zeros((N, X_train.shape[1]))
    for j, i in enumerate(np.unique(data[:, -1])):
        tmp = data[np.where(data[:, -1] == i)]
        c_new[j] = np.mean(tmp[:, :-1], axis=0)

    while True:
        if c_new.tolist() == c_old.tolist():
            cs = []

            for j, i in enumerate(c_new):
                cs.append(np.array([i, data[:, :-1][np.where(data[:, -1] == j)[0]]]))
            return cs
        else:
            arr_ = np.zeros(X_train.shape[0])
            for i in c_new:
                arr_ = np.c_[arr_, np.sum(np.square(X_train - i), axis=1)]
            idx_to_c = np.argmin(arr_[:, 1:], axis=1)
            data = np.c_[X_train, idx_to_c]
            c_old = c_new

            # Group by on labels using numpy
            c_new = np.zeros((N, X_train.shape[1]))  # wherever we see -1 that means we are ignoring label
            for j, i in enumerate(np.unique(data[:, -1])):
                tmp = data[np.where(data[:, -1] == i)]
                c_new[j] = np.mean(tmp[:, :-1], axis=0)


def SklearnSupervisedLearning(X_train, Y_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    # TODO: Must choose the hyperparameters that produce the best results based on grid search.
    knn = KNeighborsClassifier()
    svm = SVC()
    logistic_regression = LogisticRegression()
    decision_tree = DecisionTreeClassifier()

    algorithms = [svm, logistic_regression, decision_tree, knn]
    results = []
    for algorithm in algorithms:

        algorithm = algorithm.fit(X=X_train, y=Y_train)
        results.append(algorithm.predict(X_test))

    return results


def SklearnVotingClassifier(X_train, Y_train, X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    classifier_results = SklearnSupervisedLearning(X_train=X_train, Y_train=Y_train, X_test=X_test)
    classifier_results = [
        classifier_result.squeeze(axis=1) if len(classifier_result.shape) > 1 else classifier_result
        for classifier_result in classifier_results
    ]
    classifier_results = [
        classifier_result.argmax(axis=1) if len(classifier_result.shape) > 1 else classifier_result
        for classifier_result in classifier_results
    ]
    all_results = np.array(classifier_results)

    # Voting matrix has rows representing indices and columns representing classes.
    voting_matrix = np.zeros((all_results.shape[1], all_results.max() - all_results.min() + 1), dtype=int)
    for class_index in range(all_results.min(), all_results.max() + 1):
        for voting_classifier_results in classifier_results:
            # For each voting classifier, we find the values where the classifier predicted this class.
            # That counts as one vote and is added to the matrix.
            voting_matrix[:, class_index - all_results.min()] = voting_matrix[:, class_index - all_results.min()] + (
                    voting_classifier_results == class_index
            ).astype(int)

    # We can now simply find the results of the voting classifier by finding the argmax.
    results = voting_matrix.argmax(axis=-1)
    # Lastly, because our indices start at <min_value>, we need to add the min value back to the array
    # in order to have the classes align with the classes that were originally presented to us.
    results += all_results.min()
    return results


"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""


def ekstrum_plot_results(title, experiment_results, result_metric_name='Accuracy'):
    keys_to_plot = set()
    for result_set in experiment_results:
        for key_to_plot in result_set.keys():
            keys_to_plot.add(key_to_plot)
    if 'results' in keys_to_plot:
        keys_to_plot.remove('results')
    for key_to_plot in keys_to_plot:
        independent_variables = [result_set[key_to_plot] for result_set in experiment_results]
        results = [result_set['results'] for result_set in experiment_results]
        if all(isinstance(independent_variable, (float, int)) for independent_variable in independent_variables):
            # We can do a simple plot since all values are numeric.
            plt.clf()
            plt.scatter(independent_variables, results)
            plt.xlabel(key_to_plot)
            plt.ylabel(result_metric_name)
            plt.title(title)
            plt.savefig('plot_{}_{}.png'.format(title, key_to_plot))
        else:
            variables_to_plot = list(set(independent_variables))
            # Construct one dependent variable collection for each independent variable choice.
            dependent_variables_to_plot = [[] for _ in range(len(variables_to_plot))]
            for independent_variable, response in zip(independent_variables, results):
                # Add our response value to the correct bucket.
                dependent_variables_to_plot[variables_to_plot.index(independent_variable)].append(response)
            plt.clf()
            plt.boxplot(dependent_variables_to_plot)
            plt.xlabel(key_to_plot)
            plt.xticks(1 + np.arange(len(variables_to_plot)), variables_to_plot)
            plt.ylabel(result_metric_name)
            plt.title(title)
            plt.savefig('plot_{}_{}.png'.format(title, key_to_plot))


def ekstrum_plot_confusion_matrix(confusion_matrix, filename='confusion_matrix.png', model_name='our Model'):
    confusion_matrix = confusion_matrix.astype(int)
    nrows = confusion_matrix.shape[0]
    ncols = confusion_matrix.shape[1]
    true_values = confusion_matrix.sum(axis=0)
    # The below is done to prevent divide by zero errors.
    true_values[true_values == 0] = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows, ncols), sharex=True, sharey=True)
    axes[nrows // 2, 0].set_ylabel('Predicted Classes')
    axes[-1, ncols // 2].set_xlabel('True Classes')
    fig.suptitle('Confusion Matrix for {}'.format(model_name))
    for row in range(nrows):
        for column in range(ncols):
            axes[row, column].set_xticklabels([])
            axes[row, column].set_yticklabels([])
            # Color to hex is done via a nice little hack from https://stackoverflow.com/a/3380739.
            if row == column:
                # These should be "green" based on how good they were. White if completely wrong.
                accuracy_at_class = confusion_matrix[row, column] / true_values[column]
                rgb = (255 - int(255 * accuracy_at_class), 255, 255 - int(255 * accuracy_at_class))
                color = '#%02x%02x%02x' % rgb
            else:
                # These should be "red" based on how bad they were. White if completely correct.
                inaccuracy_at_class = confusion_matrix[row, column] / true_values[column]
                rgb = (255, 255 - int(255 * inaccuracy_at_class), 255 - int(255 * inaccuracy_at_class))
                color = '#%02x%02x%02x' % rgb

            axes[row, column].set_facecolor(color)
            axes[row, column].text(
                0.5, 0.5, '{}'.format(confusion_matrix[row, column]),
                horizontalalignment='center', verticalalignment='center', transform=axes[row, column].transAxes
            )

    plt.show()
    plt.savefig(filename)
    return fig


def ekstrum_grid_search(
        algorithm,
        algorithm_kwargs,
        algorithm_predict_kwargs,
        y_true,
        hyperparameter_bounds,
        continuous_partitions=10
):
    """
    Performs grid search on the specified algorithm.

    Hyperparameter bounds are provided via a mapping of the form below. Lists represent categorical hyperparameters,
    while tuples represent continuous hyperparameters. Continuous hyperparameters are segmented into
    <continuous_partitions> different values and then treated like categorical hyperparameters.

    An example for SVC:
    {
        "kernel_function": ['rbf', 'linear', 'poly', 'sigmoid'],
        "C": (0.1, 10.0),
        "degree": [2, 3, 4, 5]
    }

    :param algorithm: The algorithm to use
    :param algorithm_kwargs: Keyword arguments for the algorithm being analyzed. Used during call to `fit`.
    :param algorithm_predict_kwargs: Keyword arguments for algorithm when predicting.
    :param y_true: True values for prediction, to be used by the metric function.
    :param hyperparameter_bounds: The hyperparameter bounds for the algorithm, in the format described above.
    :param continuous_partitions: How many different combinations for continuous features should be tried.

    @author Jacob Ekstrum
    :return: A List of mappings specifying configurations and their performance.
    """
    for hyperparameter, values in hyperparameter_bounds.items():
        if isinstance(values, tuple):
            assert len(values) == 2, "Continuous tuples should be specified as (low, high)"
            try:
                float(values[0]), float(values[1])
            except ValueError:
                raise ValueError("Continuous tuples should be specified via numerical bounds.")
            interval_step_size = (values[1] - values[0]) / ((continuous_partitions - 1) or 1)
            values = [values[0] + interval_step_size * i for i in range(continuous_partitions)]
            hyperparameter_bounds[hyperparameter] = values

        assert isinstance(hyperparameter_bounds[hyperparameter], list)

    def run_from(hparams, selected):
        selection_copy = {k: v for k, v in selected.items()}
        if hparams:
            for hparam, values_ in hparams.items():
                smaller_hparams = {k: v for k, v in hparams.items() if k != hparam}
                results = []
                for value in values_:
                    selection_copy[hparam] = value
                    results = results + run_from(smaller_hparams, selection_copy)
                return results
            raise ValueError("Didn't have enough arguments in a hyperparameter for grid search.")

        model = algorithm(**selected)
        model.fit(**algorithm_kwargs)
        predictions = model.predict(**algorithm_predict_kwargs)
        # We use the accuracy metric, for now.
        selection_copy['results'] = Accuracy(y_true=y_true, y_pred=predictions)
        return [selection_copy]

    return run_from(hyperparameter_bounds, {})


def ekstrum_grid_search_svc(x_train, y_train, x_test, y_test):
    from sklearn.svm import SVC
    algorithm = SVC
    hyperparameters = {
        'C': (0.1, 10.0),
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': ['auto']
    }
    grid_search_results = ekstrum_grid_search(
        algorithm=algorithm,
        algorithm_kwargs={
            'X': x_train,
            'y': y_train
        },
        algorithm_predict_kwargs={
            'X': x_test
        },
        y_true=y_test,
        hyperparameter_bounds=hyperparameters,
        continuous_partitions=31
    )

    ekstrum_plot_results(title='Support Vector Machine Performance', experiment_results=grid_search_results)


def ekstrum_grid_search_tree(x_train, y_train, x_test, y_test):
    from sklearn.tree import DecisionTreeClassifier
    algorithm = DecisionTreeClassifier
    hyperparameters = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 1, 10],
        'max_features': ['auto', 'sqrt', 'log2', 0.1, 0.5, 1.0]
    }
    grid_search_results = ekstrum_grid_search(
        algorithm=algorithm,
        algorithm_kwargs={
            'X': x_train,
            'y': y_train
        },
        algorithm_predict_kwargs={
            'X': x_test
        },
        y_true=y_test,
        hyperparameter_bounds=hyperparameters,
        continuous_partitions=31
    )

    ekstrum_plot_results(title='Decision Tree Performance', experiment_results=grid_search_results)


def ekstrum_grid_search_knn(x_train, y_train, x_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    algorithm = KNeighborsClassifier
    hyperparameters = {
        'n_neighbors': [1, 2, 4, 8, 16, 32],
        'weights': ['distance', 'uniform'],
        'p': [1, 2, 4]
    }
    grid_search_results = ekstrum_grid_search(
        algorithm=algorithm,
        algorithm_kwargs={
            'X': x_train,
            'y': y_train
        },
        algorithm_predict_kwargs={
            'X': x_test
        },
        y_true=y_test,
        hyperparameter_bounds=hyperparameters,
        continuous_partitions=31
    )

    ekstrum_plot_results(title='KNN Performance', experiment_results=grid_search_results)
