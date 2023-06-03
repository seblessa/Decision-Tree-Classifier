from math import log2
from collections import Counter
import numpy as np


class DecisionTree:

    def __init__(self, df):
        """
        Function that initialises the decision tree.
        :param df: whole dataset.
        """
        df = self.transform_boolean_columns(df)
        possible_columns = df.columns.tolist()[1:-1]  # Exclude the first and last columns
        self.tree = self.build_tree(df, df, possible_columns)

    def predict(self, df):
        """
        Predicts the target value for each row in the provided DataFrame.
        :param df: dataFrame containing the input data.
        :return: list of predicted target values.
        """
        predictions = []
        for _, row in df.iterrows():
            prediction = self.traverse_tree(self.tree, row)
            if prediction is None:
                predictions.append(prediction)
                continue
            predictions.append(prediction[0])
        return predictions

    def traverse_tree(self, tree, row):
        """
        Traverses a decision tree recursively to predict the target value for the given input row.
        :param tree: decision tree represented as a nested dictionary.
        :param row: a row of the dataSet.
        :return: the predicted target value based on the decision tree.
                Returns none if the value is not present in the tree or an error occurs.
        """
        for attribute, subtree in tree.items():
            value = row[attribute]
            if isinstance(value,bool):
                value = str(value)
            if isinstance(subtree, dict):
                if isinstance(value, str) and value not in subtree:
                    return None  # Value not present in the tree, return None
                elif not isinstance(value, str):
                    # Numerical attribute handling
                    split_key = list(subtree.keys())[0]
                    split_operator, split_value = split_key.split(' ')

                    if split_operator == '<=':
                        try:
                            if float(value) <= float(split_value):
                                subtree = subtree['<= ' + split_value]
                            else:
                                subtree = subtree['> ' + split_value]
                        except ValueError:
                            return None  # Non-numeric value, skip the comparison
                    elif split_operator == '>':
                        try:
                            if float(value) > float(split_value):
                                subtree = subtree['> ' + split_value]
                            else:
                                subtree = subtree['<= ' + split_value]
                        except ValueError:
                            return None  # Non-numeric value, skip the comparison
                    else:
                        return None  # Invalid split operator
                else:
                    subtree = subtree[value]

                if isinstance(subtree, dict):
                    return self.traverse_tree(subtree, row)
                else:
                    return subtree
            else:
                return subtree

    def build_tree(self, df, data, attributes):
        """
        Recursively builds a decision tree based on the provided training data and attributes.
        :param df: dataFrame containing the training data
        :param data: subset of the training data for the current nome
        :param attributes: a list of attribute names available for splitting the data
        :return: a nested dictionary representing the decision tree
        """
        labels = data[data.columns[-1]].tolist()
        class_counts = Counter(labels)

        # Base cases
        if len(set(labels)) == 1:
            return [labels[0], class_counts[labels[0]]]  # Return [class_label, class_count]
        if len(attributes) == 0:
            return [class_counts.most_common(1)[0][0], len(labels)]  # Return [class_label, total_count]

        best_attribute = self.choose_best_attribute(data, attributes)
        node = {best_attribute: {}}

        attribute_values = df[best_attribute].unique()
        if data[best_attribute].dtype == 'int64' or data[best_attribute].dtype == 'float64':
            best_split_value = self.calculate_best_split_value(data, best_attribute)
            subset1 = data[data[best_attribute] <= best_split_value]
            subset2 = data[data[best_attribute] > best_split_value]
            remaining_attributes = attributes.copy()
            remaining_attributes.remove(best_attribute)
            node[best_attribute]['<= ' + str(best_split_value)] = self.build_tree(df, subset1, remaining_attributes)
            node[best_attribute]['> ' + str(best_split_value)] = self.build_tree(df, subset2, remaining_attributes)


        else:
            for value in attribute_values:
                subset = data[data[best_attribute] == value]
                if len(subset) == 0:
                    node[best_attribute][value] = [class_counts.most_common(1)[0][0], 0]
                else:
                    remaining_attributes = attributes.copy()
                    remaining_attributes.remove(best_attribute)
                    node[best_attribute][value] = self.build_tree(df, subset, remaining_attributes)

        return node

    @staticmethod
    def calculate_entropy(labels):
        """
        Calculates the entropy of a list of labels.
        :param labels: list of labels.
        :return: entropy value.
        """
        label_counts = Counter(labels)
        total_samples = len(labels)
        entropy = 0

        for count in label_counts.values():
            probability = count / total_samples
            entropy -= probability * log2(probability)

        return entropy

    def choose_best_attribute(self, data, attributes):
        """
        Selects the best attribute to split the data based on information gain.
        :param data: dataFrame containing data.
        :param attributes: list of attribute names available for splitting.
        :return: name of the best attribute.
        """
        entropy_s = self.calculate_entropy(data[data.columns[-1]].tolist())
        information_gains = []

        for attribute in attributes:
            entropy_attribute = self.calculate_attribute_entropy(data, attribute)
            information_gain = entropy_s - entropy_attribute
            information_gains.append(information_gain)

        best_attribute_index = information_gains.index(max(information_gains))
        return attributes[best_attribute_index]

    def calculate_attribute_entropy(self, data, attribute):
        """
        Calculates the entropy of an attribute in the data.
        :param data: dataFrame containing data.
        :param attribute: name of the attribute.
        :return: entropy value of the attribute.
        """
        attribute_values = data[attribute].unique()
        entropy_attribute = 0

        for value in attribute_values:
            subset = data[data[attribute] == value]
            subset_labels = subset[data.columns[-1]].tolist()
            subset_entropy = self.calculate_entropy(subset_labels)
            subset_probability = len(subset_labels) / len(data)
            entropy_attribute += subset_probability * subset_entropy

        return entropy_attribute

    def calculate_best_split_value(self, data, attribute):
        """
        Calculates the best split value for a numerical attribute based on the information gain.
        :param data: dataFrame containing data.
        :param attribute: name of the numerical attribute.
        :return: The best split value for the attribute.
        """
        attribute_values = data[attribute].unique()
        best_split_value = None
        best_information_gain = float('-inf')

        if len(attribute_values) == 1:
            # All instances have the same value for the attribute
            return attribute_values[0]

        for value in attribute_values:
            subset1 = data[data[attribute] <= value]
            subset2 = data[data[attribute] > value]

            labels1 = subset1[data.columns[-1]].tolist()
            labels2 = subset2[data.columns[-1]].tolist()

            information_gain = self.calculate_information_gain(data[data.columns[-1]].tolist(), labels1, labels2)

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_split_value = value

        return best_split_value

    def calculate_information_gain(self, parent_labels, labels1, labels2):
        """
        Calculates the information gain by splitting the parent labels into two subsets.
        :param parent_labels: list of labels from the parent node.
        :param labels1: list of labels from one subset.
        :param labels2: list of labels from the other subset.
        :return: the information gain.
        """
        parent_entropy = self.calculate_entropy(parent_labels)
        weight1 = len(labels1) / len(parent_labels)
        weight2 = len(labels2) / len(parent_labels)
        entropy1 = self.calculate_entropy(labels1)
        entropy2 = self.calculate_entropy(labels2)
        information_gain = parent_entropy - (weight1 * entropy1) - (weight2 * entropy2)
        return information_gain

    def __str__(self):
        """
        Return a string representation of the decision tree.
        :return: a string representation of the decision tree.
        """
        return self.tree_to_string(self.tree)

    def tree_to_string(self, t=None, indent=''):
        """
        Converts the decision tree to a string representation.
        :param t: the tree convert (default is the instance's tree).
        :param indent: the indentation string.
        :return: the string representation of the tree.
        """
        if t is None:
            t = self.tree

        result = ""
        if isinstance(t, dict):
            for key, value in t.items():
                if isinstance(value, dict):
                    result += f'{indent}{key}:\n'
                    result += self.tree_to_string(value, indent + '  ')
                else:
                    result += f'{indent}{key}: {value[0]}  ({value[1]})\n'
        return result

    @staticmethod
    def transform_boolean_columns(df):
        """
        Transforms boolean columns in a dataFrame to string type.
        :param df: the dataFame to transform.
        :return: the transformed dataFrame.
        """
        boolean_columns = df.select_dtypes(include=bool).columns

        for column in boolean_columns:
            df[column] = df[column].map({False: 'False', True: 'True'})

        return df

    @staticmethod
    def score(y_true, y_pred):
        """
        Computes the accuracy score between true labels and predicted labels.
        :param y_true: the tru labels.
        :param y_pred: the predicted labels.
        :return: the accuracy score.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                correct += 1
        return correct / len(y_true)
