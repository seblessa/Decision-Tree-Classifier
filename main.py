from argparse import ArgumentParser, SUPPRESS
from sys import argv
import pandas as pd
from DecisionTree import DecisionTree


def help_menu():
    print('\nPlease pass as an argument one of the following dataset:')
    print("- iris")
    print("- restaurant")
    print("- weather")
    print("- connect4\n")

    print("If you want to add a new dataset in order to test it, please add it to the datasets folder"
          " and run the program again passing it as an argument normally.")
    print("Example:\n")

    print("python3 main.py [-tr,--train] 'dataset'\n")

    print("If you want to test the tree trained before, please run the program like this:\n")

    print("python3 main.py [-tr,--train] 'dataset' [-t,--test] 'test_dataset'\n")

    exit(0)


if __name__ == '__main__':
    if len(argv) == 2:
        if argv[1] == '-h' or argv[1] == '--help':
            help_menu()
            exit()

    parser = ArgumentParser(add_help=False, description='Decision Tree Classifier')

    parser.add_argument('-h', '--help', default=SUPPRESS, help='Show the help menu')
    parser.add_argument('-tr', '--train', default=SUPPRESS, help='Dataset to be used to train the Decision Tree')
    parser.add_argument('-t', '--test', default=SUPPRESS, help='Dataset to be used to test the Decision Tree')

    args = parser.parse_args()

    if len(argv) == 1:
        parser.print_help()
        exit()

    try:
        if args.train is not None:
            dataset_path = "datasets/" + str(args.train) + ".csv"
            try:
                df = pd.read_csv(dataset_path, na_values=['NaN'], keep_default_na=False)
            except FileNotFoundError:
                print("The dataset '" + str(args.train) + "' was not found.")
                exit()
            else:
                print("Decision Tree Classifier of the '" + str(args.train) + "' dataset:\n")

                tree = DecisionTree(df)

                # print(df)
                print(tree)

            try:
                if args.test is not None:
                    dataset_path = "datasets/" + str(args.test) + ".csv"
                    try:
                        test = pd.read_csv(dataset_path, na_values=['NaN'], keep_default_na=False)
                    except FileNotFoundError:
                        print("The dataset '" + str(args.test) + "' was not found.")
                    else:
                        print("Predicted values of '" + str(args.test) + "' dataset:\n")

                        print(tree.predict(test))
            except AttributeError:
                pass
    except AttributeError:
        print("\nPlease pass a dataset to train the Decision Tree Classifier.")
        print("You can see included datasets by passing '-h' or '--help' as an argument.")
