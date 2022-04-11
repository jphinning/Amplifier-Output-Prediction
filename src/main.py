from lib.model_training import OutputPredictor as op


def main():
    test = op().coef_matrix
    print(test)


if __name__ == "__main__":
    main()
