from lib.model_training import OutputPredictor as op


def main():
    test = op(1).out_extraction_md
    print(test)


if __name__ == "__main__":
    main()
