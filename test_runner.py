from plasma.training.runner import create


if __name__ == '__main__':
    runner = create("train_examples/train.json")
    runner.run()
