from plasma.training.config_runner import ConfigRunner as Runner

if __name__ == '__main__':
    runner = Runner("train_examples/train.json")
    runner.run()
