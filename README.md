# machine-learning

> Machine Learning sandbox

This package uses fire to run command. 
If you have any troubles you could always put --help at the end
of a command and see if the man page reveals any information you are looking for.

## Classification

### Cats vs. Dogs (Microsft dataset)

Train a Convoluted neural network (tensorflow) to distinguish between cats and dogs.

```bash
python ml models classification animals train --project catsvsdogs
```

To have a look at the tensorboard

```bash
tensorboard --logdir ./artifacts/catsvsdogs/tb_logs
```

Use the previously trained model to classify the entire dataset and show the confusion matrix afterwards:

```bash
python ml models classification animals predict --project catsvsdogs
```

### Digits (MNist)

Train a neural network (tensorflow) to classify digital numbers.

```bash
python ml models classification digits train --project digits
```

To have a look at the tensorboard

```bash
tensorboard --logdir ./artifacts/digits/tb_logs
```

Use the previously trained model to classify the entire dataset and show the confusion matrix afterwards:

```bash
python ml models classification digits predict --project digits
```

## Reinforcement Learning

### Blackjack

Train a Q-Learning model to play blackjack!

First run baseline to determine how either fixed actions (always stick) or
random actions (random stick or draw a card) will perform:

```bash
# Fixed action
python ml models rl blackjack baseline --episodes=200000 --project=blackjack-baseline-fixed --mode=fixed

# Random action
python ml models rl blackjack baseline --episodes=200000 --project=blackjack-baseline-random --mode=random
```

After that you can evaluate if your Q-Learning is actually better than the baseline:

```bash
python ml models rl blackjack train --episodes=200000 --project=blackjack-qlearn --learning-rate=0.1 --gamma=0.95
```

### Mountain car

Train a Q-Learning model to solve the mountain car game and be the king of the hill!

To train the Q-Learning model:

```bash
python ml models rl mountain train --episodes=10000 --project=mountain-car
```

After that you can use the model to simulate the car to be king of the hill

```bash
python ml models rl mountain simulate ./artifacts/mountain-car/10000_qtable.pkl
```

## Text

### Generic text classifier

A generic classifier for text categorization.

To train it on the default dataset by using TF-IDF Features and the SGD Classifer:

```bash
python ml models text generic train --tfidf --sgd --project bbcnews
```

Now use the trained classifier to predict the categories for the whole dataset
and show the confusion matrix.

```bash
python ml models text generic predict --project bbcnews
```

You can ask the classifier itself for more help and how to enable / disable certain 
feature engineering techniques and algorithms.

```bash
python ml models text generic train --help
```