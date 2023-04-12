# Decision Trees and Foundations
Decision trees are a type of supervised learning algorithm used in machine learning that can be used for both regression and classification tasks. They are constructed by recursively splitting the data into subsets based on the most significant attributes that contribute to the decision-making process.

The tree structure represents a hierarchy of decisions that lead to the final decision, where each internal node of the tree corresponds to a decision based on a specific feature, and each leaf node corresponds to a predicted outcome.

Decision trees are used in machine learning for a variety of applications, including but not limited to fraud detection, customer segmentation, and medical diagnosis. They are popular because they are easy to interpret and explain, and can be visualized for easier understanding. They are also able to handle both categorical and numerical data, and can handle missing values.

## For Running the Code
``python -m pytest -s``

To run a single test, you can specify it with `-k`, e.g., `python -m pytest -s
-k test_setup_netid`.  To run a group of tests, you can use `-k` with a prefix, e.g.,
`python -m pytest -s -k test_decision` will run all tests that begin with
`test_decision`.  The `-s` means that any print statements you include will in
fact be printed; the default behavior (`python -m pytest`) will suppress
everything but the pytest output.

## Coding
- Implemented a train test split and cross validation
- Implemented some classification metrics
- Implemented a simple model that just predicts the mode (most common class)
- Computed information gain
- Implemented a decision tree with the ID3 algorithm

## Material
- [Let’s Write a Decision Tree Classifier from Scratch - Machine Learning Recipes #8](https://www.youtube.com/watch?v=LDRbO9a6XPU)

- [Decision Tree Lecture Series](https://www.youtube.com/playlist?list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO)
  1. [How it works](https://www.youtube.com/watch?v=eKD5gxPPeY0&list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO&index=1)
  2. [ID3 Algorithm](https://www.youtube.com/watch?v=_XhOdSLlE5c&list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO&index=2)
  3. [Which attribute to split on](https://www.youtube.com/watch?v=AmCV4g7_-QM&list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO&index=3)
  4. [Information Gain and Entropy](https://www.youtube.com/watch?v=nodQ2s0CUbI&list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO&index=4)

- [ID3-Algorithm: Explanation](https://www.youtube.com/watch?v=UdTKxGQvYdc)

### Entropy
- [What is entropy in Data Science (very nice explanaton)](https://www.youtube.com/watch?v=IPkRVpXtbdY)
- [Entropy as concept in Physics/Chemistry (only if you're interested)](https://www.youtube.com/watch?v=YM-uykVfq_E)

### Recursion
- [Python: Recursion Explained](https://www.youtube.com/watch?v=wMNrSM5RFMc)
- [Recursion example](https://www.youtube.com/watch?v=8lhxIOAfDss)
