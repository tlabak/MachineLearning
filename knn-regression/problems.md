## Instructions

The setup has an earlier deadline of January 31 at 11:59pm CDT.
The coding and free response questions are due on February 7 at 11:59pm.

### Setup (2 points)

All you need to do for these points is pass the `test_setup` case. This
requires putting your NetID in the `netid` file and creating five PDF files
titled `XXX_qYYY.pdf` where `XXX` is replaced with your NetID, and `YYY`
ranges from 1 to 5. The content of these PDFs won't be graded, this is just to
ensure that you can set up your repository to be autograded.

There is no `password` requirement for this assignment. Your final submission
must also pass the `test_setup` test, or you will lose these points.

### Coding (10 points)

You need to write code in every function in `src/` that raises a
`NotImplementedError` exception. Your code is graded automatically using the
test cases in `tests/`.  To see what your grade is going to be, you can run
`python -m pytest`; make sure you have installed the packages from
`requirements.txt` first. If the autograder says you get 100/100, it means you
get all 10 points.

The tests build on and sometimes depend on each other. We suggest that you
implement them in the order they appear in `tests/rubric.json`. That file also
allows you to see how many (relative) points each test is worth and which other
tests it may depend on. 

You may not use `sklearn` or `scipy` to implement the functions in this
assignment.  However, you are welcome to look at the documentation for the
[PolynomialFeatures](
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
and [LinearRegression](
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
classes.  Do not use the numpy functions `polynomial`, `polyfit` or `polyval`
for any of your solutions. Please do not use the python internal modules or
functions `importlib`, `getattr`, or `globals`. The `test_imports` case will
try to alert you if you use this disallowed packages or functions; please do
not try to circumvent these checks. If you think the test case is erroneously
penalizing you, please make a private Piazza post.
 
The grade given to you by the autograder on Canvas is the grade you should
expect receive. If something goes wrong (your code times out, you import a
disallowed package, you accidentally push a syntax error, etc.) and you need us
to grade your code manually, we will do so but subtract a 2 point penalty.
Please be careful and read the feedback that the autograder is giving you.

### Free response (10 points)

There are five free response questions. Your answer to each should be in its
own PDF file, titled `XXX_qYYY.pdf`, where `XXX` is replaced with your NetID
and `YYY` is the number of the question. So if your netid were `xyz0123`, the
answer to question 1 should be in a file named `xyz0123_q1.pdf`.  For questions
with multiple parts, put all parts in the single PDF and just clearly label
where each part begins.  Please *do not put your name in these PDFs* -- we will
grade your work anonymously using only your netid in the filenames.

## Free response questions

### Question 1 (3 points)

For this question, you will need to write some code and run some experiments
using the code in `free_response/q1.py`. Note that this file relies on your
`PolynomialRegression` and `KNearestNeighbor` code as well as some functions
from `sklearn`. The code will create some plots, some of which you will show in
part b., and you will need to create an addition plot for part a. You are
welcome to, **but not required**, to commit your code for this question to
GitHub; just make sure you do not add any `sklearn` imports into the files in
your `src/` directory.

First, add code in the two `*_regression_experiment()` functions to create a
plot for each of the 16 experiments. Use the `load_frq_data` and
`visualize_model` functions to produce those plots. You may want to add print
statements (for example, to record mean squared error results) to help with
part a.  If you run `python -m free_response.q1`, it should fill your
`free_response/` folder with 32 different plots. You do not need to push
these plots to GitHub!

- a. Look at the example plot created by the `part_a_plot` function in
  `free_response/q1.py`, which is saved as `free_response/demo_plot.png`.
  Rewrite this function so that instead of plotting random values, it plots the
  train and test MSE from the 32 plots that the `*_regression_experiment()`
  functions created. Create one set of four subplots for the KNN results with
  `K` on the X axis, and one set of four subplots for the PolynomialRegression
  results with `degree` on the X axis. In each of the subplots corresponding to
  the four different dataset sizes, plot the train and test MSE for the model
  as you increase `K` or `degree`.

  Include these two plots (not all 32 plots!) in your PDF for this question,
  and provide a one-sentence description for each. What do you notice about the
  relationship between the learned functions `h(X)` and the difference between
  train and test MSE? Does this match up with what we discussed in lecture?

- b. Look through the 32 plots created by the `*_regression_experiment()`
  functions in `free_response/q1.py`. Find one plot where the model (either
  PolynomialRegression or KNearestNeighbor) is clearly overfitting, and add it
  to your PDF. How can you tell that the model is overfitting? Then, find one
  plot where the model is clearly underfitting, and do the same. For each plot,
  include a one-sentence explanation.

- c. Assume that each dataset from this question is sampled from some true
  function `f(X)`, and that our model defines a hypothesis class `H` from which
  it tries to choose the best `h(X)` using the `n` training examples. In your
  own words, describe how overfitting and underfitting depend on `f(X)`, `H`,
  and `n`.  You are encouraged to use examples from the plots created by
  `free_response/q1.py` to support your answer.

### Question 2 (2 points)

For this question, take a look at the `KNearestNeighbor` class, the tests in
`tests/test_knn.py`, and refer back to the lecture slides on nearest neighbors.
The `aggregator` argument controls how the model aggregates the labels
across the nearest neighbors it has found. Thus if `k=3` and the three
nearest neighbors are `x1, x2, x3` with labels `y1, y2, y3`, the model
returns `aggregator(y1, y2, y3)`.

- a. For the problems we consider, why is it important that the `aggregator`
  argument can take two values: `mode` and `mean`? For what kinds of problems
  would we prefer each?
- b. Propose a third option for `aggregator` that is neither `mode` nor `mean`.
  Describe in detail how your aggregator function would work. For what kinds of
  problems would we prefer your option? What might be a drawback of your
  option? 

### Question 3 (2 points)

Suppose you have a simple dataset with two points: `[(x1=1, y1=1), (x2=2,
y2=2)]`. We have `x3=100`, and want to predict `y3`.

- a.  Let's say you fit your KNN model with k=1, euclidean distance, and
  aggregator="mean" to the dataset. What will it predict for `y3`? Why?
- b. Let's say you fit your linear regression model (i.e., degree=1) to this
  data. What will it predict for `y3`? Why?
- c. Based on your answers to a. and b., what is the *inductive bias* of both
  models? By inductive bias, we are referring to the assumptions are built into
  each model about how to generalize to new data that's unlike data it has seen
  before. How would you describe these assumptions for each model?

### Question 4 (1 point)

The [MovieLens dataset](https://grouplens.org/datasets/movielens/100k/) is a
dataset of 100,000 movie ratings, from which the provided `movielens` dataset
is sampled. In the data, the 1,000 users give ratings from 1 to 5 to movies
chosen from a list of 1,700 titles. In the data matrix (denoted `X`), if user `i`
rated movie `j`, then `X[i, j]` is that rating (an integer from 1 to 5). If
that user did not rate that movie, then `X[i, j] = 0`. Thus, most entries in
the dataset are 0, because most users only rate a small number of movies. 
You may assume that every user has rated at least one movie with each of the
five ratings; that is, in every row, there is at least one 1, one 2, one 3,
one 4, and one 5.

Suppose we wanted to use a K Nearest Neighbor model to recommend movies to
users using this data. That is, for a given user who has rated some movies, we
want to recommend a movie we *think* they will rate highly, but have not yet
seen.

- a. What would be a good distance measure for this data? Why?
- b. How does the distance measure you chose handle the fact that most entries in
the data matrix are 0?

### Question 5 (2 points)

Suppose we have a data `X_train` with shape `(n1, n_features)` and a data
matrix `X_test` with shape `(n2, n_features)`, each with a corresponding array
of labels. Imagine we fit a *k nearest neighbor regression* to `X_train` and
then use it to predict on `X_test`, also imagine we fit a *polynomial
regression* of degree `degree` to `X_train` and then use it to predict on
`X_test`.

For each question, compare the KNN and Polynomial Regression models. Provide a
detailed explanation that compares the models' behavior in terms of `n1`, `n2`,
and/or `n_features`.

- a. Which model takes longer to train on `X_train`? Why?
- b. Which model takes longer to predict on `X_test`? Why?
- c. Suppose you wanted to zip up your trained models and email them to your
  friend. Which model would require a larger filesize to store it? Why?
