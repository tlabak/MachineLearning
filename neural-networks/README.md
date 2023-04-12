# CS349 HW 3: Perceptron, Neural Networks, and Regularization

This assignment is due Thursday, February 23 at 11:59pm CST. There are two
points possible for passing the `test_setup` test case, due *early* on
Thursday, February 16 at 11:59pm CST.  Late work will not be accepted except in
extreme circumstances.

## Academic integrity

Your work must be your own. You may not work with others. Do not submit other
people's work as your own, and do not allow others to submit your work as
theirs. You may *talk* with other students about the concepts covered by the
homework, but you may not share code or answers with them in any way. If you
have a question about an error message or about why a numpy function returns
what it does, post it on Piazza. If you need help debugging your code, make a
*private* post on Piazza or come to office hours.

We will use a combination of automated and manual methods for comparing your
code and free-response answers to that of other students. If we find
sufficiently suspicious similarities between your answers and those of another
student, you will both be reported for a suspected violation. If you're unsure
of the academic integrity policies, ask for help; we can help you avoid
breaking the rules, but we can't un-report a suspected violation.

By pushing your code to GitHub, you agree to these rules, and understand that
there may be severe consequences for violating them. 

## What's changed since HW2?

- The coding and free-response portions are still worth 10 points.
  Some FRQs rely on your code working before you can answer them.
- You can reuse your conda environment from HW1.

## Important instructions
As before, your work will be graded and aggregated using an autograder that
will download the code and free response questions from each student's
repository. If you don't follow the instructions, you run the risk of getting
*zero points*. The `test_setup` test case gives you extra credit for following
these instructions and will make it possible to grade your work easily.

Note: *do not* include your name or Net ID inside of your free-response PDFs.

## Environment setup

You should be able to use the same `cs349` environment that you used for the
past two assignments.  If you deleted that environment, please refer to the HW1
readme to recreate it.

## What to do for this assignment

The detailed instructions for the work you need to do are in `problems.md`.

For the coding portion of the assignment, you will:
- Implement the Perceptron classifier
- Write the `forward`, `backward`, and `fit` functions to enable training
  a MLP written only in `numpy`.
- Implement squared error loss function
- Implement the [ReLU activation
function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- Implement regularization for the MLP's weights
- Create a feature transformation that allows a linear model to classify
  a challenging spiral dataset.

You will also write up answers to the five free response questions.

In every function where you need to write code, there is a `raise
NotImplementeError` in the code. You will replace that line with code that
completes what the function docstring asks you to do.  The test cases will
guide you through the work you need to do and tell you how many points you've
earned. The test cases can be run from the root directory of this repository
with:

``python -m pytest -s``

To run a single test, you can specify it with `-k`, e.g., `python -m pytest -s
-k test_setup`.  To run a group of tests, you can use `-k` with a prefix, e.g.,
`python -m pytest -s -k test_model` will run all tests that begin with
`test_model`.  The `-s` means that any print statements you include will in
fact be printed; the default behavior (`python -m pytest`) will suppress
everything but the pytest output.

We will use these test cases to grade your work! Even if you change the test
cases such that you pass the tests on your computer, we're still going to use
the original test cases to grade your assignment.

## Questions? Problems? Issues?

Simply post on Piazza and we'll get back to you.
