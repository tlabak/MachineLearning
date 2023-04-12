# CS349 HW 4: Naive Bayes and the EM Algorithm

This assignment is due on March 9. There are two points of extra
credit for passing the `test_setup` test case, due *early* on March 2.

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

## What's changed since HW3?

- Nothing much: the autograder should work the same as with HW3.
- Whereas HW3 used the MNIST dataset, this assignment uses a `speeches.zip`
  dataset. The first time you try to run (certain) tests, the code will
  try to unzip that file to create a folder of text data.

## Important instructions

As before, your work will be graded and aggregated using an autograder that
will download the code and free response questions from each student's
repository. If you don't follow the instructions, you run the risk of getting
*zero points*. The `test_setup` test case gives you extra credit for following
these instructions and will make it possible to grade your work easily.

## Environment setup

You should be able to use the same `cs349` environment that you've used so far.
If you deleted that environment, please refer to the HW1 readme to recreate it.

## What to do for this assignment

The detailed instructions for the work you need to do are in `problems.md`.
You will also find it very helpful to read included `naive_bayes.pdf` writeup.

For the coding portion of the assignment, you will:
- Solve some simple practice problems with sparse matrices
- Write a stable softmax and log sum functions
- Implement a fully-supervised NaiveBayes classifier
- Implement a semi-supervised NaiveBayes classifier

You will also write up answers to the free response questions.

In every function where you need to write code, there is a `raise
NotImplementedError` in the code. The test cases will guide you through the work
you need to do and tell you how many points you've earned. The test cases can
be run from the root directory of this repository with:

``python -m pytest``

To run a single test, you can call e.g., `python -m pytest -s -k test_setup`.
The `-s` means that any print statements you include will in fact be printed;
the default behavior (`python -m pytest`) will suppress everything but the
pytest output.

We will use these test cases to grade your work! Even if you change the test
cases such that you pass the tests on your computer, we're still going to use
the original test cases to grade your assignment.

## Questions? Problems? Issues?

Simply post on Piazza, and we'll get back to you.
