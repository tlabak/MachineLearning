# CS349 HW 5: Reinforcement Learning and Fairness

This assignment is due March 16 at 11:59pm. There are two points of extra
credit for passing the `test_setup` test case, due *early* on March 13.

Because we are the end of the quarter, late work cannot be accepted.

## Important instructions

Your work will be graded and aggregated using the autograder. If you
don't follow the instructions, you run the risk of getting *zero points*. The
`test_setup` test case gives you extra credit for following these instructions 
and will make it possible to grade your work easily.

The essential instructions:
- Your code and written answers must be *pushed* to GitHub for us to grade them!
  We will only grade the latest version of your code that was pushed to GitHub
  before the deadline.
- Your NetID must be in the `netid` file; replace `NETID_GOES_HERE` with your
  netid.
- Your answer to each free response question should be in *its own PDF* with
  the filename `XXX_qYYY.pdf`, where `XXX` is your NetID and `YYY` is the question
  number. So if your NetID is `xyz0123`, your answer to free response question 2
  should be in a PDF file with the filename `xyz0123_q2.pdf`.
- Please do not put your name or NetID in your free response PDFs -- we will
  grade these anonymously. 

## Changes from previous homeworks

- There is a new package in `requirements.txt`: `gym`. Read [the documentation
  here](https://www.gymlibrary.dev).
- You *must* use python 3.9 instead of 3.10, to ensure compatibility with the
  `gym` package.

## Clone this repository

If you've forgotten how to use `git`,
check out this [helpful guide](https://guides.github.com/activities/hello-world/). 

As soon as you've cloned this repo, add your NetID to the `netid` file,
run `git add netid`, then `git commit -m "added netid"`, and `git push origin main`.
If you've successfully run those commands, you're almost done with the `test_setup`
test case.

## Environment setup

This homework requires `gym`, unlike previous homeworks. You need to use python
3.9 to work with `gym` version `0.26.2`. You can either install that into your
previous environment, or create a new environment:

- ``conda create -n cs349hw5 python=3.9``
- ``conda activate cs349hw5``
- ``pip install -r requirements.txt``

## What to do for this assignment

The detailed instructions for the work you need to do are in `problems.md`.
You will also find it very helpful to read pages 32 and 131 of
[Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf).

For the coding portion of the assignment, you will:
- Learn how to interface with OpenAI Gym environments
- Implement two Reinforcement Learning algorithms
- Investigate the performance of your algorithms and the effects of various hyperparameters
- Explore questions about fairness in real-world and toy examples

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
