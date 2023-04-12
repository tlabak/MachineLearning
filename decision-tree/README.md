# HW 1: Decision Trees and Foundations

There are 22 points possible for this assignment. The setup is worth 2 points,
due on Jan 13 at 11:59pm CST. The coding and free response questions are worth
20 points, due on Jan 24 at 11:59pm CST. Late work will not be accepted except in
extreme circumstances.

## Walkthrough

For a walkthrough of how to get started with this assignment, [please watch this video](https://northwestern.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=14ecd5fd-dc61-4092-8530-af86012777e9).

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

## Important instructions
Your work will be graded and aggregated using an autograder that will download
the code and free response questions from each student's repository. If you
don't follow the instructions, you run the risk of getting *zero points*. The
`test_setup` test cases gives you points for following these instructions 
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
- Do not include your name or Net ID in the content of your free response PDFs
  -- we will grade these anonymously. 

## Clone this repository

First, you need to clone this repository. If you haven't used `git` before,
check out this [tutorial](https://guides.github.com/activities/hello-world/)
or this [guide to cloning repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

Once you have `git` installed on your computer, you'll need the link to this
repository (find it on the github.com under "Clone or Download"), which might
look something like `git@github.com:cs349/hw1-knn-decision-tree.git`. Then run:

``git clone YOUR-LINK``

As soon as you've downloaded it, go ahead and add your NetID to the `netid` file,
run `git add netid`, then `git commit -m "added netid"`, and `git push origin main`.
If you've successfully run those commands, you're almost done with the `test_setup`
test case.

## Environment setup

This course uses **Python 3**. Python 2 will not work for these assignments and
all assignments will be graded with Python 3 on our end.

If you used Python before, you've probably installed packages (e.g. `numpy`). To
avoid having to uninstall old packages in order to reinstall the correct ones,
you are strongly encouraged to create a "virtual environment" using 
[**miniconda**](https://docs.conda.io/en/latest/miniconda.html). Virtual
environments are a simple way to isolate all the dependencies for a particular
project, making it easy to work on multiple projects at once without them
interfering with each other (e.g. conflicting versions of libraries between
projects). To make sure your environment matches the testing environment that
we use for grading exactly, it's best to make a new environment for each
assignment in this course.

Install the latest version of [miniconda for your operating
system](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links).
After installing you should be able to run `conda` from your terminal. If you can't,
you may need to run `source ~/.bash_profile` or restart your terminal.
If you for some reason cannot use miniconda, try [python's venv
module](https://docs.python.org/3/library/venv.html).
Once you have `conda` set up, let's create a virtual environment by running:

``conda create -n cs349 python=3.9``

Once it's created, you can *activate* it with:

``conda activate cs349``

Here, `cs349` is the name for the environment. If you name it something
else and forget what you named it, you can call `conda env list` to list
all your conda environments. After activating it, you'll likely see that your
terminal prompt has changed to include `(cs349)`. Now, you can install
the packages necessary for this homework by going to the root directory
of this repository and running:

``pip install -r requirements.txt``

Once these install, you're all set up to do the homework! You'll need to do
this for each homework in this class. If you want to deactivate your
environment, you can simply call `conda deactivate`.

## What to do for this assignment

The detailed instructions for the work you need to do are in `problems.md`.

For the coding portion of the assignment, you will:
- Solve some numpy practice problems
- Implement a train test split and cross validation
- Implement some classification metrics
- Implement a simple model that just predicts the mode (most common class)
- Compute information gain
- Implement a decision tree with the ID3 algorithm

You will also write up answers to a few free response questions.

In every function where you need to write code, there is a `raise
NotImplementeError` which you will replace with your implementation. The test
cases will guide you through the work you need to do and tell you how many
points you've earned. The test cases can be run from the root directory of this
repository with:

``python -m pytest -s``

To run a single test, you can specify it with `-k`, e.g., `python -m pytest -s
-k test_setup_netid`.  To run a group of tests, you can use `-k` with a prefix, e.g.,
`python -m pytest -s -k test_decision` will run all tests that begin with
`test_decision`.  The `-s` means that any print statements you include will in
fact be printed; the default behavior (`python -m pytest`) will suppress
everything but the pytest output.

We will use these test cases to grade your work! Even if you change the test
cases such that you pass the tests on your computer, we're still going to use
the original test cases to grade your assignment.

## Questions? Problems? Issues?

Ask a question on Piazza, and we'll help you there.

## Helpful Material
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
