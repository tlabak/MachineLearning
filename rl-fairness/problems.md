# Coding (6 points)

Your task is to implement two reinforcement learning algorithms:

1.  Multi-armed Bandits (in `code/multi_armed_bandits.py`)
1.  Q-Learning (in `code/q_learning.py`)

Note that while these reinforcement learning methods inherently depend
on randomization, we provide a `src/random.py` package that will randomize
things in the same way for all students. Please use `src.random` anywhere
that you might have otherwise used `np.random`.

Your goal is to pass the test suite (contained in `tests/`). Once the tests are
passed, you will use your code to answer FRQ 2 and 3.
We suggest that you try to pass the tests in the order they are listed in
`tests/rubric.json`.

Your grade for this section is defined by the autograder. If it says you got an 75/100,
you get 75% of the coding points.


# Free-response Questions (14 Total points)

To answer some of these questions, you will have to write extra code (that is
not covered by the test cases). You may (but are not required to) include your
experiments in new files in the `free_response` directory. See
`free_response/q2.py` for some sample code. You can run any experiments you create
within this directory with `python -m free_response.<filename>`. For
example, `python -m free_response.q2` runs the example experiment.

## 1. (1.5 total points) Tic-Tac-Toe
Suppose we want to train a Reinforcement Learning agent to play the game of
[Tic-Tac-Toe](https://en.wikipedia.org/wiki/Tic-tac-toe), and need to construct
an environment with states and actions. Assume our agent will simply choose
actions based on the current state of the game, rather than trying to guess
what the opponent will do next.

- a. What should be the states and actions within the Tic-Tac-Toe Reinforcement
  Learning environment? Don't try to list them all, just describe how the rules
  of the game define what states and actions are possible.  How does the
  current state of the game affect the actions you can take?
- b. Design a reward function for teaching a Reinforcement Learning agent to
  play optimally in the Tic-Tac-Toe environment.  Your reward function should
  specify a reward value for each of the 3 possible ways that a game can end
  (win, loss, or draw) as well as a single reward value for actions that do not
  result in the end of the game (e.g., your starting move). Explain your
  choices.
- c. Assume your agent gets to play first (and then alternates turns with the
  opponent). For actions that do not end the game, should the environment give
  rewards to the agent before or after the opponent plays? Why?

## 2. (2.5 total points) Bandits vs Q-Learning

- a. Run `python -m free_response.q2`; it will create three plots:
  `2a_SlotMachines_Comparison.png`, `2a_FrozenLake_Comparison.png`, and
  `2a_SlipperyFrozenLake_Comparison.png`. It might help to read a bit about the
  [FrozenLake environment](
  https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) .
  Each plot will show a comparison of your MultiArmedBandit and QLearning
  models on the named environment (e.g., SlotMachines). Include those plots
  here. For each plot, provide a one-sentence description of the most notable
  trend. Pay attention to the scale on the y-axis.

- b. In which of the above plots does QLearning appear to receive higher
  rewards on average than MultiArmedBandit? Provide an explanation for
  why that happens, based on your understanding of QLearning.

- c. Following b.: in the environment(s) where MultiArmedBandit was the
  **worse** model, is there any way you could change your choice of
  hyperparameters so that MultiArmedBandit would perform as well as QLearning?
  Why or why not?

- d. In which of the above plots does MultiArmedBandit appear to receive higher
  rewards on average than QLearning? Provide an explanation for
  why that happens, based on your understanding of MultiArmedBandit.

- e. Following d.: in the environment(s) where QLearning was the **worse**
  model, is there any way you could change your choice of hyperparameters so
  that QLearning would perform as well as MultiArmedBandit?  Why or why not?

## 3. (2 total points) Exploration vs. Exploitation

- a. Look at the code in `free_response/q3.py` and complete the two sections 
  with `TODO` listed in the comments. Then, run `python -m free_response.q3`
  and include the plot it creates `free_response/3a_g0.9_a0.2.png` as your answer
  to this part.

- b. Using the above plot, describe what you notice. What seems to be the
  ``best'' value of epsilon? What explains this result?

- c. The above plot trains each agent for 50,000 timesteps. Suppose we instead
  trained them for 500,000 or 5,000,000 timesteps. How would you expect the
  trends to change or remain the same for each of the three values of epsilon?
  Give a one-sentence explanation for each value.

- d. When people use reinforcement learning in practice, it can be difficult to
  choose epsilon and other hyperparameters. Instead of trying three options
  like we did above, suppose we tried 30 or 300 different choices. What might
  be the danger of choosing epsilon this way if we wanted to use our agent in a
  new domain?


## 4. (4 total points) Fair ML in the real world

Read [Joy Buolamwini and Timnit Gebru, 2018. "Gender shades: Intersectional
accuracy disparities in commercial gender classification." Conference on
fairness, accountability and
transparency](http://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf),
then use it to help answer the following questions.

- a. Buolamwini and Gebru use PPV and FPR as metrics to measure fairness. Find
  the definition of these in the paper, then look up the corresponding
  definition for NPV and FNR (these appear in the slides). Assuming binary
  labels and predictions, describe each of these four metrics as a probability,
  e.g. "The PPV gives the probability that the label is **BLANK** given that
  the model predicts **BLANK**."

- b. Assuming you were applying for a loan and you know a ML classifier is
  deciding whether to grant it to you: would you rather have that decision made
  by a system with a high PPV or a high NPV? Why?

- c. What recommendations do Buolamwini and Gebru make regarding accountability
  and transparency of ML systems? How does this relate to specific metrics such
  as PPV or FPR?

- d. What is *intersectional* about the analysis conducted by the authors? What
  does that analysis show?

- e. In Section 4.7, the authors say that their "findings ... do not appear to
  be confounded by the quality of sensor readings." What do they mean by
  "confounded" in this context? Why is it important to the thesis of this paper
  to check whether their findings are confounded?

## 5. (4 total points) Fair ML with a toy dataset

For this question, look at the code provided in `free_response.q5`.  You only
need to write a small amount of code, and are not required to push this code to
GitHub. The data for this problem (in `data/fairness_data.csv`) has four
columns: Income, Credit, Group, and Loan. Suppose that we want to predict
whether an applicant will receive a loan based on their income and credit. Loan
is a binary variable, and Income and Credit are continuous.  Group is some
demographic category (e.g. gender or age) which is binary in this data.  We
want our classifier to be fair -- it should not perform differently overall for
individuals with G=0 or G=1. The provided code will train several
LogisticRegression models on this data, and you will analyze these models'
behavior. 

- a.  Using the definitions for PPV, NPV, FPR, and FNR from __4a__
  above, implement those metrics in the `metrics()` function. Now, run `python
  -m free_response.q5` and fill in the table with the metrics for each part.
  Note that this code will create a plot in `free_response/5a.png`, which you
  will need for parts b-d. below. The metrics you need for this question are
  printed out in the terminal, not shown in the plot.

  |Part|Model        |PPV |NPV |FPR |FNR |
  |--- |---          |--- |--- |--- |--- |
  |b.  | Overall     |    |    |    |    |
  |b.  | Group 0     |    |    |    |    |
  |b.  | Group 1     |    |    |    |    |
  |c.  | Overall     |    |    |    |    |
  |c.  | Group 0     |    |    |    |    |
  |c.  | Group 1     |    |    |    |    |
  |d.  | G=0 Overall |    |    |    |    |
  |d.  | G=1 Overall |    |    |    |    |

- b.  Look at the LogisticRegression model trained in `part_b()`
  and shown in the top left plot. In `free_response/5a.png`, the area shaded grey
  shows positive predictions; the area shaded red shows negative predictions.
  * To what extent does the classifier treat individuals in Group 0 differently from those in Group 1?
  * If you were applying for a loan and this classifier were making the
    decision, would you rather be a member of Group 0 or Group 1? Why?
  * When you look at the Overall, Group 0 and Group 1 metrics you added to the table above,
    which group's metrics are most similar to that of the Overall metrics?
    Why?

- c.  Consider the LogisticRegression model trained in `part_c()` and shown in
  the top right plot of `free_response/5a.png`. Looking at the code for
  `part_b` and `part_c`, what's different about how this model was trained
  compared to the model from part __b.__?  Does this model's decision boundary
  and fairness metrics differ from those of the model in part __b.__? Does this
  surprise you? Why or why not?

- d.  Look at the code for both LogisticRegression models trained in
  `part_d()` and visualized in the bottom two plots of `free_response/5a.png`.
  * What is different about how each of these two models were trained?
  * If you were applying for a loan and were a member of Group 0, would you
    rather be classified by the part __d.__ "G=0" classifier or the classifier
    from part __b.__? Why?
  * If you were applying for a loan and were a member of Group 1, would you
    rather be classified by the part __d.__ "G=1" classifier or the classifier
    from part __b.__? Why?

- e.  The [US Equal Credit Opportunity Act
  (ECOA)](https://www.justice.gov/crt/equal-credit-opportunity-act-3) forbids
  "credit discrimination on the basis of race, color, religion, national
  origin, sex, marital status, age, or whether you receive income from a public
  assistance program." Suppose a bank wants to use machine learning to decide
  whether to approve or reject a loan application, but also wants to comply
  with the ECOA.  What are __two__ additional challenges that might be
  introduced if your ML system needs to be fair with respect to eight protected
  attributes instead of just one?
