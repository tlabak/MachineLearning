import filetype
import os
import re

n_free_response = 3


def test_setup_netid():
    """
    To pass this test case, you need to add your Net ID to the `netid` file in
    the root directory of your repository. You then need to create
    one PDF per free-response question. These PDFs can be blank, but they need
    to be named correctly. If your Net ID is `xyz0123`, then the first
    PDF should be named `xyz0123_q1.pdf`.
    """

    with open('netid', 'r') as inf:
        lines = inf.readlines()

    assert len(lines) == 1, "Just a single line with your NetID"

    netid = str(lines[0].strip())
    assert netid != "NETID_GOES_HERE", "Add your NetID"
    assert netid.lower() == netid, "Lowercase NetID, please"
    assert re.search(r"^[a-z]{3}[0-9]{3,4}$", netid) is not None, "Your NetID looks like xyz0123"

    files = os.listdir(".")
    for i in range(1, 1 + n_free_response):
        fn = f"{netid}_q{i}.pdf"
        assert fn in files, f"Please create {fn}"
        guess = filetype.guess(fn)
        msg = f"Is {fn} actually a pdf?"
        assert guess is not None, msg
        assert guess.mime == 'application/pdf', msg


def test_setup_password():
    '''
    The autograder will give you feedback by pushing its pytest output to your
    GitHub repo's `feedback` folder.  To help you understand how this works,
    we've included a puzzle that requires you to read the feedback from the
    autograder. After you add your NetID to the `netid` file and create your
    PDFs, your local autograder should show an error such as:

    ```
        assert inf.readline().strip() == secret, msg
        AssertionError: See tests/test_a_setup.py for details on this error.
        assert 'autograder_password_goes_here' != 'No password in tests/secrets.txt', 
    ```

    Commit and push your code, and wait for the autograder to run. When it
    does, it will create a new `feedback/` folder in your repository with a
    file named something like `Jan_10_12_00__abcd1234.txt`.  You can see this
    file on github.com, or download it by calling `git pull origin main`. This
    file will show the autograder pytest output, and will contain a similar
    error message to the one you saw before, except it will contain your
    password:

    ```
        assert inf.readline().strip() == secret, msg
        AssertionError: See tests/test_a_setup.py for details on this error.
        assert 'autograder_password_goes_here' != '2da3e727', 
    ```

    In this example, `2da3e727` is your password. You need to add that to your
    `password` file, replacing the `autograder_password_goes_here` with a single
    line containing this password.  Note that when you try to commit and push this
    change, you may get a scary-looking git error such as:

    ```
       ! [rejected]        main -> main (fetch first)
      error: failed to push some refs to 'git@github.com:nucs349w23/hw1-decision-tree-username.git'
      hint: Updates were rejected because the remote contains work that you do
      hint: not have locally. This is usually caused by another repository pushing
      hint: to the same ref. You may want to first integrate the remote changes
      hint: (e.g., 'git pull ...') before pushing again.
      hint: See the 'Note about fast-forwards' in 'git push --help' for details.
    ```

    All this means is that the autograder has successfully given you feedback,
    and you need to run `git pull origin main` to download it from GitHub to
    your local machine. When you do, it will likely open the [git editor](
    https://stackoverflow.com/questions/2596805/how-do-i-make-git-use-the-editor-of-my-choice-for-editing-commit-messages),
    which may be [vim by
    default](https://stackoverflow.com/questions/11828270/how-do-i-exit-vim).
    All you need to do is type `:wq<Enter>` or `ZZ` to exit vim.

    You should then get a message saying `Merge made by the 'recursive'
    strategy`, and the autograder feedback will now be available in the
    `feedback/` folder on your local machine.  You can call `git push origin
    main` to push your updated `password` to your repo, so that you can pass
    the autograder's `test_setup_password`.

    For the purposes of your grade on Canvas, it doesn't matter whether you
    pass `test_setup_password` locally. But if you want to, you should also put your
    NetID and this password in `tests/secrets.txt`; if your NetID is `xyz0123`
    and the password given to you is `2da3e727`, then put `xyz0123:2da3e727` in
    `tests/secrets.txt`.

    This additional hurdle is designed to help you understand what the autograder
    is doing. It is running `python -m pytest` (or just `python -m pytest -k
    test_setup` for the setup points) and giving you the output in your
    `feedback/` folder. Take advantage of this feedback! If you are passing tests
    locally but not on the autograder, it's important to understand why so that you
    can fix those issues and get credit for your work. Note that your actual
    grade (and a summary of tests passed) will be uploaded to Canvas, but the
    detailed feedback will be pushed to GitHub.
    '''

    with open('netid', 'r') as inf:
        lines = inf.readlines()
        netid = str(lines[0].strip())

    with open("password", "r") as inf:
        msg = "See tests/test_a_setup.py for details on this error."
        secret = get_feedback_secret(netid)
        assert inf.readline().strip() == secret, msg


def get_feedback_secret(netid):
    '''
    On the autograder server, this will grab a 'password' for you and compare
    it against what you have in the `password` file. Once you see the feedback
    the autograder pushes to your repository, you can find the password and add
    it to your `password` file. To pass this test locally, put `netid:password`
    in `tests/secrets.txt`; e.g., if your NetID is xyz0123 and your password
    were abcd1234, you would add a line with `xyz0123:abcd1234`.
    '''
    fn = "tests/secrets.txt"
    if os.path.exists(fn):
        with open(fn) as inf:
            for line in inf:
                if line.strip().startswith(netid):
                    return line.strip().split(":")[1]

    return 'No password in tests/secrets.txt'


def test_setup_other_pdfs():
    '''
    In the past, students have occasionally uploaded PDF files to a subfolder
    which means we were unable to grade their free-response questions.  This is
    just an automated check to warn you if you've uploaded those PDFs
    elsewhere. This test is not worth any points, so feel free to ignore it if
    you have a good reason for uploading additional PDFs elsewhere.
    '''
    with open('netid', 'r') as inf:
        lines = inf.readlines()
        netid = str(lines[0].strip())

    for (root, dirs, files) in os.walk("."):
        for fn in files:
            msg = f"If you want {os.path.join(root, fn)} graded, move it to the root directory."
            if fn.endswith("pdf"):
                assert root == ".", msg
