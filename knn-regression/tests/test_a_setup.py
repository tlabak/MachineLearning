import filetype
import os
import re

n_free_response = 5


def test_setup():
    """
    For HW2, you don't need to deal with a password. Just add your Net ID
        to the `netid` file, and add your PDFs to `netid_q1.pdf` through `netid_q5.pdf`.
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
