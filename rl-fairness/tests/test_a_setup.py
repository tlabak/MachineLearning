import os
import filetype

n_free_response = 5


def test_setup():
    with open('netid', 'r') as inf:
        lines = inf.readlines()

    assert len(lines) == 1, "Just a single line with your NetID"

    netid = str(lines[0].strip())
    assert netid != "NETID_GOES_HERE", "Add your NetID"
    assert netid.lower() == netid, "Lowercase NetID, please"

    files = os.listdir(".")
    for i in range(1, 1 + n_free_response):
        fn = f"{netid}_q{i}.pdf"
        assert fn in files, f"Please create {fn}"
        guess = filetype.guess(fn)
        msg = f"Is {fn} actually a pdf?"
        assert guess is not None, msg
        assert guess.mime == 'application/pdf', msg
