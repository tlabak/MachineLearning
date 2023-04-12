import os


def test_np_suggestions():
    """
    It will make it much easier for us to help you debug if you
    use our provided random number generation in `src.random`
    instead of `np.random`.
    """

    src_random_msg = "Please use `src.random` instead of `np.random` in line {} of {}."

    import src
    for fn in os.listdir(src.__path__[0]):
        if fn.endswith(".py"):
            fn = os.path.join(src.__path__[0], fn)
            with open(fn, encoding="utf-8") as inf:
                for i, line in enumerate(inf):
                    # Avoid complaining about comments that mention both
                    if "np.random" in line:
                        assert "src.random" in line, src_random_msg.format(
                            i, fn)


def test_imports():
    """
    Please don't import sklearn or scipy to solve any of the problems in this
    assignment.  If you fail this test, we will give you a zero for this
    assignment, regardless of how sklearn or scipy was used in your code.

    the 'a' in the file name is so this test is run first.
    """
    import sys
    disallowed = ["scipy", "sklearn"]
    for key in list(sys.modules.keys()):
        for bad in disallowed:
            if bad in key:
                del sys.modules[key]

    import src
    import pkgutil
    for loader, name, is_pkg in pkgutil.walk_packages(src.__path__):
        _module = loader.find_module(name).load_module(name)
        globals()[name] = _module

    # Checking that you have not imported disallowed packages
    for key in list(sys.modules.keys()):
        for bad in disallowed:
            assert bad not in key, f"Illegal import of {key}"

    # Checking that these package names only appear in the warnings
    #   about how you should not use them.
    disallowed += ["sys", "importlib"] 
    disallowed_str = ', '.join(disallowed)
    docstring = f"Do not import or use these packages: {disallowed_str}."
    print(docstring)
    for fn in os.listdir(src.__path__[0]):
        if fn.endswith(".py"):
            fn = os.path.join(src.__path__[0], fn)
            with open(fn, encoding="utf-8") as inf:
                for i, line in enumerate(inf):
                    for key in disallowed:
                        if key in line:
                            msg = f"Don't use {key} in line {i+1} of {fn}"
                            assert line.strip() == docstring, msg

    # Checking that these function names only appear in the warnings
    #   about how you should not use them.
    disallowed_funcs = ["getattr", "globals", "eval"]
    funcs_str = ", ".join(disallowed_funcs)
    docstring = f"Do not use these numpy or internal functions: {funcs_str}"
    print(docstring)
    for fn in os.listdir(src.__path__[0]):
        if fn.endswith(".py"):
            fn = os.path.join(src.__path__[0], fn)
            with open(fn, encoding="utf-8") as inf:
                for i, line in enumerate(inf):
                    for key in disallowed_funcs:
                        if key in line:
                            msg = f"Don't use {key} in line {i+1} of {fn}"
                            assert line.strip() == docstring, msg
