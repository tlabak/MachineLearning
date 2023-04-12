# Rubric structure

`rubric.json` is structured as follows:

```
{
    "test_imports": {
            "weight": "required",
            "depends": []
        },
    "test_netid": {
        "weight": "required",
        "depends": []
    }
}
```

Each test case gets a weight. If weight is "required", you can't earn any points at all unless you
pass this test. Otherwise, weight shows the relative weight of this test compared to others.

We suggest that you solve the test cases in the order as they appear in `rubric.json`.
