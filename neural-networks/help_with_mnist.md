### Help with MNIST download

When you call `python -m free_response.q1` for the first time, the code will
try to download the MNIST dataset from OpenML servers. If you have a slow
internet connection or are behind a firewall, the code might fail to download
the data. To get around this, you can directly download the data from Canvas.


I’ve uploaded the dataset to [the Canvas Files
page](https://canvas.northwestern.edu/courses/181929/files) as both a tgz
archive and a zip archive. Try downloading one of those two from there, and
then copy it into your repository in the data/ folder. Once it’s there, you’ll
need to extract the archive. Depending on your computer and operating system,
you should have at least one of zip or tar installed; use the corresponding
file.

  - If you have the tgz file, run:
    - `tar xzvf mnist.tgz`

  - If you have the zip file, run:
    - `unzip mnist.zip`

  - This should create a openml folder such that your repository structure
    looks something like:

    ```
    ├─ data/
    │   ├─ openml/
    │   └─ circles.csv
    ├─ src/
    │   ├─ __init__.py
    │   └─ data.py
    ├─ free_response/
    │   └─ q1.py
    ```
Then, you should be able to run `python -m free_response.q1` and it will use
the pre-downloaded MNIST instead of trying to download it again.
