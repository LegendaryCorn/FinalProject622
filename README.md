# Installation
This code will only run in environments that support swig as Box2D needs it. I used an Anaconda environment in order to get past this; you can install it through their website [here](https://www.anaconda.com/products/distribution). You can then create a conda environment and install your packages in there.

You will need to install several packages in order to get the code running. These packages are:

- numpy
- matplotlib
- gym with Box2D (must be version 0.17.3)
    - `pip install gym[Box2D]==0.17.3`

Note that an older version of gym must be installed for this to work; newer versions of gym completely changed how some of the functions work and aren't compatible with the current code. Version 0.25.2 must be used in particular because random functions seem to work differently in older versions; running the same code in different versions will produce different results.
# Runtime
Running the training function does not require any command line arguments. 

`python.exe lunarlandertrain.py`

Running the test function will require you to specify the folder with the Q-Table binary files. Usually, this will be the "qmodels" directory, so all you need to do is to add "qmodels" to the end of the command line input to add it as an argument.

`python.exe lunarlandertest.py qmodels`

Lunar Lander Train will output several Q-Tables in a directory called "qmodels". The amount of tables produced is by default set to 10. You can use these tables in lunarlandertest.py.
# Videos
[This video](https://www.anaconda.com/products/distribution) goes over the code functionality and explains how everything works.

[This video](https://www.anaconda.com/products/distribution) shows the model improving over time.