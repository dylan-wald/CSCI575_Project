# CSCI575_Project

**1) Dylan Wald, ID: 10884058**

**2) Python programming language (version 3.11.0)**

**3) Code Structure:**

- Main.py - runs the entire simulation, including the following scripts:
    - data_loader.py - this loads and cleans the data
    - f_RNN.py - this is the recurrent neural network class
    - f_NSSM.py - this is the neural state space model class
        - f_LinearConstrained.py - this contains the constraints on the A matrix (described in report)
    - plot_all.py - this script plots all of the results. Train data, loss vs. epochs, training performance and test performance

**4) How to run:**
    1) in Main.py, select type of normailzation by setting equal to "True", the rest to "False" (lines 48 - 50)
    2) type "python Main.py" into the terminal and press enter

Required packages:
- numpy (version 1.24.2)
- torch (version 2.0.0)
- matplotlib (version 3.7.1)
- pandas (version 1.5.3)