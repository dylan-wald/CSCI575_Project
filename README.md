# CSCI575_Project

**1) Name and CSM ID**

- Name: Dylan Wald
- Student ID: 10884058

**2) What programming language is being used**

 - Python programming language (version 3.11.0)

**3) Code Structure:**

- Main.py - runs the entire simulation, including the following scripts:
    - sys_id_data.csv - this is the data used to train and test the models
    - data_loader.py - this loads and cleans the provided data
    - f_RNN.py - this is the recurrent neural network class
    - f_NSSM.py - this is the neural state space model class
        - f_LinearConstrained.py - this contains the constraints on the A matrix (described in report)
    - f_GRU.py - this is the gated recurrent unit model class
    - plot_all.py - this script plots all of the results. Train data, loss vs. epochs, training performance and test performance

- Other information:
    - Demo: This folder contains screenshots of the code running. 1) initialization, 2) RNN running, 3) NSSM running, 4) GRU running
    - Documents: This folder contains a pdf of the final report
        - Paper_Plots: This folder contains the plots of the results used in the paper. Included in case they are hard to see in the report.

**4) How to run:**

- in Main.py, select type of normailzation by setting desired method equal to "True" and the others equal to "False" (lines 48 - 50)
- type "python Main.py" into the terminal/command prompt and press enter

Required packages:
- numpy (version 1.24.2)
- torch (version 2.0.0)
- matplotlib (version 3.7.1)
- pandas (version 1.5.3)