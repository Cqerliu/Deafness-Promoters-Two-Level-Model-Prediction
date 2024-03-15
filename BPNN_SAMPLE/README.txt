Instruction
A two-stage cascade model was used to predict the deafness gene promoter

--DATA Data section:
--Test_DATA, test data set (length 300bp)
--Train_DATA and Validation_DATA, training and test data set (length 300bp)

--BPNN is the first-level structural model:
-- bpn_first. py: is the first level model of the experiment
--BPNN_connect.py: is the intermediate connection layer of this experiment
--BPNN_XGBoost.py: takes the XGBoost model as the second-level model
--Train_BPNN.py: is the main program that writes data to the path of the main function when training and testing, respectively

--Result part:
Gives two levels of results using the test set
--Graphics: The ROC curve is shown