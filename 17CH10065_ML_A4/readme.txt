

-->The files are arranged in following manner


17CH10065_ML_A4
├── data
│   ├── seeds_dataset.txt
│   ├── test_data.txt
│   └── train_data.txt
├── readme.txt
└── src
    ├── dateset_preparation.py
    ├── Part1.py
    └── Part2.py



##################################################################################################

to run the python code
--> navigate to src directory
--> open the terminal and enter

$ python Part1.py
$ python Part2.py

#################################################################################################

Output will generate four plots for Part1.py ,two for each part(A,B) as follows

--> error vs epoc   (to get an idea of convergence)
--> accuraccy vs epoc (as asked in the question)

##################################################################################################
Sample output

$python Part1.py
Part 1A:
Final accuracy on training data is  87.5 %
Final accuracy on testing data is  95.23809523809524 %
Part 1B:
Final accuracy on training data is  89.88095238095238 %
Final accuracy on testing data is  90.47619047619048 %

$python Part2.py
Part 2 Specification 1A :
Accuracy on training data is  100.0 %
Accuracy on testing data is  100.0 %
Part 2 Specification 1A :
Accuracy on training data is  100.0 %
Accuracy on testing data is  100.0 %



