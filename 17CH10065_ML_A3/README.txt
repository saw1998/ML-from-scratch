
-->The files are arranged in following manner

17CH10065_ML_A3
├── clusters
│   ├── agglomerative_reduced.txt
│   ├── agglomerative.txt
│   ├── kmeans_reduced.txt
│   └── kmeans.txt
├── data
│   ├── AllBooks_baseline_DTM_Labelled.csv
│   ├── data_frame_Tfidf.csv
│   └── modified.csv
├── README.txt
└── src
    ├── __pycache__
    │   ├── Task_B.cpython-37.pyc
    │   └── Task_C.cpython-37.pyc
    ├── Task_A.py
    ├── Task_B.py
    ├── Task_C.py
    ├── Task_D.py
    └── Task_E.py

##################################################################################################

to run the python code
--> navigate to src directory
--> open the terminal and enter

$ python Task_A.py
$ python Task_B.py
$ python Task_C.py
$ python Task_D.py
$ python Task_E.py

#################################################################################################

Output of the Task_A.py is stored in

->data_frame_Tfidf.csv
->modified.csv

#################################################################################################

The output of Task_E.py are

->Normalized Mutual Information using Aggolmerization is 0.02119241463800962
->Normalized Mutual Information using K-means is 0.3922776715670199
->Normalized Mutual Information using Agglometization on reduced data is 0.03621015922738853
->Normalized Mutual Information using K-means on reduced data is 0.46193934168182177
