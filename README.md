# Machine Learning (CSL7550) Assignment Solutions
This repository contains the solutions of all the assignments which were given in the course Machine Learning (CSL7550).
- **Course Code** : CSL7550
- **Run** : Fall 2021
- **Instructor** : [Dr. Gaurav Harit](http://home.iitj.ac.in/~gharit/gharit/), Associate Professor, Dept. of CSE, [IIT Jodhpur](https://www.iitj.ac.in/)

For all the Assignments, there are seperate JuPyter notebooks. For each problem, there is a seperate `.py` file containing the Python code of the same.

Code has been briefly explained in their corresponding Jupyter notebook. Find the assignment which the problem belongs to and refer to the notebook of that assignment.

### All Requirements
You need to have the below mentioned libraries/packages to be set up in your environment.
- NumPy
- Matplotlib
- PuLP
- Scikits-learn
- Pandas

Find the `requirements.txt` file and run the below command to install all the above at one go : `pip install -r requirements.txt`

### Assignment-1 Questions
Choose an appropriate dataset of your choice such that every record (example) has at least 5 features which are numeric in nature and there is at least one attribute (feature) which is binary in nature. You can use the binary attribute as the binary target label to be predicted. In case you want to use a target variable which has more than two distinct values, then you can map them into two sets and give label 1 to one of the sets and 0 to the other. Thus, a multiclass classification task can be reduced to binary classification task.

Split your dataset into a training set and a test set. You can try different splits: 70:30 (70% training, 30% testing), 80:20 or 90:10 split.
On the training set, train the following classifiers:

1. **Half Space classifier implemented using LP solver (one such solver is scipy.optimize.linprog)**
2. **Half Space classifier implemented using Perceptron Algorithm (implement the iterations)**
3. **Logistic Regression Classifier**

You can use any other LP solver also. The optimization of the Logistic Regression should be done using gradient descent algorithm.

### Assignment-2 Questions
Select an appropriate dataset, select the independent features (input features) and the dependent feature (target feature), perform dataset split and **train a linear regression classifier**. Solve for the parameters of the machine to minimize the squared error loss using
1. **Pseudo-inverse method**
2. **Gradient descent**
