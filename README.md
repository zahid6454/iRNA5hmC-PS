# <center> iRNA5hmC-PS: Accurate prediction of RNA 5-Hydroxymethylcytosine Modification by Utilizing Position-Specific k-mer and Position-Specific Gapped k-mer </center>

## Authors: Sajid Ahmed†, Zahid Hossain†, Mahtab Uddin, Ghazaleh Taherzadeh, Alok Sharma Swakkhar Shatabda*, Abdollah Dehzangi*

## Website Link: http://103.109.52.8:81/iRNA5hmC-PS


## 1. Prerequisites
- Install: Anaconda 
- Install: Jupyter in Anaconda 
- Using "pip" install the following packages: (Command: pip install <package name>)
  - python
  - pandas
  - numpy
  - sklearn
  - matplotlib
  - imblearn
  - tabulate
- In Jupyter, create a new folder with any title
- All the following files needs to be inside that folder to successfully execute the codes:
  - Dataset.csv
  - Feature Generation.ipynb
  - Model.ipynb
  
## 2. Working Procredure
- Execute the "Feature Generation.ipynb" from Jupyter clicking "Cell" dropdown button on top and click "Run All" button.
  - "Feature Generation.ipynb" uses "Dataset.csv" file to generate all the proposed features and stores that into a new file titled as "All_Features_Dataset.csv"
  - After the execution of the "Feature Generation.ipynb", a new file titled "All_Features_Dataset.csv" will be avaiable in the folder.
- When "All_Features_Dataset.csv" file is available execute the "Model.ipynb" file just like "Feature Generation.ipynb" file was executed earlier.


## 3. Results
- Cross Validation Results of the three classifiers along with Logistic Regression (Proposed as Best Method) will be shown in the output cell after the execution of the cell 
  just before the last cell. Also, AUROC and AUPR curves will also be shown there.
- After the execution of the last cell, all the Cross Validation, Independent Test Results along with Standard Daviation will be shown several tables. Also, AUROC and AUPR \
  curves will also be shown of the Independent Test results in the output cell.


## 4. Web-Server Instructions
- There are two options avaiable at our web-server for iRNA5hmC-PS. 
  1. Paste RNA Sequence(s)
  2. Upload RNA Sequence(s)
  
- For Paste RNA Sequence(s),
  -  The user has to copy and paste RNA sequence having length of 41. The format has to be FASTA format. Then click "Submit".
  -  After that it will take some time to do the computation, and when the computation is complete the web-server will redirect to another page having a download button to 
     download the results.
  -  Also there is a reset button the clear the input field.

- For Upload RNA Sequence(s),
  - The user has to upload a .fasta file that contains RNA sequence. Also the file and the RNA sequences has to be in fasta format.
  - After selecting the file, user will click the submit button to submit the file.
  - iRNA5hmC-PS will do its computation and after sometime the user will be redirected to another page that contains a download button to download the predictions.
 
 - The results contains the predictions for the RNA sequences that were submitted by the user, and for each prediction it also shows the Probability if beign "0" and the
   Probability if being "1".
