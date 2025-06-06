README for INM427 


This package contains everything needed to evaluate two pre-trained bankruptcy‑prediction models (an MLP and an SVM), reproduce at least two key figures from the accompanying paper, and inspect the full analysis pipeline.

Contents of the Submission Folder
------------------------
Files in root directory:

  • `best_mlp_weights.pth`  
    – PyTorch state dictionary for the final, best‑performing MLP model.  
  • `best_mlp.pth`  
    – Alternate serialized version of the MLP (identical weights).  
  • `best_svm.joblib`  
    – Serialized scikit‑learn SVM classifier (trained with optimal hyperparameters).  
  • `scaler.joblib`  
    – scikit‑learn StandardScaler object used to normalize test features.  
  • `pca.joblib`  
    – scikit‑learn PCA object reducing features to 42 principal components.  
  • `data.csv`  
  • `testing_FINAL.ipynb` & `testing_FINAL.html`  
    – Notebook (and HTML export) that loads the pre‑trained models and test set, runs evaluation metrics,  
      and reproduces Figure 1 (confusion matrices) & Figure 2 (ROC curves).  
  • `FINAL.ipynb` & `FINAL.html`  
    – Full analysis notebook (and HTML export): data exploration, preprocessing, SMOTE, model training,  
      hyperparameter tuning, metric comparisons, and multi‑figure generation.  
  • requirements.txt  
    – list of all Python packages needed.


Software Requirements
------------------------
The following library versions have been tested on Python 3.8–3.10:

Setup 
---------------------
1. Extract all the files from the zip archive into a folder named NC-Final
2. Change directory into the project root:
   cd NC-Final
3. Create and activate a new virtual environment, then install dependencies:
   python3 -m venv env
   source env/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
4. Launch the evaluation notebook:
   jupyter notebook testing_FINAL.ipynb
5. To view the full analysis and saved figures:
   open FINAL.html (or run FINAL.ipynb)





