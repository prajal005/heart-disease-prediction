## Heart Disease Prediction

A Machine Learning project to predict the presence of heart disease based on clinical features. Includes EDA, preprocessing, model training, and evaluation.

## Project Structure
- `data/`: Contains the dataset(s) used for training and testing
- `notebooks/`: Jupyter notebooks for exploration, preprocessing, and model experimentation
- `models/`: Trained ML models saved for reuse
- `requirements.txt`: List of dependencies to install
- `README.md`: Project documentation

## Getting Started
1. Clone the repository(First time only)

On terminal(cmd):
Only once--->git clone https://github.com/prajal005/heart-disease-prediction.git

Paste this line on the terminal, this will create the project folder on your system

--->cd heart-disease-prediction

2. Setup a virtual environment  (Only once)

Terminal(cmd):
---> python -m venv venv   (Creating a virtual environment)
---> venv\Scripts\activate  (Activating the virtual environement)
---> deactivate   

3. Install dependencies

In virtual environment is activated

--->pip install -r -requirements.txt

4. To open updated repo

----> git pull origin main

(Everytime you log in this command is necessary)

5. Saving changes

Terminal:
---> git checkout -b -preprocessing   (-preprocessing is an example)

changes that you made must be written instead of -preprocessing

---> git add .
---> git commit -m "your message"  (ONLY COMMIT FOR BIG CHANGES)

like evaluation,training,etc.

---> git push origin preprocessing


6. Pull request on Github

-- Go to GitHub repo
-- A message pop-up 'Compare & Pull request', when new branch is created
-- Add details
-- Make sure PR is from the created branch from step 5 not from main
-- Create the Pull request
