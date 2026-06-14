# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/skmadhavi21/tourism-project/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['Unnamed: 0'], inplace=True)
df.drop(columns=['CustomerID'], inplace=True)

# Treating Gender and MaritalStatus column
replace_dict_gender = {'Fe Male':'Female'}
df['Gender'] = df['Gender'].replace(replace_dict_gender)
replace_dict_status = {'Unmarried':'Single'}
df['MaritalStatus'] = df['MaritalStatus'].replace(replace_dict_status)

# Transforming CityTier column : this will help model to consider proper weightage of values
df['Tier_Transformed'] = 4 - df['CityTier']

# List of numerical features in the dataset
numeric_features = [
    'Age',                    # Customer's age
    'Tier_Transformed',       # The city category based on development, population, and living standards
    'DurationOfPitch',        # Duration of the sales pitch delivered to the customer.
    'NumberOfFollowups',      # Total number of follow-ups by the salesperson after the sales pitch.
    'PreferredPropertyStar',  # Preferred hotel rating by the customer.
    'NumberOfTrips',          # Average number of trips the customer takes annually.
    'Passport',               # Whether the customer holds a valid passport 
    'PitchSatisfactionScore', # Score indicating the customer's satisfaction with the sales pitch.
    'MonthlyIncome'           # Gross monthly income of the customer.
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # The customer was contacted (Company Invited or Self Inquiry)
    'Occupation',            # The customer’s occupation
    'Gender',                # The customer’s gender
    'ProductPitched',        # The product pitched to the customer
    'MaritalStatus',         # The customer’s marital status
    'Designation'            # Customer's designation in their current organization.
]

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="skmadhavi21/tourism-project",
        repo_type="dataset",
    )
