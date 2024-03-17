from metatree.model_metatree import LlamaForMetaTree as MetaTree
from transformers import AutoConfig
import pandas as pd 
from sklearn.model_selection import train_test_split

# Initialize Model
model_name_or_path = "metatree_model"

config = AutoConfig.from_pretrained(model_name_or_path)
model = MetaTree.from_pretrained(
    model_name_or_path,
    config=config,
)   

# Load Dataset
data = pd.read_csv(r"dataset/train.csv")
row, col = data.shape
print("Dataset Shapes X={}, y={}".format(row, col))

X = data[data.loc[:, data.columns != 'URL'].columns]
y = df['URL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)