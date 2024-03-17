from metatree.model_metatree import LlamaForMetaTree as MetaTree
from metatree.decision_tree_class import DecisionTree, DecisionTreeForest
from metatree.run_train import preprocess_dimension_patch
from transformers import AutoConfig
import imodels # pip install imodels 

# Initialize Model
model_name_or_path = "yzhuang/MetaTree"

config = AutoConfig.from_pretrained(model_name_or_path)
model = MetaTree.from_pretrained(
    model_name_or_path,
    config=config,
)   

# Load Datasets
X, y, feature_names = imodels.get_clean_dataset('fico', data_source='imodels')

print("Dataset Shapes X={}, y={}, Num of Classes={}".format(X.shape, y.shape, len(set(y))))

train_idx, test_idx = sklearn.model_selection.train_test_split(range(X.shape[0]), test_size=0.3, random_state=seed)

# Dimension Subsampling
feature_idx = np.random.choice(X.shape[1], 10, replace=False)
X = X[:, feature_idx]

test_X, test_y = X[test_idx], y[test_idx]