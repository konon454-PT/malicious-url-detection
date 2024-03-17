import random
import torch
from metatree.run_train import preprocess_dimension_patch
from metatree.decision_tree_class import DecisionTree

subset_idx = random.sample(train_idx, 256)
train_X, train_y = X[subset_idx], y[subset_idx]

input_x = torch.tensor(train_X, dtype=torch.float32)
input_y = torch.nn.functional.one_hot(torch.tensor(train_y)).float()

batch = {"input_x": input_x, "input_y": input_y, "input_y_clean": input_y}
batch = preprocess_dimension_patch(batch, n_feature=10, n_class=10)
model.depth = 2
outputs = model.generate_decision_tree(batch['input_x'], batch['input_y'], depth=model.depth)
decision_tree_forest.add_tree(DecisionTree(auto_dims=outputs.metatree_dimensions, auto_thresholds=outputs.tentative_splits, input_x=batch['input_x'], input_y=batch['input_y'], depth=model.depth))

print("Decision Tree Features: ", [x.argmax(dim=-1) for x in outputs.metatree_dimensions])
print("Decision Tree Threasholds: ", outputs.tentative_splits)