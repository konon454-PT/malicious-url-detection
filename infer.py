tree_pred = decision_tree_forest.predict(torch.tensor(test_X, dtype=torch.float32))

accuracy = accuracy_score(test_y, tree_pred.argmax(dim=-1).squeeze(0))
print("MetaTree Test Accuracy: ", accuracy)