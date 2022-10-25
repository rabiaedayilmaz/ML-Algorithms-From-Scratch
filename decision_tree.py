import numpy as np

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tree = []

class DecisionTree:
    def __init__(self):
        self.build_tree(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)

    def compute_entropy(self, y):
        entropy = 0.
        if len(y) != 0:
            p1 = len(y[y == 1]) / len(y)
            if p1 != 0 and p1 != 1:
                entropy = -np.multiply(p1, np.log2(p1)) - np.multiply((1-p1), np.log2(1-p1))
        return entropy

    def split_dataset(self, x, node_indices, feature):
        left_indices = []
        right_indices = []
        for i in node_indices:
            if x[i][feature] == 1:
                left_indices.append(i)
            else:
                right_indices.append(i)
        return left_indices, right_indices

    def compute_information_gain(self, x, y, node_indices, feature):
        left_indices, right_indices = self.split_dataset(x, node_indices, feature)

        x_node, y_node = x[node_indices], y[node_indices]
        x_left, y_left = x[left_indices], y[left_indices]
        x_right, y_right = x[right_indices], y[right_indices]

        node_entropy = self.compute_entropy(y_node)
        left_entropy = self.compute_entropy(y_left)
        right_entropy = self.compute_entropy(y_right)

        w_left = len(x_left) / len(x_node)
        w_right = len(x_right) / len(x_node)

        weighted_entropy = w_left*left_entropy + w_right*right_entropy

        information_gain = node_entropy - weighted_entropy

        return information_gain

    def get_best_split(self, x, y, node_indices):
        num_features = x.shape[1]
        best_feature = -1
        max_info_gain = 0
        for feature in range(num_features):
            information_gain = self.compute_information_gain(x, y, node_indices, feature)
            if information_gain > max_info_gain:
                best_feature = feature
                max_info_gain = information_gain
        return best_feature

    def build_tree(self, x, y, node_indices, brach_name, max_depth, current_depth):
        if current_depth == max_depth:
            print(f"Current depth {current_depth} is max depth at {brach_name}, so stopping.")
            return

        best_feature = self.get_best_split(x, y, node_indices)

        print(f"Depth {current_depth} at {brach_name}, split on feature {best_feature}")

        left_indices, right_indices = self.split_dataset(x, node_indices, best_feature)
        tree.append((left_indices, right_indices, best_feature))

        self.build_tree(x, y, left_indices, "Left", max_depth, current_depth+1)
        self.build_tree(x, y, right_indices, "Right", max_depth, current_depth+1)


DecisionTree()
