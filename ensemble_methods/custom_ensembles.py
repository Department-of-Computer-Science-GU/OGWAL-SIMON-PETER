import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class CustomRandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.trees = []
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # We use sklearn's DecisionTreeClassifier as the base estimator 
            # to avoid writing a full performant decision tree from scratch, 
            # but the ensemble logic (Random Forest) is fully from scratch.
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                max_features=self.max_features, 
                random_state=np.random.randint(0, 100000)
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict(self, X):
        X = np.array(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority voting
        preds = []
        for i in range(X.shape[0]):
            counts = Counter(tree_preds[:, i])
            preds.append(counts.most_common(1)[0][0])
        return np.array(preds)
    
    def predict_proba(self, X):
        X = np.array(X)
        tree_preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)


class CustomAdaBoost:
    def __init__(self, n_estimators=50, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        self.model_weights = []
        self.classes_ = None
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        
        # Convert y to -1, 1 if binary classification (assuming classes are 0 and 1 or -1 and 1)
        # Using the actual classes but mapping them to -1 and 1 internally
        y_encoded = np.where(y == self.classes_[0], -1, 1)
        
        weights = np.full(n_samples, (1 / n_samples))
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.models = []
        self.model_weights = []
        
        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=1, random_state=np.random.randint(0, 100000))
            stump.fit(X, y_encoded, sample_weight=weights)
            
            predictions = stump.predict(X)
            
            # Calculate error
            err = np.sum(weights[predictions != y_encoded])
            
            if err >= 0.5:
                break
            if err == 0:
                err = 1e-10
                
            # Calculate model weight (alpha)
            alpha = 0.5 * np.log((1 - err) / err)
            
            # Update instance weights
            weights *= np.exp(-alpha * y_encoded * predictions)
            weights /= np.sum(weights)
            
            self.models.append(stump)
            self.model_weights.append(alpha)
            
    def predict(self, X):
        X = np.array(X)
        stump_preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.dot(self.model_weights, stump_preds)
        
        y_pred_encoded = np.sign(weighted_preds)
        
        # Decode back to original classes
        y_pred = np.where(y_pred_encoded <= 0, self.classes_[0], self.classes_[1])
        return y_pred
    
    def predict_proba(self, X):
        X = np.array(X)
        stump_preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.dot(self.model_weights, stump_preds)
        sum_alpha = np.sum(self.model_weights)
        
        if sum_alpha == 0:
            sum_alpha = 1e-10
            
        # Normalize to [0, 1] range
        probs_class_1 = (weighted_preds + sum_alpha) / (2 * sum_alpha)
        # Ensure bounds
        probs_class_1 = np.clip(probs_class_1, 0, 1)
        probs_class_0 = 1 - probs_class_1
        
        return np.vstack((probs_class_0, probs_class_1)).T


class CustomStacking:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # Train base models
        for name, model in self.base_models:
            model.fit(X, y)
            
        # Get predictions for meta model. 
        # A more robust implementation would use K-Fold cross validation 
        # to generate these predictions to prevent overfitting, but we'll 
        # use training data predictions here for simplicity.
        base_preds = np.column_stack([model.predict(X) for name, model in self.base_models])
        
        self.meta_model.fit(base_preds, y)
        
    def predict(self, X):
        X = np.array(X)
        base_preds = np.column_stack([model.predict(X) for name, model in self.base_models])
        return self.meta_model.predict(base_preds)
        
    def predict_proba(self, X):
        X = np.array(X)
        base_preds = np.column_stack([model.predict(X) for name, model in self.base_models])
        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(base_preds)
        else:
            preds = self.meta_model.predict(base_preds)
            probs = np.zeros((len(preds), 2))
            for i, p in enumerate(preds):
                # Assuming classes are 0 and 1
                probs[i, int(p)] = 1.0
            return probs


class CustomVotingClassifier:
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting
        self.classes_ = None
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        
        for name, model in self.estimators:
            model.fit(X, y)
            
    def predict(self, X):
        X = np.array(X)
        if self.voting == 'hard':
            preds = np.column_stack([model.predict(X) for name, model in self.estimators])
            majority_votes = []
            for i in range(preds.shape[0]):
                counts = Counter(preds[i])
                majority_votes.append(counts.most_common(1)[0][0])
            return np.array(majority_votes)
        else:
            probas = np.array([model.predict_proba(X) for name, model in self.estimators])
            avg_probas = np.mean(probas, axis=0)
            return self.classes_[np.argmax(avg_probas, axis=1)]
