import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

def prepare_knn_data(df, features=['a','b','tau','I','v0','w0'], label_col='symbol_binary'):
    """
    Prepare features and labels for KNN classification.
    """
    df_knn = df.dropna(subset=features + [label_col]).copy()
    df_knn['label'] = df_knn[label_col].map({'N':0, 'not N':1})
    
    X = df_knn[features].values
    y = df_knn['label'].values
    groups = df_knn['recording'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, groups, df_knn


def prepare_knn_data_general(df, features=['a','b','tau','I','v0','w0'], label_col='symbol'):
    
    # Drop rows with missing features or label
    df_knn = df.dropna(subset=features + [label_col]).copy()
    
    # Convert categorical labels to numeric
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_knn[label_col])
    
    # Features
    X = df_knn[features].values
    
    # Optional: group identifiers if you want Leave-One-Group-Out CV
    groups = df_knn['recording'].values if 'recording' in df_knn.columns else None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    class_names = list(label_encoder.classes_)

    return X_scaled, y, groups, df_knn, label_encoder, class_names


def knn_leave_one_group_out(X, y, groups, n_neighbors=5):
    """
    Run KNN with Leave-One-Group-Out cross-validation.
    """
    logo = LeaveOneGroupOut()
    y_preds = np.zeros_like(y)
    
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        # left_out_group = groups[test_idx][0]
        # print(f"Leaving out group: {left_out_group}, size={len(test_idx)}")
                                                           
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_preds[test_idx] = knn.predict(X_test)
    
    return y_preds

def classification_metrics(y_true, y_pred, target_names=['N','not N']):
    """
    return accuracy, classification report, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    return acc, report, cm
