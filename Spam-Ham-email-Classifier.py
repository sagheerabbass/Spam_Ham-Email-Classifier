import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# Data loading and Preprocessing
df=pd.read_csv('spam.csv',encoding='latin-1')[['v1','v2']]
df.columns=['label','message']

# Encode labels: ham = 0, spam = 1
df['label_num']=df['label'].map({'spam':1,'ham':0})
# Split Dataset
X_train,X_test,y_train,y_test=train_test_split(df['message'],df['label_num'],test_size=0.3,random_state=42)

# Convert text to TF-IDF features
vectorizer=TfidfVectorizer()
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)

# Train SVM Classifier
svc=SVC(kernel='linear',C=1.0)
svc.fit(X_train_vec,y_train)
# Predictions
y_pred=svc.predict(X_test_vec)

# Evaluation
print('Accuracy:',accuracy_score(y_test,y_pred))
print('\n📊 Classification Report :\n',classification_report(y_test,y_pred))
print('\n🧩 Confusion Matrix :\n',confusion_matrix(y_test,y_pred))

# Problem Understanding 
# Data Collection
# Data Preprocessing :
# Load dataset using pandas.

# Drop unnecessary columns (the extra ones in spam.csv).

# Rename columns to label and message.

# Encode labels → spam = 1, ham = 0.

# Split dataset → training and testing sets.