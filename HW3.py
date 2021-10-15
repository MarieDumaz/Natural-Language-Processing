'''
CS 593A 
Assesment 3
'''

###############################################################################
''' IMPORTS '''

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import csv
import json
from tqdm import tqdm
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

###############################################################################
''' PRE PROCESSING '''

# Import metadata
meta = pd.read_csv("movie.metadata.tsv", sep = '\t', header = None)
# Rename columns
meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]

# Import movie plots
plots = []

with open("plot_summaries.txt", 'r') as f:
    # Open text file
    reader = csv.reader(f, dialect='excel-tab') 
    for row in tqdm(reader):
        # Read each line
        plots.append(row)
        
# Create dataframe containing plots and movie ID
movie_id = []
plot = []

for i in tqdm(plots):
    # For each element, separate ID and plot
    movie_id.append(i[0])
    plot.append(i[1])

# Create dataframe
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

# Combine plots and metadata of importance
# Change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)

# Merge meta with movies
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

# Extract genres
genres = [] 

for i in movies['genre']: 
    # For each movie, convert string to dict and keep the value only
    genres.append(list(json.loads(i).values())) 

# Add the genres to the dataframe  
movies['genre_new'] = genres

# Remove samples with 0 genre tags
movies_new = movies[~(movies['genre_new'].str.len() == 0)]

# Cleaning text
def clean_text(text):
    # Remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # Remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # Remove whitespaces 
    text = ' '.join(text.split()) 
    # Convert text to lowercase 
    text = text.lower() 
    
    return text

# Call function and add clean plot to dataframe
movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))

movies_new.head()

# Download nltk stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

# Call function and remove stopwords
movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))

# One hot encoding
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre_new'])

# Transform target variable
y = multilabel_binarizer.transform(movies_new['genre_new'])

###############################################################################
''' TRAIN/TEST/VALIDATION SPLIT '''

xtrain, xsplit, ytrain, ysplit = train_test_split(movies_new['clean_plot'], y, test_size=0.1, random_state=9)
xval, xtest, yval, ytest = train_test_split(xsplit, ysplit, test_size=0.5, random_state=9)

valid_set = pd.DataFrame(zip(xval, yval))
train_set = pd.DataFrame(zip(xtrain, ytrain))
test_set = pd.DataFrame(zip(xtest, ytest))
valid_set.columns = ['plot_text', 'genres']
train_set.columns = ['plot_text', 'genres']
test_set.columns = ['plot_text', 'genres']

###############################################################################
''' DEFINE PARAMETERS '''

# Defining some key variables that will be used later on in the training
MAX_LEN = 300
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-06

###############################################################################
''' DATA PREPARATION '''

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.plot_text = dataframe.plot_text
        self.targets = self.data.genres
        self.max_len = max_len

    def __len__(self):
        return len(self.plot_text)

    def __getitem__(self, index):
        plot_text = str(self.plot_text[index])
        plot_text = " ".join(plot_text.split())

        inputs = self.tokenizer.encode_plus(
            plot_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation = True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
        
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
training_set = CustomDataset(train_set, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_set, tokenizer, MAX_LEN)
validation_set = CustomDataset(valid_set, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

# DataLoader to load data to NN
training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **test_params)
testing_loader = DataLoader(testing_set, **valid_params)

###############################################################################
''' CREATE NN '''

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 363)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = BERTClass()
model.to(device)

# Creating loss function
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

###############################################################################
''' TRAINING '''

def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%1000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

import time        
start = time.time()
        
for epoch in range(EPOCHS):
    train(epoch)

end = time.time()
print(end - start)

###############################################################################
''' 
TRAINING RESULTS

Epoch: 0, Loss:  0.7143189907073975
Epoch: 0, Loss:  0.4134242832660675
Epoch: 0, Loss:  0.2510915994644165
Epoch: 0, Loss:  0.15346306562423706
Epoch: 0, Loss:  0.10449010878801346

'''

###############################################################################
''' VALIDATION '''

# Get predictions for validation set
def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

# Compute accuracy and F1 score with different threshold
for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    for i in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]:
        outputs1 = np.where(np.array(outputs)>=i, 1, 0)
        outputs1 = np.argmax(outputs1, axis=1)
        targets1 = np.argmax(targets, axis=1)
        accuracy = metrics.accuracy_score(targets1, outputs1)
        f1_score_micro = metrics.f1_score(targets1, outputs1, average='micro')
        f1_score_macro = metrics.f1_score(targets1, outputs1, average='macro')
        print(f"THRESHOLD = {i}")
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        print("\n")

###############################################################################
'''
VALIDATION RESULTS

All metrics are the highest at threshold = 0.3

THRESHOLD = 0.1
Accuracy Score = 0.13636363636363635
F1 Score (Micro) = 0.13636363636363635
F1 Score (Macro) = 0.0022429906542056075


THRESHOLD = 0.15
Accuracy Score = 0.14641148325358852
F1 Score (Micro) = 0.14641148325358852
F1 Score (Macro) = 0.002387156163699624


THRESHOLD = 0.2
Accuracy Score = 0.14641148325358852
F1 Score (Micro) = 0.14641148325358852
F1 Score (Macro) = 0.002387156163699624


THRESHOLD = 0.25
Accuracy Score = 0.14641148325358852
F1 Score (Micro) = 0.14641148325358852
F1 Score (Macro) = 0.002387156163699624


THRESHOLD = 0.3
Accuracy Score = 0.16507177033492823
F1 Score (Micro) = 0.16507177033492823
F1 Score (Macro) = 0.002648294920263294


THRESHOLD = 0.35
Accuracy Score = 0.16507177033492823
F1 Score (Micro) = 0.16507177033492823
F1 Score (Macro) = 0.002648294920263294


THRESHOLD = 0.4
Accuracy Score = 0.16507177033492823
F1 Score (Micro) = 0.16507177033492823
F1 Score (Macro) = 0.002648294920263294


THRESHOLD = 0.5
Accuracy Score = 0.0004784688995215311
F1 Score (Micro) = 0.0004784688995215311
F1 Score (Macro) = 8.939066850811444e-06

'''
    
###############################################################################
''' TESTING '''

def test(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


for epoch in range(EPOCHS):
    outputs, targets = test(epoch)
    outputs1 = np.where(np.array(outputs)>=0.3, 1, 0)
    outputs1 = np.argmax(outputs1, axis=1)
    targets1 = np.argmax(targets, axis=1)
    accuracy = metrics.accuracy_score(targets1, outputs1)
    f1_score_micro = metrics.f1_score(targets1, outputs1, average='micro')
    f1_score_macro = metrics.f1_score(targets1, outputs1, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    
###############################################################################
'''
TESTING RESULTS

Accuracy Score = 0.18038277511961723
F1 Score (Micro) = 0.18038277511961723
F1 Score (Macro) = 0.00302608290825029

'''
