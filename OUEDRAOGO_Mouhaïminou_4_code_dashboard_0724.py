import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('my_data.csv')
    df_sample = df.sample(frac=0.1, random_state=42)
    return df_sample

df = load_data()

# Titre et sous-titres
st.title('Dashboard de Prédiction de Tags')

st.subheader('Analyse Exploratoire des Données')

# Affichage du DataFrame avec un titre
st.markdown('### Aperçu du Dataset')
st.write(df.head())

# Affichage des valeurs manquantes avec un titre
st.markdown('### Nombre de Valeurs Manquantes par feature')
st.write(df.isnull().sum())

# Affichage des statistiques descriptives avec un titre
st.markdown('### Statistiques Descriptives')
st.write(df.describe())

# Sélecteur de nombre de tags
k = st.slider('Choisissez le nombre de tags à afficher', min_value=1, max_value=50, value=10)

# Fonction pour obtenir les tags les plus fréquents
@st.cache_data
def get_top_tags(df, k):
    # Séparation des tags
    tags = []
    for i in range(len(df)):
        tags.append(df["Tags"].iloc[i].split('|'))

    # Aplatir la liste des listes en une seule liste
    nested_tags = [subtag.strip() for tag in tags for subtag in tag if subtag.strip()]

    # Création d'un DataFrame pour compter les occurrences des tags
    df_tags = pd.DataFrame(nested_tags, columns=['tags'])
    tag_counts = df_tags['tags'].value_counts().reset_index()
    tag_counts.columns = ['tags', 'count']
    top_tags = tag_counts.head(k)
    return top_tags

# Obtenir les tags les plus fréquents avec cache
top_tags = get_top_tags(df, k)

# Création du graphique
fig, ax = plt.subplots()
sns.barplot(x=top_tags['tags'], y=top_tags['count'], ax=ax, palette='deep')
ax.set_title(f'Top {k} Tags les Plus Fréquents')
ax.set_xlabel('Tags')
ax.set_ylabel("Nombre d'occurrences")
plt.xticks(rotation=45, ha='right')

# Affichage du graphique dans Streamlit
st.pyplot(fig)

# Sélecteur pour choisir la colonne à afficher
column = st.selectbox('Choisissez la colonne à afficher', ['Title', 'Body'])

# Fonction pour obtenir la distribution des longueurs de texte
@st.cache_data
def get_text_length_distribution(df, column):
    # Convertir les valeurs en chaînes de caractères et calculer les longueurs
    text_lengths = df[column].astype(str).apply(len)
    return text_lengths

# Obtenir la distribution des longueurs de texte pour la colonne choisie
text_lengths = get_text_length_distribution(df, column)

# Création du graphique pour la distribution des longueurs de texte
fig_length = go.Figure([go.Histogram(x=text_lengths, marker_color='lightcoral')])
fig_length.update_layout(
    title=f'Distribution des Longueurs de Texte dans {column}',
    xaxis_title='Longueur du Texte',
    yaxis_title='Fréquence',
    template='plotly_white'
)

# Affichage du graphique de la distribution des longueurs de texte dans Streamlit
st.plotly_chart(fig_length)

# Définir la classe du modèle
class SciBERTCNN(nn.Module):
    def __init__(self, num_labels, max_length):
        super(SciBERTCNN, self).__init__()
        self.scibert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64 * (max_length // 2 // 2), 64)
        self.fc2 = nn.Linear(64, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.scibert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state.permute(0, 2, 1)
        x = self.conv1(last_hidden_state)
        x = torch.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def load_model(model_path, num_labels, max_length):
    model = SciBERTCNN(num_labels=num_labels, max_length=max_length)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
    return model

# Chemin vers le modèle sauvegardé
MODEL_PATH = 'checkpoint.pth'
MAX_LENGTH = 64
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

# Charger le modèle et le binarizer
model = load_model(MODEL_PATH, num_labels=11, max_length=MAX_LENGTH)
model.eval()

# Définir les tags réels utilisés lors de l'entraînement
real_tags = ['python', 'c#', 'javascript', 'java', 'android', 'c++', '.net', 'ios', 'php', 'html', 'others'] 
mlb = MultiLabelBinarizer(classes=real_tags)
mlb.fit([real_tags])

def predict_tags(text, model, tokenizer, max_length):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).numpy()
        predicted_labels = (probs > 0.5).astype(int)
        predicted_tags = mlb.inverse_transform(predicted_labels)

    return predicted_tags

# Moteur de Prédiction de Tags
st.subheader('Moteur de Prédiction de Tags')
title = st.text_input('Titre', help="Saisir le titre de la question.")
body = st.text_area('Contenu', help="Saisir le contenu de la question.")

if st.button('Prédire'):
    if title and body:
        combined_text = title + " " + body
        predicted_tags = predict_tags(combined_text, model, tokenizer, MAX_LENGTH)
        st.write(f'Tag Prédit : {predicted_tags}')
    else:
        st.write('Veuillez saisir à la fois le titre et le contenu de la question.')

# Assurer la navigabilité au clavier et l'accessibilité
st.markdown("""
<style>
    .css-1aumxhk {
        outline: none !important;
    }
    .css-1aumxhk:focus {
        outline: 2px solid #005a9e !important;  /* Contraste élevé pour l'accessibilité */
    }
</style>
""", unsafe_allow_html=True)
