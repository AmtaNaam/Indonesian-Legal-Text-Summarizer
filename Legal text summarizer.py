from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LogisticRegression
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import PyPDF2

# Load model LEGAL-BERT dan IndoBERT
legal_bert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_bert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

indobert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
indobert_tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# Fungsi untuk ekstraksi fitur menggunakan LEGAL-BERT
def legal_bert_embedding(text):
    inputs = legal_bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = legal_bert_model(**inputs)
    return outputs.pooler_output.detach().numpy()

# Fungsi untuk ekstraksi fitur menggunakan IndoBERT  
def indobert_embedding(text):
    inputs = indobert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = indobert_model(**inputs)
    return outputs.pooler_output.detach().numpy()

# Baca dokumen PDF
pdfFileObj = open('tempatkan file pdf kalian disini.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pages = []
for page in range(pdfReader.numPages):
    pageObj = pdfReader.getPage(page)
    pages.append(pageObj.extractText())
text = ' '.join(pages)  

# Pra-pemrosesan teks
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()  
clean_text = stopword.remove(text)
sentences = clean_text.split(". ")

# Ekstraksi fitur menggunakan LEGAL-BERT dan IndoBERT
legal_bert_features = legal_bert_embedding(sentences)
indobert_features = indobert_embedding(sentences)

# Gabungkan fitur dari kedua model
features = np.concatenate((legal_bert_features, indobert_features), axis=1)

# Definisikan model stacking
estimators = [
    ('legal_bert', legal_bert_embedding),
    ('indobert', indobert_embedding)
]

clf = StackingRegressor(
    estimators=estimators, 
    final_estimator=LogisticRegression(),
    cv=5
)

# Latih model stacking
clf.fit(features)  

# Identifikasi kalimat penting dengan model stacking
n_sentences = 3
scores = clf.predict(features)
top_sentences_indices = scores.argsort()[-n_sentences:][::-1]

# Hasil ringkasan
summary = ". ".join([sentences[i] for i in top_sentences_indices])

print("Ringkasan:")  
print(summary)