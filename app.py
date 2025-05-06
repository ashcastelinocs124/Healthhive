from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
from PIL import Image
import numpy as np
import cv2
import pytesseract
import cv2
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow
from openai import OpenAI
#import xmltodict
from transformers import CLIPModel, CLIPProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Import the Document class
from flask import Flask, request,jsonify
from flask_cors import CORS


code = {
    'Blood, Heart and Circulation' : 'bls7_us',
    'Bones, Joints and Muscles' : 'bls10_us',
    'Brain and Nerves' : 'bls14_us',
    'Digestive System' : 'bls2_us',
    'Ear, Nose and Throat' : 'bls16_us',
    'Endocrine System' : 'bls23_us',
    'Immune System' : 'bls22_us',
    'Kidneys and Urinary System' : 'bls11_us',
    'Lungs and Breathing' : 'bls15_us'
}

definition = {
  'Blood, Heart and Circulation' : 'The cardiovascular system, responsible for pumping blood through the heart and vessels, delivering oxygen and nutrients while removing waste from tissues.',
  'Bones, Joints and Muscles' : 'The musculoskeletal system, comprising bones for structure, joints for movement, and muscles for mobility and support.',
  'Brain and Nerves' : 'The nervous system, including the brain, spinal cord, and peripheral nerves, responsible for processing information, controlling bodily functions, and enabling sensory perception.',
  'Digestive System' : 'The system that breaks down food, absorbs nutrients, and eliminates waste, consisting of the stomach, intestines, liver, pancreas, and associated organs.',
  'Ear, Nose and Throat' : ' The otolaryngological system, involving sensory organs responsible for hearing, balance, smell, taste, and vocalization.',
  'Endocrine System' : 'A network of glands that produce hormones to regulate metabolism, growth, reproduction, and other bodily functions.',
  'Immune System' : 'The bodys defense system against infections and diseases, including white blood cells, antibodies, and lymphatic organs.',
  'Kidneys and Urinary System' : ' The renal system, responsible for filtering blood, removing waste through urine, and maintaining fluid and electrolyte balance.',
  'Lungs and Breathing' : 'The respiratory system, facilitating oxygen intake, carbon dioxide removal, and gas exchange through the lungs and airways.'
}


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text = list(code.keys())
context = model.encode(text)
dimension = 384
index = faiss.Index


def match_medical_category(input):
  encoded = model.encode([input])
  similarities = cosine_similarity(context, encoded)
  #best_match_index = similarities.n


def sub_category_information(category):
  category = category.replace(" ","")
  url = f"https://medlineplus.gov/{category.lower()}.html"
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  return url



def mongo_db(title,content, URL):

  client = MongoClient("mongodb://127.0.0.1:27017/")

  # Select database and collection
  db = client.Symptoms  # Fixed typo ("Symtoms" -> "Symptoms")
  collection = db.BloodHeartCirculation


  article_data = {
      "title": title,
      "content": content,
      "source": URL
  }

  insert_result = collection.insert_one(article_data)

  # Print confirmation
  print(f"Inserted document ID: {insert_result.inserted_id}")


def get_information(category):
  category = category.replace(" ", "")
  url =  f"https://medlineplus.gov/{category.lower()}.html"
  response = requests.get(url)

  if response.status_code != 200:
    return f"Failed to fetch page: {response.status_code}"
  soup = BeautifulSoup(response.text, 'html.parser')
  disease_links = soup.find("div", {"id": "topic-summary"})
  soup = soup.find("p")
  results = {}
  mongo_db(category,str(disease_links.text),url)
 

def sub_category_information(category):
  category = category.replace(" ","")
  url = f"https://medlineplus.gov/{category.lower()}.html"
  response = requests.get(url)
  soup = BeautifulSoup(response.text,"html.parser")
  soup = soup.find_all("li", {"class": "item"})
  items = []
  for i in soup:
    i = i.text.replace(" ","").replace("\n","")
    get_information(i)
    items.append(i)

  return items

#(sub_category_information("Blood Heart and Circulation"))

def text_splitter(document):
  textsplitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
  )
  document = Document(page_content = document)
  docs = textsplitter.split_documents([document])
  return docs


def definition_mongodb():

  client = MongoClient("mongodb://127.0.0.1:27017/")
  db = client.Symptoms
  collection = db.Definition

  for key, value in definition.items():
    document = {
      "Name" : key,
      "Definition" : value
    }
    insert_result = collection.insert_one(document)
    print(f"Inserted document ID: {insert_result.inserted_id}")


def IndexIVFFLAT(category,further):
  model = SentenceTransformer("bert-base-nli-mean-tokens")
  d = 128
  nb = 10000

  vectors = []
  names = []

  client = MongoClient("mongodb://127.0.0.1:27017/")
  db = client.Symptoms
  collection = db.ImmuneSystem

  documents = collection.find()
  

  for document in documents:
    vector = model.encode(document["content"])
    vectors.append(vector)
    names.append(document["title"])
  
  vectors = np.array(vectors, dtype = 'float32')

  dimension = vectors.shape[1]

  index = faiss.IndexFlatL2(dimension)
  index.add(vectors)

  query_vector = model.encode([further])
  distances, indices = index.search(np.array(query_vector, dtype = 'float32'), k = 1)

  top_idx = indices[0][0]
  print(names[top_idx])
  documents = collection.find()
  for document in documents:
    if document["title"] == names[top_idx]:
      return (text_splitter(document["content"]))
      print(document["content"]) 
      break

def classification(issue):
  model = SentenceTransformer("bert-base-nli-mean-tokens")
  d = 128
  nb = 100

  vectors = []
  names = []
  client = MongoClient("mongodb://127.0.0.1:27017/")
  db = client.Symptoms
  collection = db.Definition

  documents = collection.find()
  print(documents)

  vectors = []
  names = []

  for document in documents:
    vector = model.encode(document['Definition'])
    vectors.append(vector)
    names.append(document['Name'])
  

  vectors = np.array(vectors, dtype = 'float32')

  dimension = vectors.shape[1]
  index = faiss.IndexFlatL2(dimension)

  index.add(vectors)
  
  query = input("How can I help you")
  query_vector = model.encode([query])
  return query_vector



def IndexFlatL2(query_vector,further):
  model = SentenceTransformer("bert-base-nli-mean-tokens")
  d = 128
  nb = 100

  vectors = []
  names = []
  client = MongoClient("mongodb://127.0.0.1:27017/")
  db = client.Symptoms
  collection = db.Definition

  documents = collection.find()
  print(documents)

  vectors = []
  names = []

  for document in documents:
    vector = model.encode(document['Definition'])
    vectors.append(vector)
    names.append(document['Name'])
  

  vectors = np.array(vectors, dtype = 'float32')

  dimension = vectors.shape[1]
  index = faiss.IndexFlatL2(dimension)

  index.add(vectors)

  distances, indices = index.search(np.array(query_vector, dtype='float32'), k=5)
  if len(indices[0]) > 0:
    top_idx = indices[0][0]
    return {"similar_case_summary": f"Closest match is {names[top_idx]} with distance {distances[0][0]}"}
  else:
    return None 
  # Print the results
  #print("Top 5 closest matches:")
  #print(distances[0][0])
  #print(indices[0][0])
  IndexIVFFLAT(query_vector, further)

#IndexFlatL2()



class externalknowledge:

  def abstract(search_term, base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'):
    url = f'{base_url}?db=pubmed&term={search_term}&retmode=json'
    response = requests.get(url)
    xml_data = response.text
    dictionary = xmltodict.parse(xml_data)
    for i in dictionary.values():
      i = i.values()
      record = list(i)[0]
    abstract_texts = record['MedlineCitation']['Article']['Abstract']['AbstractText']
    conclusion = next((item['#text'] for item in abstract_texts if item.get('@Label') == 'CONCLUSION'), None)
    return conclusion
  
  def keyword(search_term, base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'):
    url = f'{base_url}?db=pubmed&term={search_term}&retmode=json' 

    try:
      response = requests.get(url)
      data = response.json()

      data = (data['esearchresult'])
      id = (data['idlist'][1])
      abstract(id)
      article_url = f'https://pubmed.ncbi.nlm.nih.gov/{id}/'
      response = requests.get(article_url)
    except Exception as e:
      response = requests.get(url)
  

  class ocr:
    def opencv_format(self):
      pil_image = Image.open(imagepath).convert("RGB")
      numpy_image = np.array(pil_image)
      opecv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
      return opecv_image

    def grayscale(self, image = None):#Binarization
      if image is None:
        image = self.opencv_format()
        img = Image.open(imagepath).convert('L')
      return img

    def threshold(self):
      image = self.grayscale()
      img = cv2.imread(imagepath, 0)
      text = pytesseract.image_to_string(image)
      blur = cv2.GaussianBlur(img, (5, 5), 0)
      _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      print(text)
      #Value of a pixel is less then zero, we make it black if it is more then zero we will convert it to 1
      return 0
  class machinelearningmodel:
    def brain():

      model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
      processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
      image = img
      labels = ["broken skull","a normal skull", "a cracked skull","broken hand","broken arm", "hand fracture",""]
      inputs = processor(text = labels, images = image, return_tensors = "pt",padding = True)

      outputs = model(**inputs)
      logits_per_image = outputs.logits_per_image
      probs = logits_per_image.softmax(dim =  1)

      max = 0
      labell = []
      for label, prob in zip(labels, probs[0]):
        if (prob.item() > max):
          labell.clear()
          labell.append(label)
          max = prob.item()
        else:
          max = max
      print(labell[0],":",max)
        #print(f"{label}: {prob.item():.2%}")


from openai import OpenAI




from openai import OpenAI

class MedicalAssistant:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.messages = [{"role": "system", "content": self.context()}]
        self.follow_up_stage = 0
        self.additional_info = []
        self.query_vector = None
        self.model = SentenceTransformer("bert-base-nli-mean-tokens")

    def context(self):
        return (
            "You are a medical AI assistant responsible for answering queries related to medical issues.\n"
            "Start with 'Hey, how are you feeling?'. Then follow up with two questions, then give a final diagnosis based on a similarity result."
        )

    def fetch_vectors_from_collection(self, db_name, collection_name, field_name, label_field):
        client = MongoClient("mongodb://127.0.0.1:27017/")
        collection = client[db_name][collection_name]

        vectors = []
        labels = []

        for doc in collection.find():
            if field_name in doc and label_field in doc:
                vector = self.model.encode(doc[field_name])
                vectors.append(vector)
                labels.append(doc[label_field])

        if not vectors or not labels:
            raise ValueError("No valid data found in the collection.")

        return np.array(vectors, dtype='float32'), labels

    def classify(self, query):
        return self.model.encode([query])

    def IndexFlatL2(self, query_vector):
        vectors, names = self.fetch_vectors_from_collection("Symptoms", "Definition", "Definition", "Name")

        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        distances, indices = index.search(np.array(query_vector, dtype='float32'), k=1)
        top_idx = indices[0][0]
        return {"similar_case_summary": f"Closest match is {names[top_idx]} with distance {distances[0][0]}"}

    def handle_user_input(self, user_input):
        if self.follow_up_stage == 0:
            self.messages.append({"role": "user", "content": user_input})
            self.query_vector = self.classify(user_input)
            self.follow_up_stage = 1
            return {"response": "Can you describe your symptoms in more detail?"}

        elif self.follow_up_stage == 1:
            self.additional_info.append(user_input)
            self.follow_up_stage = 2
            return {"response": "How long have you been experiencing this?"}

        elif self.follow_up_stage == 2:
            self.additional_info.append(user_input)
            similarity_result = self.IndexFlatL2(self.query_vector)
            return {"response": self.generate_final_response(similarity_result)}

    def generate_final_response(self, similarity_info):
        final_prompt = (
            f"Symptoms: {self.additional_info[0]}\n"
            f"Duration: {self.additional_info[1]}\n"
            f"Similar case: {similarity_info['similar_case_summary']}\n\n"
            "Please provide a helpful diagnosis and recommendations."
        )

        self.messages.append({"role": "user", "content": final_prompt})

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.messages
        )

        return response.choices[0].message.content


from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Create a global assistant instance
medical_assistant = MedicalAssistant(api_key= api_key)

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    user_input = data.get("message")

    if user_input:
        response = medical_assistant.handle_user_input(user_input)
        return jsonify(response)
    else:
        return jsonify({"error": "No input provided"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
