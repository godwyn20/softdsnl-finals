#  SOFTDSNL Final Project — Image Captioning + Sentiment Analysis (Flickr8k)

 A Django-based AI project that combines three deep learning systems:
 Image Captioning — generates captions from uploaded images.
 Sentiment Analysis — analyzes whether a text or caption is positive or negative.
 Combined Model — uploads an image, auto-generates a caption, and instantly predicts its sentiment.

---

##  Overview
- This project integrates Computer Vision (CNN) and Natural Language Processing (LSTM) to create an end-to-end intelligent system.
- It uses the Flickr8k dataset to train the captioning model and a custom-trained sentiment model to analyze emotions.

---

##  Dataset
- **Dataset:** [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- It contains:
  - ~8,000 images
  - Each image has 5 human-written captions
- You will:
  1. Preprocess images (resize, normalize)
  2. Tokenize and encode text captions
  3. Train a CNN + LSTM model to predict captions from images
  4. Perform sentiment analysis (positive/neutral/negative) on the generated captions

---


##  Project Structure
```
softdsnl-finals/
│
├── data/
│   ├── images/                     # Flickr8k image dataset
│   └── captions.txt                # Image-caption pairs
│
├── flickr8k_backend/               # Django backend project
│   ├── combined/                   # Combined caption + sentiment endpoint
│   │   ├── __pycache__/
│   │   ├── urls.py
│   │   └── views.py
│   │
│   ├── image_caption/              # Image captioning app
│   │   ├── __pycache__/
│   │   ├── migrations/
│   │   ├── utils/                  # Helper functions
│   │   │   ├── caption_generator.py
│   │   │   ├── preprocess.py
│   │   │   └── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── tests.py
│   │   ├── urls.py
│   │   └── views.py
│   │
│   ├── sentiment/                  # Sentiment analysis app
│   │   ├── __pycache__/
│   │   ├── migrations/
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── tests.py
│   │   ├── urls.py
│   │   ├── views.py
│   │   └── __init__.py
│   │
│   ├── asgi.py
│   ├── manage.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── model_training/                 # Model training scripts
│   ├── train_caption_model.py
│   └── train_sentiment_model.py
│
├── result/                         # Trained models and outputs
│   ├── caption/
│   │   ├── caption_model.h5
│   │   ├── caption_training_table.png
│   │   ├── image_features.npy
│   │   └── tokenizer.pkl
│   │
│   └── sentiment/
│       ├── sentiment_model.h5
│       ├── sentiment_tokenizer.pkl
│       ├── sentiment_training_results.png
│       └── sentiment_training_table.png
│
├── venv/                           # Virtual environment
├── db.sqlite3
├── requirements.txt
├── .gitignore
└── README.md

```

---

## Installation
(For Python, personally, I use Python 3.10.11)
 Clone the Repository
```
git clone https://github.com/godwyn20/softdsnl-finals.git
cd softdsnl-finals
```

---

 Setup Virtual Environment
```
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

```

---

 Install Dependencies
```
pip install -r requirements.txt

```

---

 Download the Dataset
Download the Flickr8k dataset. Extract it. After extracting, follow this. 
```
data/                          # Rename the folder from 'archive' to 'data'
 ├── images/                   # Change the big letter 'I' to small letter 'i'
 └── captions.txt

```

---

## 🧠 Training the Models

 Train the Image Captioning Model
```
cd model_training
python train_caption_model.py

```

---

 Train the Sentiment Analysis Model
```
python train_sentiment_model.py
```

---

 Output Folder Structure
```
result/
 ├── caption/
 │   ├── caption_model.h5
 │   └── tokenizer.pkl
 └── sentiment/
     ├── sentiment_model.h5
     └── sentiment_tokenizer.pkl
```

---

## Running the Django API
 Apply Migrations
```
python manage.py migrate

```

---

Start the Server
```
python manage.py runserver

```

---


Go to Postman. Your API will be available at:
```
http://127.0.0.1:8000/

```

---

## API Endpoints
| Method | Endpoint                        | Description                                     |
| ------ | ------------------------------- | ----------------------------------------------- |
| `POST` | `/api/predict_caption/`   | Upload an image → Returns **generated caption** |
| `POST` | `/api/predict_sentiment/`  | Send text → Returns **sentiment**               |
| `POST` | `/api/predict_image_sentiment/` | Upload image → Returns **caption + sentiment**  |


---


## Testing with Postman

You’ll use Postman to test the three API endpoints.

 Step 1: Open Postman

- Launch Postman on your computer.
- Make sure your Django server is running at http://127.0.0.1:8000/.

1. Test Image Captioning (/api/predict_caption/)

Goal: Upload an image → get caption.

Setup in Postman:
1. Create a new request.
2. Select POST.
3. Enter URL:
```
http://127.0.0.1:8000/api/image/predict_caption/

```
4. Go to the Body tab → select form-data.
5. Add a new key:
   Key: file
   Type: File
   Value: (choose an image file from your dataset)
6. Click Send.
   
Expected Output:
```
{
  "caption": "a group of people walking on a beach"
}
```

2. Test Sentiment Analysis (/api/text/predict_sentiment/)
Goal: Send a caption → get sentiment prediction.

Setup in Postman:
1. Create another POST request.
2. URL:
```
http://127.0.0.1:8000/api/predict_sentiment/
```
3. Go to Body → raw → JSON.
4. Paste this: (the sentence is just an example)
```
{
  "text": "a group of people walking on a beach"
}

```
5. Click Send.

Expected Output:
```
{
  "text": "a group of people walking on a beach",
  "sentiment": "positive",
  "confidence": 0.8214
}
```

3. Test Combined Model (/api/predict_image_sentiment/)
Goal: Upload image → get both caption + sentiment.

Setup in Postman:
1. Create another POST request.
2. URL:
```
http://127.0.0.1:8000/api/predict_image_sentiment/
```
3. Go to the Body tab → select form-data.
4. Add a new key:
   Key: file
   Type: File
   Value: (choose an image file from your dataset)
5. Click Send.

Expected Output:
```
{
  "caption": "a man riding a skateboard on the street",
  "sentiment": "positive",
  "confidence": 0.9032
}
```
