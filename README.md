
# 📚 Book Recommendation System

This Streamlit app recommends books using both **content-based filtering** (book titles) and **collaborative filtering** (user ratings). It was built from a Jupyter Notebook and deployed using Streamlit Cloud.

## 🚀 Features
- Recommend similar books based on book title (TF-IDF)
- Suggest personalized books for users using collaborative filtering
- Simple, interactive UI powered by Streamlit

## 📁 Folder Structure

```
book-recommender/
├── app.py               # Streamlit application
├── Books.csv            # Dataset of books
├── Ratings.csv          # User ratings
├── requirements.txt     # Required Python packages
├── README.md            # Project documentation
```

## 🔧 Requirements

Install dependencies locally:

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

```bash
streamlit run app.py
```

Then go to `http://localhost:8501` in your browser.

## 🌐 Deploy on Streamlit Cloud

1. Push this project to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app, select the repo, and point it to `app.py`
4. Done!

## 📬 Contact

Built by [Your Name]. Feel free to reach out for collaboration or suggestions!

---

**Demo:** _Add your Streamlit Cloud URL here once deployed._
