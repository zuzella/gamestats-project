# GameStats ⚽

GameStats is a football match winner prediction app built with Python, Streamlit and Machine Learning.

## Features
- Select two national teams
- ML-powered prediction using Random Forest Classifier
- Recent form, goals per game & weighted Head-to-Head analysis
- Deployed on AWS EC2

## Tech Stack
- Python
- Pandas
- Streamlit
- scikit-learn (Random Forest)
- AWS EC2 (Amazon Linux 2023)

## Dataset
International football results dataset from Kaggle.

## Live Demo
http://16.170.173.180:8501

## Run locally
```bash
pip3 install -r requirements.txt
python3 train_model.py
streamlit run app.py
```

## Deploy on EC2
```bash
ssh -i "your-key.pem" ec2-user@your-ec2-ip
pip3 install streamlit pandas scikit-learn numpy
python3 train_model.py
python3 -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
```
