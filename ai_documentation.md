# AI Documentation

## Case 1: Circular logic in ML model
AI trained the model on home_score & away_score to predict the winner — 
but these are the actual match scores, which you don't know before the game. 
We fixed it by retraining the model using meaningful features: 
recent form, goals per game, and weighted Head-to-Head record.

## Case 2: Wrong features in model.pkl
app.py was loading the old model.pkl trained with wrong features, 
causing a ValueError on prediction. 
We deleted the old model.pkl and ran the new train_model.py 
to generate a correct model.

## Case 3: Streamlit not found on EC2
After deploying on EC2, running `streamlit run app.py` 
returned "command not found" error. 
We fixed it by using `python3 -m streamlit run app.py` instead.
