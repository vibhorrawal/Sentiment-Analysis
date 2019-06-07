# Sentiment-Analysis
Use *main.py* to run your query.
# Setup
You can create Virtual Environment using <br>
`virtualenv -p python3 my_env` 
`source my_env/bin/activate`
<br>
Install requirements by running
<br>
`pip install -r requirements.txt`
<br>
`python -m spacy download en_core_web_sm`
<br>
For checking sentiments pertaining to a given topic or a hashtag or even a user on **Twitter**,
 you can use any query. eg- *#Avengers*, *@twitter* or even *'Europe elections'*
<br>	To run *<your-query>* :

`python main.py -q "<your-query>"` or
`python main.py -query "<your-query>"`