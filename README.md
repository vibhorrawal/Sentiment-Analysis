# Sentiment-Analysis
Use *main.py* to run your query.
# Setup
You can create Virtual Environment using: <br>
`python3 -m venv my_env` <br>
`source my_env/bin/activate`
<br><br>
Install requirements by running
<br>
`pip3 install -r requirements.txt`
<br>
`python3 -m spacy download en_core_web_sm`
<br>
# Running main.py
For checking sentiments pertaining to a given topic or a hashtag or even a user on **Twitter**, <br>
 you can use any query. eg- *#Avengers*, *@twitter* or even *'Europe elections'*
<br><br>	To run *\<your-query\>* :<br>
`python3 main.py -q "<your-query>"` or
`python3 main.py -query "<your-query>"`
<br><br>
# Further Description
Following models, currently, are used and their votes are averaged to produce final sentiment score with confidence(out of 1) in the form of [Negative, Postive] :
- LSTM (RNN architecture)
- Naive Bayes classifier for Multinomial model
- Naive Bayes classifier for multivariate Bernoulli model
- Gaussian Naive Bayes
- Logistic Regression
- C-Support Vector Classification
- Linear Support Vector Classification
- Nu-Support Vector Classification.
<br>
After finding some free corpuses(corpi?) online, the models are trained. If you want to train yourself, there are some corpi in <a href="training-data">training-data</a>, you can add your files in that folder and/or change <a href="train.py">train.py</a> according to your data and train for weights which are stored in <a href="weights">weights</a>.
<br><br>Note: 
After some testing I found LSTM, SVC and NuSVC to perform much better than others as is also visible by the size of their weights, and therefore if no flag to include all models is passed, by default those are used for predictions. If you want to include all, run with --all flag, as in
`python3 main.py --all` 
. I have still included them, if you find more data, training with it will improve the model, also if you fine tune hyperparameters that will help too, kindly let me know also.
