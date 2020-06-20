# Importing essential libraries
from flask import Flask, render_template, request, jsonify
import pickle
import re

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'restaurant-sentiment-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))
#from nltk.stem.porter import PorterStemmer
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

@app.route('/predictor',methods=['POST'])
def predictor():
    st_array=list(request.get_json(force=True).values())[0]
    corpus=[]
    for st in st_array:
        review=re.sub('[^a-zA-z]',' ',st)
        review=review.lower()
        review=review.split()
        #ps=PorterStemmer()
        review = [word for word in review if not word in ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]]
        review=' '.join(review)
        corpus.append(review)
    trail_x=cv.transform(corpus).toarray()
    trail_pred=classifier.predict(trail_x)
    lst=[]
    for i in trail_pred:
        lst.append(int(i))
    return jsonify(outputs=lst)
    #return(trail_pred)
if __name__ == '__main__':
	app.run(debug=True)
