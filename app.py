from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import pandas as pd
from joblib import load
import seaborn as sns
import io
from wordcloud import WordCloud
import base64
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from google_play_scraper import app, Sort, reviews_all
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
from matplotlib.sankey import Sankey
import networkx as nx

app = Flask(__name__)

def preprocess_text(text):
    if text is not None:
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back into string
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    else:
        return ''

def preprocess_dataframe(df):
    # Drop unnecessary columns
    df.drop(['userName', 'reviewId', 'userImage', 'reviewCreatedVersion', 'at'], axis=1, inplace=True)
    
    # Convert 'repliedAt' column to datetime
    df['repliedAt'] = pd.to_datetime(df['repliedAt'])
    
    # Extract month and year from 'repliedAt'
    df['RepliedMonth'] = df['repliedAt'].dt.month
    df['RepliedYear'] = df['repliedAt'].dt.year
    
    # Drop the original 'repliedAt' column
    df.drop('repliedAt', axis=1, inplace=True)
    
    # Convert 'replyContent' to binary indicator
    df['IsReplied'] = df['replyContent'].apply(lambda x: 'Yes' if x and x.strip() != '' else 'No')

    # Drop 'replyContent' column
    df.drop('replyContent', axis=1, inplace=True)

    # Fill missing values in 'appVersion' with '0'
    df['appVersion'].fillna('0', inplace=True)
    
    # Only keep necessary columns (content, score, IsReplied)
    df = df[['content', 'score', 'IsReplied']]
    
    return df

def analyze_sentiment(text, score):
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    # Perform sentiment analysis
    sentiment_score = analyzer.polarity_scores(text)['compound']
    
    if sentiment_score >= 0.05 and score >= 3:
        return 'positive'
    elif sentiment_score <= -0.05 and score < 3:
        return 'negative'
    else:
        return 'neutral'

@app.route('/predict/app', methods=['POST'])
def predict_appFraud():
    # Get the app ID and other necessary data from the form
    app_id = request.form['app-id']
    app_name = request.form['app-name']
   
    # Scrape reviews for the specified app
    reviews = reviews_all(app_id, sleep_milliseconds=0, lang="Eng", country="in", sort=Sort.NEWEST)
    df = pd.json_normalize(reviews)

    # Preprocess the DataFrame
    df = preprocess_dataframe(df)

    # Perform sentiment analysis
    df['sentiment'] = df.apply(lambda row: analyze_sentiment(row['content'], row['score']), axis=1)
    # Generate result based on sentiment
    positive_count = (df['sentiment'] == 'positive').sum()
    negative_count = (df['sentiment'] == 'negative').sum()

    if positive_count > negative_count:
        result = "The App is Not Fraud"
    else:
        result = "The App is Fraud"
    
    total_reviews = len(df)
    positive_reviews = (df['sentiment'] == 'positive').sum()
    negative_reviews = (df['sentiment'] == 'negative').sum()
    neutral_reviews = (df['sentiment'] == 'neutral').sum()
    average_rating = round(df['score'].mean(), 2)
    positive_percentage = round((positive_reviews / total_reviews) * 100, 2)
    negative_percentage = round((negative_reviews / total_reviews) * 100, 2)
    neutral_percentage = round((neutral_reviews / total_reviews) * 100, 2)
    replied_percentage = round((df['IsReplied'] == 'Yes').mean() * 100, 2)

    # Generate visualizations
    # 1. Percentage pie chart of reviews
    reviews_counts = df['sentiment'].value_counts()
    labels = reviews_counts.index
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(6, 4))
    plt.pie(reviews_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of Reviews in Fraud App')
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    buffer_data1 = base64.b64encode(buffer1.getvalue()).decode()
    plt.close()

    # 2. Count plot of each type of review
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sentiment', data=df, palette={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
    plt.title('Count of Each Review Type in Fraud App')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    buffer_data2 = base64.b64encode(buffer2.getvalue()).decode()
    plt.close()

    # 3. Histogram for each type of score
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='score', hue='sentiment', multiple='stack', bins=20)
    plt.title('Histogram of Rating for Each Review Type in Fraud App')
    plt.xlabel('Score')
    plt.ylabel('Count')
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    buffer_data3 = base64.b64encode(buffer3.getvalue()).decode()
    plt.close()

    # 4. Pie chart of isreplied (Yes vs No)
    replied_counts = df['IsReplied'].value_counts()
    labels = replied_counts.index
    plt.figure(figsize=(6, 4))
    plt.pie(replied_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral'])
    plt.title('Percentage of Replies in Fraud App Reviews')
    buffer4 = io.BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    buffer_data4 = base64.b64encode(buffer4.getvalue()).decode()
    plt.close()

    # 5. Violin plot of review vs score
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='sentiment', y='score', data=df, palette={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
    plt.title('Violin Plot of Review vs Rating in Fraud App')
    plt.xlabel('Sentiment')
    plt.ylabel('Score')
    buffer5 = io.BytesIO()
    plt.savefig(buffer5, format='png')
    buffer5.seek(0)
    buffer_data5 = base64.b64encode(buffer5.getvalue()).decode()
    plt.close()

    # 6. Joint count plot for positive, negative, and neutral reviews based on isreplied (Yes or No)
    plt.figure(figsize=(6, 4))  # Set the size of the figure
    sns.catplot(x='sentiment', kind='count', hue='IsReplied', data=df, palette='Set1',height=4,aspect=1)
    plt.title('Sentiments vs Review Reply Status')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    buffer6 = io.BytesIO()
    plt.savefig(buffer6, format='png')
    buffer6.seek(0)
    buffer_data6 = base64.b64encode(buffer6.getvalue()).decode()
    plt.close()

    # Render template with result and any other data you want to display
    return render_template('app_result.html', result=result, app_name=app_name,
                           total_reviews=total_reviews, positive_reviews=positive_reviews,
                           negative_reviews=negative_reviews, neutral_reviews=neutral_reviews,
                           average_rating=average_rating, positive_percentage=positive_percentage,
                           negative_percentage=negative_percentage, neutral_percentage=neutral_percentage, replied_percentage=replied_percentage, plot1=buffer_data1, plot2=buffer_data2,
                           plot3=buffer_data3, plot4=buffer_data4, plot5=buffer_data5, plot6=buffer_data6)

# Load the pre-trained model
best_rf_classifier = load('RFModel.pkl')

# Load X_train
X_train = pd.read_csv('X_train.csv')

# Load the dataset
df = pd.read_csv('DVCarFraudDetection.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vehicle_insurance')
def vehicle_insurance():
    return render_template('vehicle.html')

@app.route('/predict/insurance')
def predict_insurance():
    return render_template('vehicle.html')

@app.route('/dataset')
def dataset_display():
    # Generate visualizations
    fig1, ax1 = plt.subplots(figsize=(6, 4))  
    sns.countplot(y='CarCompany', data=df)
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    buffer_data1 = base64.b64encode(buffer1.getvalue()).decode()
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))  
    sns.countplot(x='BasePolicy', hue='IsFraud', data=df, palette={0: 'green', 1: 'red'})
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    buffer_data2 = base64.b64encode(buffer2.getvalue()).decode()
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))  
    past_claims_counts = df['PastNumberOfClaims'].value_counts()
    ax3.pie(past_claims_counts, labels=past_claims_counts.index, autopct='%1.1f%%')
    ax3.set_title('Past Number of Claims Count')
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    buffer_data3 = base64.b64encode(buffer3.getvalue()).decode()
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(6, 4))  # Adjust the figsize as per your preference
    sns.countplot(x='IsAddressChanged', hue='IsFraud', data=df, palette={0: 'green', 1: 'red'})
    ax4.set_title('Address Change and Fraud Distribution')
    ax4.set_xlabel('Is Address Changed?')
    ax4.set_ylabel('Count')
    plt.legend(title='Is Fraud')
    buffer4 = io.BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    buffer_data4 = base64.b64encode(buffer4.getvalue()).decode()
    plt.close(fig4)
    
    fig5, ax5 = plt.subplots(figsize=(6, 4))  # Adjust the figsize as per your preference
    heatmap_data = df.groupby(['CarCompany', 'OwnerGender']).size().unstack()
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', ax=ax5)
    ax5.set_title('Car Company vs Owner Gender')
    ax5.set_xlabel('Owner Gender')
    ax5.set_ylabel('Car Company')
    plt.yticks(rotation=0)  # Rotate y-axis labels for better readability
    plt.tight_layout()
    buffer5 = io.BytesIO()
    plt.savefig(buffer5, format='png')
    buffer5.seek(0)
    buffer_data5 = base64.b64encode(buffer5.getvalue()).decode()
    plt.close(fig5)

    fig6, ax6 = plt.subplots(figsize=(6, 4))  
    num_supplements_counts = df['NumberOfSuppliments'].value_counts()
    ax6.pie(num_supplements_counts, labels=num_supplements_counts.index, autopct='%1.1f%%')
    ax6.set_title('NUmber of Suplements Count')
    buffer6 = io.BytesIO()
    plt.savefig(buffer6, format='png')
    buffer6.seek(0)
    buffer_data6 = base64.b64encode(buffer6.getvalue()).decode()
    plt.close(fig6)


    fig7, ax7 = plt.subplots(figsize=(6, 4))  
    sns.countplot(x='PoliceReportFiled', hue='IsFraud', data=df)
    buffer7 = io.BytesIO()
    plt.savefig(buffer7, format='png')
    buffer7.seek(0)
    buffer_data7 = base64.b64encode(buffer7.getvalue()).decode()
    plt.close(fig7)

    fig8, ax8 = plt.subplots(figsize=(6, 4))  
    sns.violinplot(x='OwnerGender', y='OwnerAge', data=df, palette={'Male': 'blue', 'Female': 'pink'}, ax=ax8)
    buffer8 = io.BytesIO()
    plt.savefig(buffer8, format='png')
    buffer8.seek(0)
    buffer_data8 = base64.b64encode(buffer8.getvalue()).decode()
    plt.close(fig8)

    fig9, ax9 = plt.subplots(figsize=(6, 4))  # Create a new figure and axis
    sns.scatterplot(x='OwnerAge', y='NumberOfSuppliments', data=df, ax=ax9)
    plt.title('Scatter Plot of OwnerAge vs NumberOfSuppliments')  # Set the title of the plot
    plt.tight_layout()  # Ensure tight layout
    buffer9 = io.BytesIO()  # Create a BytesIO buffer to store the plot image
    plt.savefig(buffer9, format='png')  # Save the plot to the buffer in PNG format
    buffer9.seek(0)  # Reset the buffer position to the start
    buffer_data9 = base64.b64encode(buffer9.getvalue()).decode()  # Encode the plot image as base64
    plt.close(fig9)  # Close the figure to release resources


    fig10, ax10 = plt.subplots(figsize=(6, 4))  
    sns.boxplot(x='CarCategory', y='CarPrice', data=df, ax=ax10)
    buffer10 = io.BytesIO()
    plt.savefig(buffer10, format='png')
    buffer10.seek(0)
    buffer_data10 = base64.b64encode(buffer10.getvalue()).decode()
    plt.close(fig10)


    # Render the dataset template with plots
    return render_template('dataset.html', df=pd.read_csv('env\DVCarFraudDetection.csv'), plot1=buffer_data1, plot2=buffer_data2,
                           plot3=buffer_data3, plot4=buffer_data4, plot5=buffer_data5, plot6=buffer_data6,
                           plot7=buffer_data7, plot8=buffer_data8, plot9=buffer_data9, plot10=buffer_data10)


@app.route('/predict/insurance', methods=['POST'])
def make_prediction():
    # Get the form data
    CarCompany = request.form['CarCompany']
    AccidentArea = request.form['AccidentArea']
    OwnerGender = request.form['OwnerGender']
    OwnerAge = int(request.form['OwnerAge'])
    Fault = request.form['Fault']
    CarCategory = request.form['CarCategory']
    CarPrice = int(request.form['CarPrice'])
    PoliceReportFiled = request.form['PoliceReportFiled']
    WitnessPresent = request.form['WitnessPresent']
    AgentType = request.form['AgentType']
    NumberOfSuppliments = int(request.form['NumberOfSuppliments'])
    BasePolicy = request.form['BasePolicy']
    IsAddressChanged = request.form['IsAddressChanged']
    PastNumberOfClaims = int(request.form['PastNumberOfClaims'])

    # Preprocess the input data
    car_price = CarPrice / 10  # scaling car price as in your previous code
    user_input = {
        'CarCompany': [CarCompany],
        'AccidentArea': [AccidentArea],
        'OwnerGender': [OwnerGender],
        'OwnerAge': [OwnerAge],
        'Fault': [Fault],
        'CarCategory': [CarCategory],
        'CarPrice': [car_price],
        'PoliceReportFiled': [PoliceReportFiled],
        'WitnessPresent': [WitnessPresent],
        'AgentType': [AgentType],
        'NumberOfSuppliments': [NumberOfSuppliments],
        'BasePolicy': [BasePolicy],
        'IsAddressChanged': [IsAddressChanged],
        'PastNumberOfClaims': [PastNumberOfClaims]
    }
    user_df = pd.DataFrame(user_input)
    processed_user_input = pd.get_dummies(user_df)
    # Assuming X_train is your training data, you need to replace it with your actual training data
    processed_user_input = processed_user_input.reindex(columns=X_train.columns, fill_value=0)

    # Make prediction
    prediction = best_rf_classifier.predict(processed_user_input)

    # Return prediction result
    if prediction[0] == 1:
        result = "Fraud in Insurance"
    else:
        result = "No Fraud in Insurance"

    # Generate visualizations
    fig1, ax1 = plt.subplots(figsize=(6, 4))  # Adjust the figsize as per your preference
    sns.countplot(x='OwnerGender', hue='IsFraud', data=df, ax=ax1)
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    buffer_data1 = base64.b64encode(buffer1.getvalue()).decode()
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))  # Adjust the figsize as per your preference
    sns.violinplot(x='CarCategory', y='CarPrice', data=df, ax=ax2)
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    buffer_data2 = base64.b64encode(buffer2.getvalue()).decode()
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))  # Adjust the figsize as per your preference
    sns.countplot(x='AgentType', hue='IsFraud', data=df, ax=ax3)
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    buffer_data3 = base64.b64encode(buffer3.getvalue()).decode()
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(6 , 4))  # Adjust the figsize as per your preference
    policy_fraud_counts = df[df['IsFraud'] == 1]['BasePolicy'].value_counts()
    ax4.pie(policy_fraud_counts, labels=policy_fraud_counts.index, autopct='%1.1f%%')
    buffer4 = io.BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    buffer_data4 = base64.b64encode(buffer4.getvalue()).decode()
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    fraud_data = df[df['IsFraud'] == 1]
    non_fraud_data = df[df['IsFraud'] == 0]
    sns.boxplot(x='IsFraud', y='CarPrice', data=fraud_data, ax=ax5)
    sns.boxplot(x='IsFraud', y='CarPrice', data=non_fraud_data, ax=ax5)
    ax5.set_xlabel('Fraud Status')
    ax5.set_ylabel('Car Price')
    ax5.set_title('Box Plot of Car Price for Fraud and Non-Fraud Cases')
    handles, labels = ax5.get_legend_handles_labels()
    ax5.legend(handles, labels)
    buffer5 = io.BytesIO()
    plt.savefig(buffer5, format='png')
    buffer5.seek(0)
    buffer_data5 = base64.b64encode(buffer5.getvalue()).decode()
    plt.close(fig5)
      
    fig6, ax6 = plt.subplots(figsize=(6, 4))  # Adjust the figsize as per your preference
    sns.histplot(data=df, x='PastNumberOfClaims', bins=range(max(df['PastNumberOfClaims'])+2), kde=False, ax=ax6)
    ax6.set_ylabel('Fraud cases count')
    buffer6 = io.BytesIO()
    plt.savefig(buffer6, format='png')
    buffer6.seek(0)
    buffer_data6 = base64.b64encode(buffer6.getvalue()).decode()
    plt.close(fig6)

    fig7, ax7 = plt.subplots(figsize=(6, 4))  # Adjust the figsize as per your preference
    policy_fraud_counts = df[df['IsFraud'] == 1]['CarCategory'].value_counts()
    ax7.pie(policy_fraud_counts, labels=policy_fraud_counts.index, autopct='%1.1f%%')
    buffer7 = io.BytesIO()
    plt.savefig(buffer7, format='png')
    buffer7.seek(0)
    buffer_data7 = base64.b64encode(buffer7.getvalue()).decode()
    plt.close(fig7)

    fig8, ax8 = plt.subplots(figsize=(6, 4))  # Adjust the figsize as per your preference
    sns.countplot(x='PastNumberOfClaims', hue='IsFraud', data=df, ax=ax8)
    buffer8 = io.BytesIO()
    plt.savefig(buffer8, format='png')
    buffer8.seek(0)
    buffer_data8 = base64.b64encode(buffer8.getvalue()).decode()
    plt.close(fig8)
    
    # Return prediction result and base64 encoded images
    return render_template('prediction_result.html', result=result, plot1=buffer_data1, plot2=buffer_data2,
                           plot3=buffer_data3, plot4=buffer_data4, plot5=buffer_data5, plot6=buffer_data6,
                           plot7=buffer_data7, plot8=buffer_data8)
    
@app.route("/predict/app")
def predict_app():
    return render_template('fraudapp.html')


@app.route("/mobile_app")
def mobile_app():
    return render_template('fraudapp.html')


@app.route('/analysis/app')
def analysis_app():
    return render_template('app_analysis.html')

@app.route('/analysis/app', methods=['POST'])
def analysisresult_app():
    app_id = request.form['app-id']
    app_name = request.form['app-name']
   
    # Scrape reviews for the specified app
    reviews = reviews_all(app_id, sleep_milliseconds=0, lang="Eng", country="in", sort=Sort.NEWEST)
    df = pd.json_normalize(reviews)

    # Preprocess the DataFrame
    df = preprocess_dataframe(df)

    # Perform sentiment analysis
    df['sentiment'] = df.apply(lambda row: analyze_sentiment(row['content'], row['score']), axis=1)

    # Word Cloud
    text = ' '.join(df['content'].astype(str).tolist())
    wordcloud = WordCloud(width=600, height=400, background_color='white').generate(text)
    img_buffer1 = save_wordcloud_to_buffer(wordcloud)

    stop_words = set(stopwords.words('english'))

    # Add more words if necessary
    additional_stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])
    stop_words.update(additional_stopwords)

    # Count Plot of 10 Most Repeated Proper Nouns
    proper_nouns = []
    for review in df['content']:
        words = review.split()
        for word in words:
            if word.istitle() and word.isalpha() and word.lower() not in stop_words:
                proper_nouns.append(word)
    top_proper_nouns = Counter(proper_nouns).most_common(10)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.countplot(y=proper_nouns, order=[word[0] for word in top_proper_nouns], palette='viridis', ax=ax2)
    ax2.set_title('Count Plot of 10 Most Repeated Proper Nouns')
    ax2.set_xlabel('Count')
    buffer2 = save_plot_to_buffer(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    is_replied_no_df = df[df['IsReplied'] == 'No']
    sentiment_counts = is_replied_no_df['sentiment'].value_counts()
    ax3.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red', 'blue'])
    ax3.set_title('Pie Chart of Sentiment Distribution for IsReplied NO')
    buffer3 = save_plot_to_buffer(fig3)

    # Calculate Review Length
    df['review_length'] = df['content'].apply(lambda x: len(x.split()))
    
    # Create a pivot table to aggregate sentiment scores by review length
    sentiment_distribution = df.pivot_table(index='review_length', columns='sentiment', values='score', aggfunc='mean')

    # Plot the heatmap
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.heatmap(sentiment_distribution, cmap='YlGnBu', linewidths=0.5, ax=ax4)
    ax4.set_title('Sentiment Distribution Heatmap')
    ax4.set_xlabel('Sentiment')
    ax4.set_ylabel('Review Length')

    # Save the plot to buffer
    buffer4 = save_plot_to_buffer(fig4)

    # Heatmap of Word Frequency
    word_lengths = df['content'].apply(lambda x: len(x.split()))
    word_freq = pd.DataFrame({'Word Length': word_lengths, 'Rating': df['score']})

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.heatmap(word_freq.corr(), annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title('Heatmap of Word Length vs Rating')
    buffer5 = save_plot_to_buffer(fig5)

    # Joint Count Plot of Score for Positive, Negative, and Neutral
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x='score', hue='sentiment', multiple='stack', palette='husl', ax=ax6)
    ax6.set_title('Joint Count Plot of Score for Positive, Negative, and Neutral')
    ax6.set_xlabel('Score')
    ax6.set_ylabel('Count')
    buffer6 = save_plot_to_buffer(fig6)

    return render_template('app_analysis_final.html', df=df, app_name=app_name,
                           buffer1=img_buffer1, buffer2=buffer2, buffer3=buffer3,
                           buffer4=buffer4, buffer5=buffer5, buffer6=buffer6)


# Function to save plot to buffer
def save_plot_to_buffer(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    buffer_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return buffer_data

# Function to save WordCloud image to buffer
def save_wordcloud_to_buffer(wordcloud):
    img = wordcloud.to_image()
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    buffer = base64.b64encode(img_buffer.getvalue()).decode()
    img_buffer.close()
    return buffer


@app.route('/analysis/insurance')
def analysis_insurance():
    # Generate visualizations
    # Visualization 1: Distribution of Car Prices
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(df['CarPrice'], kde=True, color='skyblue', ax=ax1)
    ax1.set_title('Distribution of Car Prices')
    ax1.set_xlabel('Car Price')
    ax1.set_ylabel('Frequency')
    buffer1 = save_plot_to_buffer(fig1)

    # Visualization 2: Distribution of Owner Ages
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(df['OwnerAge'], kde=True, color='salmon', ax=ax2)
    ax2.set_title('Distribution of Owner Ages')
    ax2.set_xlabel('Owner Age')
    ax2.set_ylabel('Frequency')
    buffer2 = save_plot_to_buffer(fig2)

    # Visualization 3: Count of Claims by Base Policy
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='CarCategory', hue='IsFraud', data=df, palette='coolwarm', ax=ax3)
    ax3.set_title('Count of Claims by Car category')
    ax3.set_xlabel('Car category')
    ax3.set_ylabel('Count')
    buffer3 = save_plot_to_buffer(fig3)

    # Visualization 4: Distribution of Car Prices by Fraud Status
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='IsFraud', y='CarPrice', data=df, palette='Set2', ax=ax4)
    ax4.set_title('Distribution of Car Prices by Fraud Status')
    ax4.set_xlabel('Fraud Status')
    ax4.set_ylabel('Car Price')
    buffer4 = save_plot_to_buffer(fig4)

    # Visualization 5: Count of Claims by Accident Area
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='AccidentArea', hue='IsFraud', data=df, palette='husl', ax=ax5)
    ax5.set_title('Count of Claims by Accident Area')
    ax5.set_xlabel('Accident Area')
    ax5.set_ylabel('Count')
    buffer5 = save_plot_to_buffer(fig5)

    # Visualization 6: Distribution of Number of Supplements
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.histplot(df['NumberOfSuppliments'], kde=True, color='orange', ax=ax6)
    ax6.set_title('Distribution of Number of Supplements')
    ax6.set_xlabel('Number of Supplements')
    ax6.set_ylabel('Frequency')
    buffer6 = save_plot_to_buffer(fig6)

    # Visualization 7: Count of Claims by Witness Presence
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='WitnessPresent', hue='IsFraud', data=df, palette='viridis', ax=ax7)
    ax7.set_title('Count of Claims by Witness Presence')
    ax7.set_xlabel('Witness Presence')
    ax7.set_ylabel('Count')
    buffer7 = save_plot_to_buffer(fig7)

    # Visualization 8: Distribution of Past Number of Claims
    fig8, ax8 = plt.subplots(figsize=(6, 4))
    sns.histplot(df['PastNumberOfClaims'], kde=True, color='purple', ax=ax8)
    ax8.set_title('Distribution of Past Number of Claims')
    ax8.set_xlabel('Past Number of Claims')
    ax8.set_ylabel('Frequency')
    buffer8 = save_plot_to_buffer(fig8)
    
    numeric_columns = df.select_dtypes(include='number')

    # Compute the correlation matrix
    corr = numeric_columns.corr()

    # Create the heatmap
    fig9, ax9 = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax9)
    ax9.set_title('Heatmap of Correlation Matrix')
    buffer9 = save_plot_to_buffer(fig9)

    # Visualization 10: Network Graph of Car Brands and Fraud Status
    fig10, ax10 = plt.subplots(figsize=(6, 4))
    G = nx.from_pandas_edgelist(df, 'CarCompany', 'IsFraud')
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, ax=ax10)
    ax10.set_title('Network Graph of Car Brands and Fraud Status')
    buffer10 = save_plot_to_buffer(fig10)

    # Visualization 11: Violin Plot of Accident Area and Car Price
    fig11, ax11 = plt.subplots(figsize=(6, 4))
    sns.violinplot(x='AccidentArea', y='CarPrice', data=df, hue='IsFraud', split=True, palette='husl', ax=ax11)
    ax11.set_title('Violin Plot of Accident Area and Car Price')
    buffer11 = save_plot_to_buffer(fig11)

    fig12, ax12 = plt.subplots(figsize=(6, 4))
    hb = ax12.hexbin(df['CarPrice'], df['OwnerAge'], gridsize=50, cmap='inferno')
    ax12.set_title('Hexbin Plot of Car Prices and Owner Ages')
    ax12.set_xlabel('Car Price')
    ax12.set_ylabel('Owner Age')
    cb = fig12.colorbar(hb, ax=ax12)
    cb.set_label('Frequency')
    buffer12 = save_plot_to_buffer(fig12)

    # Return render template with the additional plots
    return render_template('insurance_analysis.html', plot1=buffer1, plot2=buffer2,
                           plot3=buffer3, plot4=buffer4, plot5=buffer5, plot6=buffer6,
                           plot7=buffer7, plot8=buffer8, plot9=buffer9, plot10=buffer10,
                           plot11=buffer11, plot12=buffer12)


if __name__ == "__main__":
    app.run(debug=True)
    
   
   
