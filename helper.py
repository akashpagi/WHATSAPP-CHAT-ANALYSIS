from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user, df):
    try:
        f = open('stop_hinglish.txt', 'r')
        stop_words = f.read()

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        temp = df[df['user'] != 'group_notification']
        temp = temp[temp['message'] != '<Media omitted>\n']

        def remove_stop_words(message):
            y = []
            for word in message.lower().split():
                if word not in stop_words:
                    y.append(word)
            return " ".join(y)

        wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
        temp['message'] = temp['message'].apply(remove_stop_words)
        df_wc = wc.generate(temp['message'].str.cat(sep=" "))
        return df_wc

    except Exception as e:
        print(f"Warning in create_wordcloud: {str(e)}")
        


def most_common_words(selected_user,df):
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def sentiment_table(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    # Download the VADER lexicon
    nltk.download('vader_lexicon')
    s_df = pd.DataFrame(df, columns=["date", "year", "user", "message"])

    sentiments = SentimentIntensityAnalyzer()
    s_df["positive"] = [sentiments.polarity_scores(i)["pos"] for i in s_df["message"]]
    s_df["negative"] = [sentiments.polarity_scores(i)["neg"] for i in s_df["message"]]
    s_df["neutral"] = [sentiments.polarity_scores(i)["neu"] for i in s_df["message"]]

    # Return the DataFrame
    return s_df

def overall_sentiment(selected_user, df):
    # Call the sentiment_table function with appropriate arguments
    s_df = sentiment_table(selected_user, df)

    x = sum(s_df["positive"])
    y = sum(s_df["negative"])
    z = sum(s_df["neutral"])

    # Calculate percentages
    total = x + y + z
    percentage_positive = (x / total) * 100
    percentage_negative = (y / total) * 100
    percentage_neutral = (z / total) * 100

    # Create a percentage DataFrame
    percentage_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Percentage': [percentage_positive, percentage_negative, percentage_neutral]
    })

    return percentage_df



    



    












