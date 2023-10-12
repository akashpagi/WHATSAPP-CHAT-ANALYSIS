import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
import plotly.express as px
from wordcloud import STOPWORDS, ImageColorGenerator, WordCloud
import nltk


st.sidebar.title("Whatsapp Group Chat Analysis")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)  

        if df is None:
            st.error("Error occurred during data preprocessing. Please check your file format.")
        else:
            # fetch unique users
            user_list = df['user'].unique().tolist()
            user_list.remove('group_notification')
            user_list.sort()
            user_list.insert(0, "Overall")
            
            selected_user = st.sidebar.selectbox("Show analysis With respect to", user_list)

            if st.sidebar.button("Show Analysis"):
                # Display whatsapp chat table 
                st.title('Whatsapp Chat Table')
                st.dataframe(df)

                # Stats Area
                num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
                st.title("Top Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.subheader("Total messages")
                    st.title(num_messages)
                with col2:
                    st.subheader("Total words")
                    st.title(words)
                with col3:
                    st.subheader("Shared media")
                    st.title(num_media_messages)
                with col4:
                    st.subheader("Shared Links")
                    st.title(num_links)

            

                # Add other analysis sections here...
                
                    # monthly timeline
                st.title("Monthly Timeline")
                # Create a Plotly Express line chart
                timeline = helper.monthly_timeline(selected_user,df)
                fig = px.line(
                    timeline,
                    x='time',
                    y='message',
                    labels={'time': 'Time', 'message': 'Messages'},
                    line_shape='linear',  
                    markers=True,
                    color_discrete_sequence=['green'],  # Change to the desired line color
                    height=500,  # Set the height
                    width=700,
                )

                # Customize the layout
                fig.update_layout(xaxis_tickangle=-45, xaxis_title_text='Time', yaxis_title_text='Messages')

                # Display the plot in Streamlit
                st.plotly_chart(fig)


                #######################################################


                # Daily Timeline
                st.title("Daily Timeline")
                daily_timeline = helper.daily_timeline(selected_user, df)

                # Create a Plotly Express line chart
                fig = px.line(
                    daily_timeline,
                    x='only_date',
                    y='message',
                    labels={'only_date': 'Date', 'message': 'Messages'},
                    line_shape='linear',  
                    markers=True,
                    color_discrete_sequence=['orange'],  # Change to the desired line color
                    height=500, 
                    width=700,
                )
                # Customize the layout
                fig.update_layout(xaxis_tickangle=-45, xaxis_title_text='Date', yaxis_title_text='Messages')
                # Display the plot in Streamlit
                st.plotly_chart(fig)


                #######################################################

                # activity map
                st.title('Activity Map')
                col1,col2 = st.columns([3,2])

                with col1:
                    st.subheader("Most Busy Day")
                    busy_day = helper.week_activity_map(selected_user, df)
                    fig = px.bar(
                        busy_day,
                        x=busy_day.index,
                        y=busy_day.values,
                        labels={'x': 'Day', 'y': 'Messages'},
                        color_continuous_scale='viridis',   # Change to the desired bar color
                        height=310, 
                        width=300
                    )
                    fig.update_layout(xaxis_tickangle=-45, xaxis_title_text='Day', yaxis_title_text='Messages')
                    st.plotly_chart(fig)

                st.markdown("<div style='margin-left: 100px;'></div>", unsafe_allow_html=True)

                with col2:
                    st.subheader("Most Busy Month")
                    busy_month = helper.month_activity_map(selected_user, df)
                    fig = px.bar(
                        busy_month,
                        x=busy_month.index,
                        y=busy_month.values,
                        labels={'x': 'Month', 'y': 'Messages'},
                        color_discrete_sequence=['lightgreen','green'],  # Change to the desired bar color
                        height=310, 
                        width=320,
                    )
                    fig.update_layout(xaxis_tickangle=-45, xaxis_title_text='Month', yaxis_title_text='Messages')
                    st.plotly_chart(fig)
                    
                st.title("Weekly Activity Map (Heatmap)")
                user_heatmap = helper.activity_heatmap(selected_user,df)
                fig,ax = plt.subplots()
                ax = sns.heatmap(user_heatmap, linewidth=1, linecolor='black', square=True, cmap='Blues', cbar_kws={'label': 'Colorbar for Heatmap', 'orientation': 'horizontal'}) #annot=True
                st.pyplot(fig)


                #######################################################

                # finding the busiest users in the group(Group level)
                if selected_user == 'Overall':
                    x,new_df = helper.most_busy_users(df)
                    fig, ax = plt.subplots()
                    plt.xlabel("Group users")
                    plt.ylabel("Messages")
                    
                    #col1, col2 = st.columns(2)
                    col1,col2 = st.columns([2,3])
                    with col1:
                        st.title('Most Busy Users')
                        ax.bar(x.index, x.values,color='skyblue')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    with col2:
                        st.title('Messages Contribution') 
                        st.dataframe(new_df)
                        

                #########################################################


                st.title('WordCloud')
                df_wc = helper.create_wordcloud(selected_user, df)

                if df_wc is not None:
                    # Plot the WordCloud using Matplotlib
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc, interpolation='bilinear')
                    ax.axis('off')  # Turn off the axis

                    # Show the plot in Streamlit
                    st.pyplot(fig)
                else:
                    st.warning("Unable to generate WordCloud. No words found.")


                #######################################
                st.title("Most Common Words")
                # Assuming you have already calculated the most_common_df
                most_common_df = helper.most_common_words(selected_user, df)

                # Check if the DataFrame is not empty
                if not most_common_df.empty:
                    # Print column names
                    print(most_common_df.columns)

                    # Reset index if needed
                    most_common_df = most_common_df.reset_index()

                    fig = px.bar(
                    most_common_df, 
                    x='index',  # Use the correct column name for the x-axis
                    y=0,   # Use the correct column name for the y-axis
                    orientation='h',  # Horizontal bar chart
                    labels={'index': 'Messages (words) count', '0': 'Common words in chat'},
                    )



                    # Show the plot in Streamlit
                    st.plotly_chart(fig)
                else:
                    # Display a warning message if the DataFrame is empty
                    st.warning("No data available to plot !")



                ##########################################################   


                # Sentiment Dataframe
                st.title('Sentiment Analysis')
                sentiments = helper.sentiment_table(selected_user,df)
                st.dataframe(sentiments)

                # Overall sentiment
                st.title('Sentiment Analysis')
                s = helper.overall_sentiment(selected_user,df)
                st.write(s)

                col1, col2 = st.columns([4,4])
                with col1:
                    st.subheader('Overall Sentiment Analysis')
                    fig = px.bar(
                        s, 
                        x='Sentiment', 
                        y='Percentage', 
                        color='Sentiment',
                        labels={'Percentage': 'Percentage (%)'},
                        height=300, 
                        width=300,
                        color_discrete_map={'Positive': 'lightgreen', 'Negative': 'red', 'Neutral': 'blue'}
                    )
                    st.plotly_chart(fig)

                st.markdown("<div style='margin-left: 100px;'></div>", unsafe_allow_html=True)

                with col2:
                    # Sentiment Pie Chart 
                    st.subheader('Sentiment Pie Chart')
                    fig = px.pie(
                        s,
                        names='Sentiment',
                        values='Percentage',
                        color='Sentiment',
                        labels={'Percentage': 'Percentage (%)'},
                        height=300, 
                        width=300,
                        color_discrete_map={'Positive': 'lightgreen', 'Negative': 'red', 'Neutral': 'blue'},
                    )
                    st.plotly_chart(fig)





    #######################################################


            
    except UnicodeDecodeError:
        # Display a warning message if a UnicodeDecodeError occurs
        st.warning(f"Invalid File Format: Please check the file format.")
    except ValueError:
        # Display a warning message if a UnicodeDecodeError occurs
        st.warning(f"Invalid File Format: Please check the file format.")
    except NameError as ne:
        # Handle the NameError
        st.warning("Please try again or check your file.")









