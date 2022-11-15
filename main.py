from textblob import TextBlob       # TextBlob để phân tích sentiment
import pandas as pd                 # Pandas để đọc dữ liệu 
import streamlit as st              # Streamlit để deploy lên localhost     
import cleantext                    # Cleantext để lọc câu
import seaborn as sns               # Seaborn được sử dụng để giảm bớt nhiệm vụ trực quan hóa dữ liệu
import matplotlib.pyplot as plt     # Matplotlib để vẽ biểu đồ
from wordcloud import WordCloud





st.header('Sentiment Analysis')        
with st.expander('Analyze Text'):      
    text = st.text_input('Text here: ') 
    if text:
        blob = TextBlob(text)           
        st.write('Polarity: ', round(blob.sentiment.polarity,2)) 

    pre = st.text_input('Clean Text: ') 
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,     
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True))
        
        

with st.expander('Analyze CSV'):        
    uploadfile = st.file_uploader('Upload file')

    def score(x):                       
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

#
    def analyze(x):                     
        if x >= 0.1:                   
            return 'Positive'
        elif x <= -0.1:                
            return 'Negative'
        else:
            return 'Neutral'  

                  

#
    if uploadfile:
        df = pd.read_csv(uploadfile)    

        df['score'] = df['reviews'].apply(score)    
                                                    
                                                    
        df['analysis'] = df['score'].apply(analyze)
                                                    
                                                    
        st.write(df.head(10))         
                     
        
        most_positive = df[df.score > 0.8].reviews.head(10)  
        most_negative = df[df.score < -0.8].reviews.head(10) 
        
        posi = ' '.join([twts for twts in df[df.score > 0.8].reviews.head(10)])
        st.set_option('deprecation.showPyplotGlobalUse', False)

       
        nega = ' '.join([twts for twts in df[df.score < -0.8].reviews.head(10)])
        st.set_option('deprecation.showPyplotGlobalUse', False)

       
       
        
        option = ["most negative","most positive"]          
        option_selected = st.selectbox("Select Most Negative or Most Positive", 
                                       options= option)                         
        if option_selected == "most negative":   
            st.write(most_negative) 
            wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(nega)
            plt.imshow(wordCloud, interpolation="bilinear")
            plt.axis('off')
            plt.show()
            st.pyplot()
            
                         
        elif option_selected == "most positive": 
            st.write(most_positive) 
            wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(posi)
            plt.imshow(wordCloud, interpolation="bilinear")
            plt.axis('off')
            plt.show()
            st.pyplot()     
            
        
        new_df = df['analysis'].value_counts().rename_axis('sub_cat_values').reset_index(name='counts')
        total_of_counts = new_df.counts.sum()
        your_lables = new_df.sub_cat_values
        your_values = new_df.counts
        your_explode = [0.1, 0, 0] # you can create this input from datafram or list
        # visualize the pie chart 
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.axis('equal')
        ax.pie(your_values, labels = your_lables, explode=your_explode,textprops={'fontsize': 14})
        plt.show()
        st.pyplot()   

# autopct function  returns list of  strings used to label the wedge with their numerical value.
# explode - list the margin in the seqance of data
        
                  

        figg = plt.figure(figsize=(10, 7))
        plt.title("Sentiment Analysis", ) 
        plt.xlabel('Sentiment')
        plt.ylabel('Counts')
        df['analysis'].value_counts().plot(kind = 'bar')
        plt.show()
        st.pyplot(figg)
        

        

        fig = plt.figure(figsize=(19, 5))   
        sns.histplot(data=df, x="score")    
        plt.title("Sentiment Analysis")     
        st.pyplot(fig)
        
        



        
        

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(                 #nút download 
            label="Download data as CSV",   #tên nút 
            data=csv,                     
            file_name='sentiment.csv',      #tải về với tên sentiment.csv
            mime='text/csv',
        )


