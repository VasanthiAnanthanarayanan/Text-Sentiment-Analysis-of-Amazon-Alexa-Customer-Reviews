install.packages("wordcloud")
library(wordcloud)
install.packages("RColorBrewer")
library(RColorBrewer)
install.packages("wordcloud2")
library(wordcloud2)
install.packages("tm")
library(tm)
install.packages("SnowballC")
library(SnowballC)
install.packages("textstem")
library(textstem)
install.packages("SentimentAnalysis")
library(SentimentAnalysis)
install.packages("RSentiment")
library(RSentiment)
install.packages("RWeka")
library(RWeka)

# Sentiment analysis
library(syuzhet)
library(lubridate)
library(dplyr)

#Viewing the data
str(amazon_alexa)
View(amazon_alexa)

#Removing NA's as they are less in number
sum(is.na(amazon_alexa))
amazon_alexa<- na.omit(amazon_alexa)
dim(amazon_alexa)

#Converting data types
amazon_alexa$rating <- as.factor(amazon_alexa$rating)
amazon_alexa$variation <- as.factor(amazon_alexa$variation)
amazon_alexa$feedback <- as.factor(amazon_alexa$feedback)
amazon_alexa$date <- as.Date(amazon_alexa$date, "%d-%b-%y")

head(amazon_alexa)

#Performing exploratory data analysis

par(mfrow=c(2,2))
#Univariate analysis for rating
rating <- table(amazon_alexa$rating) 
rating
addmargins(prop.table(rating)*100)
barplot(rating, main = "Rating Analysis", xlab = "Ratings", ylab = "Count" , col = c("red","violet","orange","skyblue","darkgreen"))

#Insight - rating 5 has maximum number of occurences which is 2246 as compared to other ratings

#Univariate analysis for feedback
feedback <- table(amazon_alexa$feedback) 
feedback
addmargins(prop.table(feedback)*100)
barplot(feedback, main = "Feedback Analysis", xlab = "Feedback", ylab = "Frequency" , col = c("red","darkgreen"))

#The above barplot shows that approx 90% of alexa feedback is positive.


#Univariate analysis for different variants of alexa
variants <- table(amazon_alexa$variation) 
variants
addmargins(prop.table(variants)*100)

#From the above table, looks like black dot and charcoal fabric are the most popular variants in alexa.

#Putting all the reviews in text
text <- amazon_alexa$verified_reviews
#Create a corpus
docs <- Corpus(VectorSource(text))

#Data cleaning 
docs <- docs %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace)
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeWords, stopwords("english"))
for (i in seq(docs)) {
  docs[[i]] <- gsub('[^a-zA-Z|[:blank:]]', "", docs[[i]])
}
#docs <- tm_map(docs, stemDocument)
docs <- lemmatize_words(docs)
inspect(docs[1:5])

#Generating term document matrix
dtm <- TermDocumentMatrix(docs) 
matrix <- as.matrix(dtm) 
words <- sort(rowSums(matrix),decreasing=TRUE) 
df <- data.frame(word = names(words),freq=words)

#Creating Word Cloud summing overall reviews
par(mfrow=c(1,1))
set.seed(1234) # for reproducibility 
wordcloud(words = df$word, freq = df$freq, min.freq = 2,max.words=110, 
          random.order=FALSE, rot.per=0.35,scale = c(7,0.3),            
          colors=brewer.pal(8, "Dark2"))


#Performing bigram tokenization
minfreq_bigram<-2

token_delim <- " \\t\\r\\n.!?,;\"()"
bitoken <- NGramTokenizer(docs, Weka_control(min=2,max=2, delimiters = token_delim))
two_word <- data.frame(table(bitoken))
sort_two <- two_word[order(two_word$Freq,decreasing=TRUE),]
head(sort_two)

#Displaying bigrams in WordCloud to get more insights 
wordcloud(sort_two$bitoken,sort_two$Freq,random.order=FALSE,scale = c(5,0.2), min.freq = minfreq_bigram,colors = brewer.pal(4,"Spectral"),max.words=60)


#Sentiment Analysis
sentiment <- analyzeSentiment(docs)
p<-convertToDirection(sentiment$SentimentQDAP)

#Sentiment Analysis continued
s <- get_nrc_sentiment(amazon_alexa$verified_reviews)
head(s)

par(mfrow=c(2,1))
# Bar plot of sentiment analysis
barplot(colSums(s),
        las = 2,
        col = c("gray","orange","purple","pink","lightgreen","skyblue","yellow","violet","red","darkgreen"),
        ylab = 'Count',
        main = 'Sentiment scores of Alexa')

par(mfrow=c(1,1))

#getting sentiment score
z <- get_sentiment(amazon_alexa$verified_reviews)
View(z)

#Creating a new dataframe to display highly positive and negative comments
newdata <- data.frame(amazon_alexa,Sentiment_Score=z,stringsAsFactors=F )
#Displaying 15 highly negative comments
negative <- newdata$verified_reviews[newdata$Sentiment_Score< -0.25 & newdata$feedback==0]
negative
head(negative,n=15)

#Displaying 15 highly positive comments
positive<-newdata$verified_reviews[newdata$Sentiment_Score > 0.50 & newdata$feedback==1]
head(positive,n=15)

#Creating a new df to segregate positive negative and neutral comments
dataframe <- data.frame(text=sapply(docs, identity), 
                        stringsAsFactors=F)
View(dataframe)
df2 = data.frame(dataframe,Sentiment_Score=z,stringsAsFactors=F )

#Segregating sentiments further
df2$Sentiment_Score[df2$Sentiment_Score==0]<-"Neutral"
df2$Sentiment_Score[df2$Sentiment_Score> 0]<-"Positive"
df2$Sentiment_Score[df2$Sentiment_Score< 0]<-"Negative"
str(df2)
View(df3)


#Univarite for Sentiments
t<- table(df2$Sentiment_Score)
barplot(t, main = "Bar Plot", xlab = "Sentiments", ylab = "Frequency", col = c("red","purple","green"))
par(mfrow=c(1,1))

table(amazon_alexa$feedback,df2$Sentiment_Score)


# Visualize the words by polarity
df3 <- df2 %>%
  group_by(Sentiment_Score) %>%
  summarise(pasted=paste(text, collapse=" "))

# create corpus

a <- Corpus(VectorSource(df3$pasted))
a <- a %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace)
a <- tm_map(a, content_transformer(tolower))
a <- tm_map(a, removeWords, stopwords("english"))
for (i in seq(a)) {
  a[[i]] <- gsub('[^a-zA-Z|[:blank:]]', "", a[[i]])
}
a <- lemmatize_words(a)

#Generating term document matrix
tdm = TermDocumentMatrix(a)
tdm = as.matrix(tdm)
colnames(tdm) = df3$Sentiment_Score
View(tdm)



#Generating comparison word cloud for positive and negative comments.
comparison.cloud(tdm, colors = c("indianred3","darkgreen"),min.freq = 2,max.words=75,
                 scale = c(6,.7),rot.per=0.35, random.order = FALSE)










