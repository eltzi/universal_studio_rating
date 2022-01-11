import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.feature_selection import SelectPercentile, f_classif
nltk.download('stopwords')


def nltk_data_load(text_column_name):
    df = pd.read_csv(os.getcwd() + "\\data\\universal_studio_branches.csv")
    df[text_column_name] = df['title'] + " " + df['review_text']

    percentage_review = (df.rating.value_counts() / len(df.rating)) * 100
    print(percentage_review)

    return df


def nltk_dataframe_string_column_process():
    dataframe = nltk_data_load("text")

    stemmer = PorterStemmer()
    words = stopwords.words("english")

    # stemming & remove stopwords
    dataframe['processed_text'] = dataframe["text"].apply(lambda x: " ".join([
        stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

    dataframe['processed_text'] = dataframe['processed_text'].str.lower()

    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    dataframe['processed_text'] = [dataframe['processed_text'][row].translate(table) for row in
                                   range(len(dataframe['processed_text']))]

    # remove hash tags
    dataframe['processed_text'] = dataframe['processed_text'].str.replace("#", " ")

    # remove words less than 1 character
    dataframe['processed_text'] = dataframe['processed_text'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

    # remove rare words
    list_of_words = dataframe['processed_text'].str.split().to_list()
    words_counter = Counter(chain.from_iterable(list_of_words))
    dataframe['processed_text'] = [' '.join([j for j in i if words_counter[j] > 2]) for i in list_of_words]

    return dataframe


def nltk_worldcloud(dataframe):
    freq_words = ' '.join([text for text in dataframe['processed_text']])
    wordcloud = WordCloud(width=800, height=500, random_state=1, max_font_size=90,
                          max_words=50).generate(freq_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def nltk_train_test_split(dataframe, y_column_name, X_column_name):
    y = dataframe[y_column_name]
    X = dataframe[X_column_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=43, shuffle=True)

    vectorized_tfidf = TfidfVectorizer(stop_words='english', sublinear_tf=True)
    train_tfIdf = vectorized_tfidf.fit_transform(X_train.values.astype('U'))
    test_tfIdf = vectorized_tfidf.transform(X_test.values.astype('U'))
    print(train_tfIdf.shape, test_tfIdf.shape, y_train.shape, y_test.shape)

    return train_tfIdf, test_tfIdf, y_train, y_test


def nltk_borderlinesmote_oversample(X_train, X_test, y_train, y_test):
    smote = BorderlineSMOTE()
    new_x_train, new_y_train = smote.fit_resample(X_train, y_train)
    new_x_test, new_y_test = smote.fit_resample(X_test, y_test)

    print(new_x_train.shape, new_x_test.shape, new_y_train.shape, new_y_test.shape)

    return new_x_train, new_x_test, new_y_train, new_y_test


def nltk_percentile_obeservations_select(X_train, X_test, y_train):
    selector = SelectPercentile(score_func=f_classif, percentile=90).fit(X_train, y_train)

    X_train_selected = selector.transform(X_train)
    print(X_train_selected.shape)

    X_test_selected = selector.transform(X_test)
    print(X_test_selected.shape)

    return X_train_selected, X_test_selected


def nltk_model_build(X_train, X_test, y_train, y_test):
    m = ComplementNB()
    model = m.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))

    df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df_pred)


if __name__ == "__main__":
    dataset = nltk_dataframe_string_column_process()

    # under-presented classes
    X_train, X_test, y_train, y_test = nltk_train_test_split(dataset, "rating", "processed_text")

    # model score: 0.67 test score:0.59
    nltk_model_build(X_train, X_test, y_train, y_test)

    # oversampling - model score: 0.72 test score:0.50
    X_train_oversmpling, X_test_oversmpling, y_train_oversmpling, y_test_oversmpling = \
        nltk_borderlinesmote_oversample(X_train, X_test, y_train, y_test)
    nltk_model_build(X_train_oversmpling, X_test_oversmpling, y_train_oversmpling, y_test_oversmpling)

    # oversampling train - model score: 0.72 test score:0.56
    nltk_model_build(X_train_oversmpling, X_test, y_train_oversmpling, y_test)

    # percentile features selected - model score: 0.67 test score:0.59
    X_train_selected, X_test_selected= nltk_percentile_obeservations_select(X_train, X_test, y_train)
    nltk_model_build(X_train_selected, X_test_selected, y_train, y_test)

    # oversampling & percentile selection - model score: 0.78 test score:0.54
    X_train_selected_oversmpling, X_test_selected_oversmpling, y_train_oversmpling, y_test_oversmpling = \
        nltk_borderlinesmote_oversample(X_train_selected, X_test_selected, y_train, y_test)
    nltk_model_build(X_train_selected_oversmpling, X_test_selected, y_train_oversmpling, y_test)

