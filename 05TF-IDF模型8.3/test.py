from sklearn.feature_extraction.text import TfidfVectorizer

# 文本文档列表
text = ['I have a pen', 'I have a apple']

# 创建变量函数
vectorizer = TfidfVectorizer()

# 词条化以及创建词汇表
vectorizer.fit(text)

print(vectorizer.vocabulary_)