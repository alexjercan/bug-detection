import pickle
import codenet

from gensim.models import FastText
from sklearn.tree import DecisionTreeClassifier

models_path = "../models/"

vectorizer_path = models_path + "vectorizer_fasttext.model"
classifier_path = models_path + "classifier_decisiontree.pkl"


def inference(
    source_code: str, vectorizer: FastText = None, clf: DecisionTreeClassifier = None
) -> str:
    if vectorizer is None:
        vectorizer = FastText.load(vectorizer_path)
    if clf is None:
        with open(classifier_path, "rb") as f:
            clf = pickle.load(f)

    token_df = codenet.run_pythontokenizer_str(source_code)
    tokens = token_df["text"].values
    tokens = vectorizer.wv[tokens]

    prediction = clf.predict(tokens)
    token_df["prediction"] = prediction
    token_err_df = token_df[token_df["prediction"] != "Accepted"]

    return token_err_df.to_csv(index=False)


if __name__ == "__main__":
    source_code = input()

    result_csv = inference(source_code)

    print(result_csv)
