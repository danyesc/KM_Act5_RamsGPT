import json
import re

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------- Knowledge base ----------
with open("knowledge_base.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

MIN_SCORE = 0.12      # below this (expanded question), answer "not found"
MIN_RAW_SCORE = 0.10  # the original question must also match something in the KB
MAX_RESULTS = 3       # how many entries can be combined in one answer
RELATIVE_CUTOFF = 0.7 # extra entries must score at least 70% of the best one


def doc_text(doc):
    """Text indexed for each entry. Title and tags are repeated so they weigh more than the body."""
    title = doc.get("title", "")
    tags = " ".join(doc.get("tags", []))
    years = " ".join(str(y) for y in doc.get("years", []))
    return f"{title} {title} {tags} {tags} {doc['category']} {years} {doc['content']}"


texts = [doc_text(d) for d in documents]

# Two views of the same text: whole words (+ bigrams) and character n-grams.
# Character n-grams make "played" match "play", "won" match "win", typos, etc.
word_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
word_matrix = word_vec.fit_transform(texts)
char_matrix = char_vec.fit_transform(texts)

# Small query expansion: map the way people ask to the words used in the entries.
EXPANSIONS = [
    (r"\b(won|win|wins|winning|champion\w*|titles?|trophy|trophies|rings?)\b", "super bowl championship titles"),
    (r"\b(played|play|plays|home|venue|stadiums?)\b", "stadium home games played"),
    (r"\b(moved|move|relocat\w*|left|leave)\b", "move relocation"),
    (r"\b(founded|started|began|origins?)\b", "founding history origin"),
    (r"\b(coach|coached|coaches)\b", "head coach"),
    (r"\b(qb|quarterbacks?)\b", "quarterback"),
    (r"\b(wr|receivers?)\b", "wide receiver"),
    (r"\b(legends?|hall of fame|hof|all[- ]time)\b", "legends hall of fame"),
    (r"\b(beat|defeated?|lost|loss|losses|lose)\b", "lost super bowl losses"),
    (r"\b(colou?rs?|logo|mascot)\b", "colors logo mascot identity"),
]


def expand(question):
    q = question.lower()
    extra = [add for pattern, add in EXPANSIONS if re.search(pattern, q)]
    return q + " " + " ".join(extra)


def score_all(text):
    word_scores = cosine_similarity(word_vec.transform([text]), word_matrix).ravel()
    char_scores = cosine_similarity(char_vec.transform([text]), char_matrix).ravel()
    return 0.5 * word_scores + 0.5 * char_scores


def search_knowledge_base(question):
    # The original question decides IF there is an answer (keeps off-topic questions out);
    # the expanded question decides WHICH entries rank first.
    if score_all(question.lower()).max() < MIN_RAW_SCORE:
        return []

    scores = score_all(expand(question))
    order = scores.argsort()[::-1]
    best = scores[order[0]]
    if best < MIN_SCORE:
        return []

    return [
        (documents[i], float(scores[i]))
        for i in order[:MAX_RESULTS]
        if scores[i] >= best * RELATIVE_CUTOFF
    ]


# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")  # index.html must live in ./templates/


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"answer": "Please enter a question.", "source": None})

    results = search_knowledge_base(question)

    if not results:
        return jsonify({
            "answer": "I don't have information about that yet. Try asking about Rams history, "
                      "championships, stadiums, coaches or players.",
            "source": None,
        })

    answer = "\n\n".join(doc["content"] for doc, _ in results)
    sources = list(dict.fromkeys(doc["source"] for doc, _ in results))  # unique, keeps order

    return jsonify({"answer": answer, "source": ", ".join(sources)})


if __name__ == "__main__":
    app.run(debug=True)