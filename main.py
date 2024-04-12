import os
import random
import spacy
from spacy.util import minibatch, compounding

TEST_REVIEW = """
Transcendentally beautiful in moments outside the office, it seems almost
sitcom-like in those scenes. When Toni Colette walks out and ponders
life silently, it's gorgeous.<br /><br />The movie doesn't seem to decide
whether it's slapstick, farce, magical realism, or drama, but the best of it
doesn't matter. (The worst is sort of tedious - like Office Space with less humor.)
"""


def load_training_data(
    data_directory: str = "aclImdb/train",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label
                            }
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]


def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    nlp = spacy.load('en_core_web_sm')
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architectures": "simple_cnn"}
        )
        nlp.add_pipe(textcat)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()

        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF1 Score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )
        for i in range(iterations):
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size = batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(
                    text,
                    labels,
                    drop = 0.2,
                    sgd = optimizer,
                    loss = loss
                )
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer = nlp.tokenizer,
                    textcat = textcat,
                    test_data = test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")


def evaluate_model(
        tokenizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]
        for predicted_label, score in review.cats.items():
            if (
                predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label == "pos":
                true_positives += 1
            elif score >= 0.5 and true_label == "neg":
                false_positives += 1
            elif score < 0.5 and true_label == "neg":
                true_negatives += 1
            elif score < 0.5 and true_label == "pos":
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f_score": f_score}


def test_model(input_data: str = TEST_REVIEW):
    loaded_model = spacy.load("model_artifacts")
    parsed_text = loaded_model(input_data)
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )


if __name__ == '__main__':
    train, test = load_training_data(limit=2500)
    train_model(train, test)
    print("Testing Model")
    test_model()
