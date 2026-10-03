import pickle
from utils import clean_text

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# label: 1 = real, 0 = fake
samples = [
    ("""The central bank kept its benchmark interest rate unchanged on Wednesday and said it would continue to watch inflation data closely before making any further decisions, according to a statement released after the meeting.""", 1),

    ("""The city council approved a new budget for road repairs on Monday, with work expected to begin next spring. Officials said the plan includes funding for bridges and public transport.""", 1),

    ("""The national cricket team won the series 3-1 after a strong batting performance on the final day, with the captain scoring a century in the deciding match.""", 1),

    ("""SHOCKING: Scientists are hiding a secret cure that big companies don't want you to know about! Share this before they delete it!!!""", 0),

    ("""You won't believe what the government is secretly doing to your food. Anonymous insiders reveal the truth the media will never tell you!""", 0),

    ("""BREAKING: Famous celebrity EXPOSED in massive cover-up, and the mainstream media is too scared to report it. Wake up before it's too late!""", 0),
]

correct = 0

for text, label in samples:
    pred = model.predict(vectorizer.transform([clean_text(text)]))[0]
    ok = bool(pred == label)
    correct += ok
    print("OK   " if ok else "WRONG", "| predicted:", "Real" if pred == 1 else "Fake", "|", text[:70])

print(f"\nAccuracy: {correct}/{len(samples)} = {correct / len(samples) * 100:.1f}%")