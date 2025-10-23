from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Input sequence and candidate labels
sequence_to_classify = "how are u"
candidate_labels = ["WELCOME", "ANSWER", "CLARIFY", "SKIP", "OFFTOPIC", "FEEDBACK", "END"]

# Run the classifier
result = classifier(sequence_to_classify, candidate_labels)

# Print result in a structured way
print("Input:", sequence_to_classify)
print("\nIntent Classification Results:")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label:10s} -> {score:.4f}")

# Print the top label (predicted intent)
print("\nPredicted Intent:", result['labels'][0].upper())
