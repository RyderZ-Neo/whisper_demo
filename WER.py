import Levenshtein

def calculate_wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()

    distance = Levenshtein.distance(ref_words, hyp_words)
    wer = distance / len(ref_words)

    return wer*100

def print_differences(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()

    differences = Levenshtein.editops(ref_words, hyp_words)
    print(differences)
    for op, i, j in differences:
        if op == 'insert':
            print(f"Insert word '{hyp_words[j]}' at position {i}")
        elif op == 'delete':
            print(f"Delete word '{ref_words[i]}' at position {i}")
        elif op == 'replace':
            print(f"Replace word '{ref_words[i]}' with '{hyp_words[j]}' at position {i}")

# Example usage
reference = "This is the reference transcript!"
hypothesis = "This is the transcript!"

wer = calculate_wer(reference, hypothesis)
print("WER:", wer,"%.")

print("Differences:")
print_differences(reference, hypothesis)
