#!/bin/bash

touch temp_sentences.txt
touch temp_test.txt
touch log_results.txt

# Remove all contents from the two text files
echo -n "" > temp_sentences.txt
echo -n "" > temp_test.txt
echo -n "" > log_results.txt

# Generate 10 sentences and save them to a temporary file
ngram -lm 2gram.lm -gen 10 > temp_sentences.txt

# Process each sentence
while IFS= read -r line; do
    # Truncate to the first 5 words plus the end-of-sentence token
    truncated_sentence=$(echo $line | awk '{print "<s> " $1, $2, $3 " </s>"}')
    echo "Truncated Sentence: $truncated_sentence"

    # Write the truncated sentence to a temporary file for probability calculation
    echo $truncated_sentence >> temp_test.txt

done < temp_sentences.txt

# Calculate the probability of the truncated sentence
ngram -lm 2gram.lm -ppl temp_test.txt -debug 2 > log_results.txt

# Clean up temporary files
# rm temp_sentences.txt temp_test.txt
