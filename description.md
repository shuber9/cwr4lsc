# Collecting what the scripts do

## collect from coha
uses usage_collector.collect_from_coha to get dict lemma -> [(vector, sentence, word_position, decade), (v, s, p, d), ...] for target words 

## usage_collector 

requires a .txt file as input, one sentence per line. file names are all_xxxx.txt where xxxx is the decade in the original usecase, i might do buckets of smaller size