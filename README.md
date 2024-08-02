# Ribonanza-RNA-Folding

Last year, Stanford hosted a competition focused on predicting RNA base reactivity using Map and 2A3 reactivity data. 

To address this challenge, This repository employs an encoder-decoder architecture.  

The conformer-based encoder captures global features of the RNA sequence. Subsequently, the decoder leverages positional encoding to process the data and a masking mechanism to prevent interactions between non-interacting bases, thereby refining the reactivity predictions. 