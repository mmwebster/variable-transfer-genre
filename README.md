# Variable Transfer Learning for Music Genre Prediction
## Summary
Models trained in this project predict the genre of a track from its spectrogram using the FMA dataset and a Transfer Learning framework. This project replicates the findings of Kim et al (https://arxiv.org/pdf/1805.02043.pdf), but on a 25k-sample subset of the FMA dataset. We've borrowed initial feature extraction and some additional scripts from a similar project that attempted to replicate another portion of this work (https://github.com/falkaer/artist-group-factors). Our contribution is to experiment with transfer from different layers of the DCNNs to an MLP.
