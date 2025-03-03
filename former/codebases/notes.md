ways to circumvent memorization :
- deconvolution with adding noise to images https://arxiv.org/pdf/2502.05446 with https://github.com/baptistar/DiffusionModelDynamics
- or manifold interpolation
- or ensemble training to mitigate https://arxiv.org/pdf/2502.09434
- or regularization 
- or large learning rates https://arxiv.org/pdf/2502.03435



faire une présentation qui suit le temps et on déroule les résultats anciens et nouveaux

beyong the empirical score hypothesis, donc toutes lesm éthodes citées de regu ou de changement du score permet tout ça
- une sorte de compromis de regularization du score pour 

the volume argument wedevelop in this work can be applied beyond the “exact empirical score” hypothesis, and could offer a way to analyze quantitatively how the collapse depends on n, d and the capacity of the model used to learn the score.

--- ça veut dire qu'avec un score inexact on peut quand même étudier non paramétriquement les phénomènes sur un modèle bien
entrainé

- prendre en compte les zones pour un meilleur training ? pendant le trianing les identifier non paramétriquement
et 