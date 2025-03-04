s'arrêter avant Tc sans le rendre petit ? est ce que ça marche bien?

maths derriere a tracer, expression de landau
et lien avec flowmatching

autre utilité : une fois qu'on a commit sur une classe, on peut utiliser un meilleur score

faire attention, on a aussi un aléatoire sur les datapoints a pour le random energy model, qui sont tirés selon une gaussienne

un petit point sur la génération conditionnelle ? parce que conditionner sur une classe change le score
à quel point ? car sur des petits temps les 2 scores doivent coincider

et que faire si on train un score en enlevant tous les atures, et dès qu'on est trop près on change de score


si orthogonal alors on peut faire une svd très simple, et avec la svd (de la matrice de covariance) on retrouve bien 2 axes principaux avec des valeurs propres distinctes, et donc la hiérarchie sort de là, maintenant quand on calcule les scores conditionnels (conditionnés au fait que x collapse sur m ou -m, ensuite on regarde avec ce conditionnement le potentiel qui donc avec epsilon doit nous montrer la même structure qu'auparavant (collapse vers + ou - eps) et on peut même au final trouver un critère sur epsilon en fonction de mu et d pour que l'on ait autosimilarité des potentiels !!!


faire du CFG avec plusieurs classes (et genre on push les directions sur un nombre p de dimensions (2^p encodage))

