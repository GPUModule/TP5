## TP5 : Programmation GPU en CUDA


### Objectifs du TP :
---------------------
* Comprendre comment utiliser la bibliothèque Thrust pour effectuer le tri des paires key-value.
* Comprendre comment utiliser la bibliothèque Thrust pour simplifier l'algorithme d'analyse
* Comprendre comment les primitives parallèles (fonctions haut-niveau de thrust) peuvent être utilisées dans un problème complexe (interactions entre particules)
* Voir si il existe des différences de performances des streams via deux méthodes distinctes.

### Informations sur le TP :
----------------------------
* Le TP est à réaliser seul ou en binôme.
* Un compte rendu du TP sera à fournir avec le code (Latex, Word ou MarkDown)
* A rendre sur ametice en spécifiant si vous avez fait le TP seul ou en binôme, en mentionnant votre prénom ainsi que votre nom (prénom et nom de chaque membre si en binôme).
* A rendre avant le 10 Avril 2022, 24:00.


### Exercice 1 : Streams et execution asynchrone
------------------------------------------------

* Vous programmerez sur le fichier **exercice01.cu**

L'exercice consiste à utiliser une calculatrice parallèle qui appliquera un programme fixe de calculs à un ensemble de données d'entrée en utilisant un modèle SPMD (Single Program Multiple Data, plusieurs processeurs autonomes exécutent simultanément le même programme en des points indépendants). Une implémentation GPU a déjà été fournie. La partie de l'implémentation dont les performances sont critiques (et donc chronométrée) nécessite de copier les données d'entrée sur le GPU, d'effectuer les calculs et enfin de relire les données sur le GPU. 

#### 1.1. Stream par defaut
Lisez et comprenez ce que fait le kernel parallelcalculator puis codez les instructions suivantes

##### 1.1.1. Copie Host 2 Device
Completez l'instruction situé sous le commentaire suivant :

```c
// Exercice 1.1.1 Copie H2D
...
```

##### 1.1.2. Lancement du kernel
Completez l'instruction situé sous le commentaire suivant :

```c
// Exercice 1.1.2 Lancement du kernel
...
```

##### 1.1.3. Copie Device 2 Host
Completez l'instruction situé sous le commentaire suivant :

```c
// Exercice 1.1.3 
...
```


L'implémentation fournie est strictement synchrone dans son exécution et utilise uniquement le stream CUDA par défaut. Il est possible d'améliorer les performances en décomposant la tâche en parties indépendantes plus petites et en utilisant les streams pour s'assurer que les parties copie en memoire et de calcul sont occupés à des tâches indépendantes. Vous allez implémenter 2 versions différentes de streaming et comparer les performances de chacune. La première version asynchrone que vous implémenterez bouclera sur chaque streams et effectuera une copie vers le GPU, l'exécution du kernel et la copie en retour du GPU. Chaque streams sera responsable d'un sous-ensemble de données (de la taille SAMPLES divisée par NUM_STREAMS). Implémentez cette version en effectuant les tâches suivantes qui nécessitent de modifier la fonction cudaCalculatorNStream1;

#### 1.2. cudaCalculatorNStream1

##### 1.2.1. Allocation de la mémoire CPU (épinglé) et GPU
Ecrivez les instructions vous permettant d'allouer la mémoire sur CPU et GPU.

```c
// Exercice 1.2.1. Allocation de la mémoire CPU et GPU
```

##### 1.2.1. Initialisation des streams
Ecrivez les instructions vous permettant de créer vos streams dans un tableau de taille NUM_STREAMS 

```c
// Exercice 1.2.2. Initialisation des streams
```

##### 1.2.3. Boucle sur chaque streams
Ecrivez les instructions vous permettant de d'iterer sur les streams. Pour chaque streams, lancez (1) une copie de mémoire asynchrone depuis le CPU vers le GPU, (2) le kernel, et (3) une copie de mémoire asynchrone du GPU vers le CPU. Attention, chaque flux ne doit opérer que sur un sous-ensemble de données.

```c
// Exercice 1.2.3. Boucle sur les streams : Copie H2D, lance le kernel, copie D2H
```

##### 1.2.4. Destruction des streams
Ecrivez les instructions vous permettant de detruire les streams

```c
// Exercice 1.2.4. destruction des streams```

Pour la deuxième version asynchrone, nous apporterons une modification subtile à la façon dont le travail dans les streams est planifié.

##### 1.2.5. Verification
Decommentez l'instruction suivante et executez le programme.

```c
//DECOMMENTEZ//cudaCalculatorNStream1(...);
```
Le programme devrait afficher le temps d'exécution de la version synchrone, et asynchrone 1. Avec 0 erreurs pour les deux versions.

#### 1.3. cudaCalculatorNStream2


##### 1.3.1. Allocation de la mémoire CPU (épinglé) et GPU
Ecrivez les instructions vous permettant d'allouer la mémoire sur CPU et GPU.

```c
// Exercice 1.3.1. Allocation de la mémoire CPU et GPU
```

##### 1.3.2. Initialisation des streams
Ecrivez les instructions vous permettant de créer vos streams dans un tableau de taille NUM_STREAMS 

```c
// Exercice 1.2.2. Initialisation des streams
```

##### 1.3.3. Copie de mémoire asynchrone
Ecrivez les instructions vous permettant d'itérer sur les streams et programmez chaque streams pour effectuer une copie de la mémoire du CPU vers le GPU sur un sous-ensemble approprié de données.

```c
// Exercice 1.3.3. Copie H2D Asynchrone
```

##### 1.3.4. Lancement du kernel
Ecrivez les instructions vous permettant d'itérer sur les streams et programmez chaque streams pour lancer le kernel sur un sous-ensemble approprié de données.

```c
// Exercice 1.3.4. Execution des kernels
```

##### 1.3.4. Lancement du kernel
Ecrivez les instructions vous permettant d'itérer sur les streams et programmez chaque streams pour effectuer une copie de la mémoire du GPU vers le CPU sur un sous-ensemble approprié de données.

```c
// Exercice 1.3.5. Copie D2H Asynchrone
```

##### 1.3.5. Destruction des streams
Ecrivez les instructions vous permettant de detruire les streams

```c
// Exercice 1.3.5. destruction des streams
```

#### 1.4. Execution du programme

Decommentez l'instruction suivante et executez le programme.

```c
//DECOMMENTEZ//cudaCalculatorNStream2(...);
```

Le programme devrait afficher le temps d'exécution de la version synchrone, asynchrone 1, et asynchrone 2. Avec 0 erreurs pour les trois versions.

##### 1.4.1. 
Que constastez vous sur les temps de calcul ? Pourquoi, selon vous, le temps des deux versions asynchrones ne sont pas très différent ?

##### 1.4.2. 
Modifier le programme afin d'utiliser 3 streams. Est-ce normal d'avoir des erreurs lors de l'execution ? si oui, comment peut-on calculer le nombre d'erreurs qu'il y aura avec SAMPLE, NUM_STREAMS et TBP (threadsPerBlock).


### Exercice 2 : Thrust
------------------------------------------------

Dans l'exercice deux, Vous allez modifier une implementation sur la  simulation de particules simple. L'implémentation calcule le voisin le plus proche de chaque particule dans une plage fixe. Il le fait à travers les étapes suivantes.

1. Génére des paires clé-valeur: où la clé est la case auquel appartient la particule et la valeur est l'indice de la particule.
2. Trie les paires clé valeur : les paires clé valeur sont triées selon leur clé
3. Réorganisation des particules: les particules sont réorganisées à partir des paires de valeurs clés triées.
4. Calcule un histogramme des particules : Pour chaque case qui discrétise l'espace ou se situe les particules, un
compte du nombre de particules présent dans la case est effectué.
5. Algorithme de Somme de préfixe ([Lien](https://stringfixer.com/fr/Prefix_sum)) des cases: compte tenu du nombre de case, une somme de préfixe est utilisée pour déterminer où
l'index de départ de chaque case est situé dans la liste triée des particules
6. Calcule le voisin le plus proche pour chaque particule en considérant tous les voisins dans un intervalle de distances.

L'implémentation fournie utilise le CPU pour la plupart de ces étapes (autre que le calcul du plus proche voisin). Pour améliorer ce code, nous allons réimplémenter les fonctions CPU sur le GPU en effectuant un certain nombre d'étapes. Effectuez les opérations suivantes

#### 2.1 Implementation GPU des versions CPU
Les versions CPU sont déjà implémenté est situé en fin du programme

##### 2.1.1. keyValues
Implementez le kernel keyValues qui correspond à la version GPU de **keyValuesCPU(...)**

```c
 // Exercice 2.1.1
```

##### 2.1.2. ReorderParticles
Implementez le kernel keyValues qui correspond à la version GPU de **reorderParticlesCPU(...)**

```c
 // Exercice 2.1.2
```

##### 2.1.3. HistogramParticles
Implementez le kernel HistogramParticles qui correspond à la version GPU de **histogramParticlesCPU(...)**. Attention il faudra utiliser une fonction atomique. Trouvez et expliquez pourquoi.

```c
 // Exercice 2.1.3
```

#### 2.2.

Dans la fonction **particlesGPU(...)**, effectuez les opérations suivantes.

##### 2.2.1. Copie host 2 device
Ecrivez le code permettant de faire de copier h_particles dans d_particles de taille **particles**

```c
// Exercice 2.2.1. Copie H2D
```

##### 2.2.2. Fonction de trie par clefs
Utilisez la fonction **thrust::sort_by_key();** prenant respectivement d_key_values->sorting_key, (d_key_values->sorting_key + NUM_PARTICLES), et d_key_values->value comme argument.

Attention, c'est trois entier sont des pointeurs, il faut au prealable les transforment en structure device_ptr de thrust avec **thrust::device_ptr<int>(pointeur)**

```c
//Exercice 2.2.2. On trie par key
```

##### 2.2.3. reorderParticles
Appelez le kernel reorderParticles en utilisant comme dimension de grille le nombre de particules divisé par le nombre de thread par bloc (TPB).

Les arguments du kernel sont respectivement, d_key_values, de_particles et d_particles_sorted.

```c
//Exercice 2.2.3. On appelle le kernel reorderParticles
```

##### 2.2.4. histogramParticles
Appelez le kernel histogramParticles avec pour arguments d_particles_sorted et d_env. Les dimensions de la grille et des blocs sont même sont les même que pour reorderParticles

```c
//Exercice 2.2.4. On appelle le kernel histogramParticles
```

##### 2.2.5. somme de prefix avec thrust

La somme de prefix avec thrust se fait avec la fonction **thrust::exclusive_scan(InputIterator first, InputIterator last, OutputIterator result)**

Les arguments à utiliser sont l'InputIterator first **d_env->count**, lÌnputIterator last **d_env->count + ENV_BINS** et L'OutputIterator **d_env->start_index**

Attention, **d_env->count**, **d_env->count + ENV_BINS** et **d_env->start_index** sont des pointeurs. transformez les en Iterator via la fonction **thrust::device_pointeur_cast(d_env->count)**.

##### 2.2.6. Executez le programme
Que constatez vous au niveau du temps d'execution de la fonction particlesGPU par rapport à la fonction particlesCPU ?





