# KNN-Matting
KNN-Matting Code zum Verstehen

### Motivation: 
Mitte März 2024 habe ich mich mit Alpha-Matting im Rahmen meiner Bachelorarbeit beschäftigt und hatte da das ein oder eine Problemschen gehabt. Und da ich weiß wie es ist sich wochenlang mit einem Thema zu beschäftigen ohne das Gefühl zu haben voran zu kommen, da alles unnötig kompliziert erklärt wird. Versuche ich hiermit KNN-Matting, für die deutschsprachigen Kollegen (Auf Englisch gibts eh genug Ressources zur Verfügung, die sollen sich damit zufrieden geben :) ), kurz und knackig, anhand eines Codebeispiels zu veranschaulichen. 

### Ziel des KNN-Algo:
  Wir versuchen ein Bild, mithilfe einer Trimap, durch das Verwenden vom KNN (k-Nearest-Neighbour) in Vordergrund und Hintergrund zu unterteilen.

### Grundlagen:
  - Trimap: 
    - Die Unterteilung des Eingabebildes in 100% Vordergrund(weißer Teil), 100% Hintergrund(schwarzer Teil) und Unknown(grauer Teil) aufzuteilen.
  
  ![me](https://github.com/IliasIhadian/Closed-Form-Matting/assets/74773501/30138eb8-6c60-42f3-af13-f6c1b5a29773)
  ![metri](https://github.com/IliasIhadian/Closed-Form-Matting/assets/74773501/1ece1560-cf44-49c1-9c14-b84dcee2be6e)

  - KNN:
    -Pixel welche ähnlich aussehen sollen den selben Alpha-Wert besitzten. 

  - Cost-Function:
    - Eine Cost Function (Kostenfunktion) in maschinellem Lernen ist eine mathematische Funktion, die die Fehler oder Unterschiede zwischen den vorhergesagten Werten eines Modells und den tatsächlichen Werten misst, um zu optimieren, wie gut das Modell funktioniert. Sie wird verwendet, um das Modell während des Trainings anzupassen, indem sie minimiert wird.

  - Laplace Matrix:
    - Ist eine Matrix wo die Diagonale positive Werte besitzt und alle anderen Werte negativ sind.

  - KNN
    - findet im Allgemeinen heraus welche die k nähsten Daten eines Datenpunktes sind und berechnet anhand dessen aus zur welcher Kategorie der Datenpunkt gehört. In diesem Kontext wird viel eher herausgefunden welche die nähsten Nachbarn eines Pixel sind. (Die nähsten sind diese welche am ähnlichsten aussehen und auch lokal nah sind) 
  
      ![image](https://github.com/IliasIhadian/KNN-Matting/assets/74773501/90f6334a-c1a0-413c-bfbd-110582ad8ab7)


### Mathe-Stuff:
Auf die Einzelheiten der Mathematik hinter dem KNN-Algo werde ich nicht eingehen, hierfür leite ich euch auf die letze Quelle. Wir versuchen $$\alpha = \lambda b_S(L + \lambda D_S)^{-1}$$ auszurechnen. <br />
$$b_S$$ ist ein Vektor, welcher für die markierten Pixel eine Alphawert hat und für die Unmarkierten eine 0.
$$D_S$$ ist eine Diagonalmatrix welche für die markierten Pixel eine 1 hat und für die unmarkierten eine 0.
$$L$$ ist eine Laplace Matrix, (i,j) Wert, so berechnet wird: $$L_{i j}= I_{ij} - W_{ij}$$, wobei I eine Identitätsmatrix ist und W eine "Weight"matrix.
$$W$$ ist eine "Weight"matrix, welche bei einem durchgang von KNN anzeigt wie oft ein gewisser Pixel j der Nearest Neighbour für den Pixel i war.
  
## KNN Algo Plan: (binary)

1. image + trimap laden
2. Laplacian matrix berechnen
    1. wir lesen zunächst die höhe, breite, r,g,b, n ein
    2. bauen ein koordinaten system
    3. gehen durch n_neighbors, distance_weights gleichzeitig durch n_neighbours = [20,10] distance_weights = [2.0, 1]
        1. berechnen den feature vector zu allen Pixeln durch
        2. index der nähsten nachbarn für alle feature vectores berechnen
        3. i und j berechnen
        4. eine coo_data = k*n einsen
    4. i und j verdoppeln wir und speichern es
    5. damit bauen wir dann die weight-matrix
    6. L = I - W, wobei I = scipy.sparse.identity(n)
3. diese formel anwenden

   ![image](https://github.com/IliasIhadian/KNN-Matting/assets/74773501/521b601a-3b85-499f-9f9a-2e7b1f3d02e7)
5. alpha werte ausgeben
6. Vordergrund als Bild abgespeichert und ausgegeben

  


### Quelle:

Das Paper welches den KNN-Mating-Algo zuallererst veröffentlicht hat: <br />
  **Titel**: KNN Matting <br />
  **Authoren**: Qifeng Chen, Dingzeyu Li, Chi-Keung Tang <br />
  **Erscheinungsjahr**: 2013  <br />
  [Paper](https://dingzeyu.li/files/knn-matting-tpami.pdf)


  
