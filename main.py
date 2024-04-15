from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import scipy.sparse
import numpy as np
from PIL import Image


"""
Schreib dein eigenes KNN-Matting(binary) Programm
Ziel: alpha ausrechnen (alles hier auf diesem file)
"""

"""
1. Image und Trimap geladen
"""

image  = np.array(Image.open( "lemur.png").convert("RGB"))/255.0
trimap = np.array(Image.open("lemur_trimap.png").convert(  "L"))/255.0


"""
2. Laplacian Matrix berechnen
"""

#2.1. Wir lesen zunächst die Höhe, breite, R, G, B und n ein
h, w, _ = image.shape
r, g, b = image.reshape(-1, 3).T
n = w * h


#2.2. Wir bauen ein Koordinatensystem
x = np.tile(np.linspace(0, 1, w), h)
y = np.repeat(np.linspace(0, 1, h), w)


#2.3. nn und dw gleichzeitig durchgehen (nn = Anzahl der k Nachbarn, dw = Distanz-Gewicht)
nn = [20, 10]
dws = [2.0, 0.1]
i, j, d = [], [], []


for k, dw in zip(nn, dws):

    #2.3.1. berechnen den feature vector zu allen Pixeln durch
    f = np.stack([r, g, b, x*dw, y*dw], axis=1, out=np.zeros((n, 5), dtype=np.float32))

    #2.3.2. index der nähsten nachbarn ni für alle feature vectores berechnen
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(f)
    _, ni = nbrs.kneighbors(f)




    #2.3.3. i und j berechnen
    i.append(np.repeat(np.arange(n), k))
    j.append(ni.ravel())


    #2.3.4. data berechnen
    d.append(np.ones(k*n))


#2.4. i,j,d verdoppeln
ij= np.concatenate(i+j)
ji = np.concatenate(j+i)
d = np.concatenate(d+d)

#2.5. Weightmatrix berechnen
W = scipy.sparse.csr_matrix((d, (ij, ji)), (n, n))

#Einschub: W normalisieren
W = normalize(W, axis=1, norm='l1')

#Laplace endlich berechnen
L = np.identity(n) - W


'''
3. Nun könne wir nach alpha ausrechnen
'''

#Hier wird die Trimap aufgebaut
is_fg = (trimap > 0.9).flatten()
is_bg = (trimap < 0.1).flatten()
is_known = is_fg | is_bg
is_unknown = ~is_known

#Hier wird die Diagonalmatrix gebaut
d = is_known.astype(np.float64)
D = scipy.sparse.diags(d)

#Hier wird die Diagonalmatrix mit lambda multipliziert und mit LM addiert
lambda_value = 100.0
A = lambda_value * D + L

#Hier wird b_S berechnet
b = lambda_value * is_fg.astype(np.float64)

#Hier wird das LGS für alpha = lambda * b_S * (L + lambda * D_S)^{-1} berechnet
alpha = scipy.sparse.linalg.spsolve(A, b).reshape(h, w)

#Wir fusionieren die Alphawerte mit den dem Bild
cutout = np.concatenate([image, alpha[:, :, np.newaxis]], axis=2)



'''
4. Nun werden die Alpha-werte als Bild abgespeichert und ausgegeben
'''

#Hier clippen wir die werte wieder zurück zu 0-255 und konvertieren es zu uint8
alpha = np.clip(alpha*255, 0, 255).astype(np.uint8)

#Hier speichern wir Alphawerte als Bild ab
Image.fromarray(alpha).save("lemur_alpha.png")

#Hier werden die Bilder geöffnet und uns gezeigt
Image.fromarray(alpha).show(title="alpha")



'''
5. Nun wird der Vordergrund als Bild abgespeichert und ausgegeben
'''

#Hier clippen wir die werte wieder zurück zu 0-255 und konvertieren es zu uint8
cutout = np.clip(cutout*255, 0, 255).astype(np.uint8)

#Hier speichern wir Alphawerte als Bild ab
Image.fromarray(cutout).save("lemur_foreground.png")

#Hier werden die Bilder geöffnet und uns gezeigt
Image.fromarray(cutout).show(title="foreground")






