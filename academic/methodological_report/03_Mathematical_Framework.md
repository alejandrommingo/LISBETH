# 3. Marco Matemático: Subespacios y Ortogonalidad

Para modelar la "identidad" de Yape no como un punto estático, sino como un constructo dinámico multidimensional, se formalizó el siguiente pipeline algebraico.

## 3.1 Definición de Subespacios (SVD)

Sea $X_t \in \mathbb{R}^{n \times d}$ la matriz de embeddings de todas las $n$ ocurrencias de la marca en la ventana de tiempo $t$. Centramos los datos restando la media $\mu_t$.
Aplicamos **Descomposición en Valores Singulares (SVD)**:
$$ X_t = U \Sigma V^T $$
El subespacio semántico $S_t$ está definido por los primeros $k$ vectores filas de $V^T$ (los componentes principales). Este subespacio captura las dimensiones de máxima varianza en el uso de la palabra clave.

## 3.2 Selección de Dimensionalidad ($k$)

No todas las dimensiones son informativas; muchas son ruido.
Se empleó el **Análisis Paralelo de Horn**, comparando los autovalores de $X_t$ con los de una matriz de ruido aleatorio de igual tamaño. Se retienen solo las dimensiones donde $\lambda_{obs} > \lambda_{ruido}$. Típicamente, $k \in [3, 6]$.

## 3.3 Alineamiento Temporal (Procrustes Ortogonal)

Los espacios latentes generados independientemente en $t$ y $t+1$ pueden estar rotados arbitrariamente. Para comparar $S_t$ con $S_{t+1}$, debemos alinearlos.
Se busca la matriz de rotación ortogonal $Q$ que minimice la distancia entre los marcos de referencia:
$$ \min_Q || S_{t+1} - S_t Q ||_F $$
Esto permite calcular el **Semantic Drift** real, descartando rotaciones espurias.

## 3.4 Ortogonalización de Anclas (Gram-Schmidt)

Para medir la proyección de la marca sobre conceptos sociopolíticos (Confianza, Inclusión, Riesgo), definimos vectores ancla iniciales $a_1, a_2, a_3$.
Para garantizar independencia estadística (que "Riesgo" no esté correlacionado con "Confianza"), aplicamos ortogonalización:
1.  Norma $\hat{u}_1 = a_1$.
2.  $\hat{u}_2 = a_2 - \text{proj}_{\hat{u}_1}(a_2)$.
3.  Etc.
Esto crea una base ortonormal sobre la cual se puede "radiografiar" la posición de la marca sin colinealidad.
