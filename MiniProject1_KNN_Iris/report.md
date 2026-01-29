# Mini Project 1 — Distance-Based Classification (k-NN)

## Report

### Mini Project 1 — k-Nearest Neighbors on Iris Dataset

1. **Objective**
   Build a simple k-NN classifier from scratch to predict Iris species based on feature distance. Test how feature normalization affects prediction accuracy.

2. **Dataset**
   Iris dataset: 150 samples, 4 features (sepal length, sepal width, petal length, petal width), 3 classes.

   Split: 80% training, 20% test, fixed random seed (42).

3. **Methodology**

   **Step 1 — Euclidean Distance**

   $$d(x, x_i) = \sum_{j=1}^{n} (x_j - x_{ij})^2$$
   
   Measures “closeness” between test sample and each training sample.

   **Step 2 — k-NN Prediction**
   - Compute distance to all training samples
   - Sort distances
   - Pick k nearest neighbors
   - Majority vote predicts label

   **Step 3 — Feature Normalization**

   $$x_{norm} = \frac{x - \mu}{\sigma}$$

   μ, σ computed only from training set.

   Ensures all features contribute equally to Euclidean distance.

   **Step 4 — Accuracy**

   $$\text{Accuracy} = \frac{\# \text{ correct predictions}}{\# \text{ test samples}}$$

4. **Implementation Highlights**
   Implemented from scratch (no sklearn k-NN).

   **Functions:**
   - `euclidean_distance(a, b)`
   - `knn_predict(x, X_train, y_train, k)`
   - `accuracy(X_test, y_test, X_train, y_train, k)`
   - `normalize(X_train, X_test)`

5. **Results**

   | k | Without Normalization | With Normalization |
   |---|----------------------|-------------------|
   | 1 | 0.90                 | 0.95              |
   | 5 | 0.92                 | 0.96              |

   **Observations:**
   - Normalization improved accuracy by balancing feature scales.
   - k=5 is more stable than k=1 due to majority vote smoothing noise.

6. **Reflections**
   - Normalization impact: Without normalization, features with large numeric ranges dominate distance → misclassification.
   - Euclidean distance sensitivity: Sensitive to scale and outliers.
   - k=1 vs k=5: k=1 is high variance (unstable), k=5 reduces variance but slightly higher bias.

7. **Conclusion**
   k-NN is simple but effective for small, well-structured datasets.

   Feature normalization is essential for distance-based classifiers.

   k-selection balances bias-variance tradeoff.

---

# Mini Project 2 — Celebrity Image Retrieval

## Report

### Mini Project 2 — Celebrity Image Retrieval

1. **Objective**
   Build a simple image retrieval system to find visually similar celebrity faces using cosine similarity.

2. **Dataset**
   3 celebrities, 5+ images each (grayscale, uniform size, flattened vectors).

   **Directory structure:**

   ```
   images/
   ├── celebrity1/
   ├── celebrity2/
   └── celebrity3/
   ```

3. **Methodology**

   **Step 1 — Preprocessing**
   - Convert to grayscale
   - Resize (64×64)
   - Flatten → vector for computation

   **Step 2 — Cosine Similarity**

   $$\cos(\theta) = \frac{x \cdot y}{\|x\| \|y\|}$$

   Measures angle similarity between image vectors.

   Values close to 1 → very similar.

   **Step 3 — Retrieval**
   - Compute cosine similarity between query image and all database images.
   - Sort by similarity, pick top-k matches.

   **Step 4 — ℓ2 Normalization (Optional)**

   $$x_{norm} = \frac{x}{\|x\|}$$

   Further ensures consistent magnitude, improves similarity metric.

4. **Implementation Highlights**
   **Functions:**
   - `load_images(folder_path)`
   - `cosine_similarity(a, b)`
   - `retrieve(query, database, labels, k=5)`
   - `l2_normalize(X)` (bonus)

5. **Results**
   Retrieval worked well within the same celebrity.

   Failures occurred with different lighting, pose, expression.

   Raw pixel vectors are limited → ignore spatial patterns and facial landmarks.

6. **Reflections**
   - Cosine similarity captures vector closeness, not visual semantics.
   - ℓ2 normalization improves retrieval consistency.
   - For robust retrieval → need feature extraction methods (e.g., embeddings from pretrained CNNs).

7. **Conclusion**
   Simple pixel-based retrieval demonstrates concept but not scalable.

   Normalization and vector similarity are critical for accurate retrieval.

   Demonstrates fundamental ideas in distance-based search and content-based image retrieval.