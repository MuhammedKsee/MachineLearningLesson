import numpy as np

class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        # Taşma hatalarını önlemek için clipping ekleyelim
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, n_features):
        # Ağırlıkları küçük rastgele değerlerle başlatın
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

    def compute_cost(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        a = self.sigmoid(z)
        
        # Binary cross entropy (daha dengeli hesaplama için epsilon eklenir)
        epsilon = 1e-15
        cost = -1/m * np.sum(y * np.log(a + epsilon) + (1 - y) * np.log(1 - a + epsilon))
        return cost

    def compute_gradients(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        a = self.sigmoid(z)
        
        dw = 1/m * np.dot(X.T, (a - y))
        db = 1/m * np.sum(a - y)
        
        return dw, db

    def fit(self, X, y):
        # Giriş verilerini doğru şekilde hazırlama
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Parametreleri başlat
        n_features = X.shape[1]
        self.initialize_parameters(n_features)
        
        # Gradient descent döngüsü
        for i in range(self.num_iterations):
            # Gradyanları hesapla
            dw, db = self.compute_gradients(X, y)
            
            # Parametreleri güncelle
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Maliyet fonksiyonunu hesapla ve kaydet
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                self.loss_history.append(cost)
        
        return self

    def predict_proba(self, X):
        # Olasılık tahminlerini döndür
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        # 0-1 sınıflandırması için ikili tahminler döndür
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def score(self, X, y):
        # Model doğruluğunu hesapla
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    

        
    