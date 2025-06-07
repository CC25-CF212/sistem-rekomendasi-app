import pandas as pd
import numpy as np
import psycopg2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, Embedding, Flatten, Concatenate, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import hashlib
from datetime import datetime, timezone
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafeLabelEncoder:
    """A wrapper around LabelEncoder that handles unknown categories"""
    def __init__(self):
        self.encoder = LabelEncoder()
        self.classes_ = None
        self.unknown_class = 'unknown'
        
    def fit(self, y):
        # Add 'unknown' to the training data if not present
        y_with_unknown = list(y) + [self.unknown_class]
        self.encoder.fit(y_with_unknown)
        self.classes_ = self.encoder.classes_
        return self
    
    def transform(self, y):
        # Replace unseen categories with 'unknown'
        y_safe = []
        for item in y:
            if item in self.classes_:
                y_safe.append(item)
            else:
                y_safe.append(self.unknown_class)
        return self.encoder.transform(y_safe)
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)

class PostgreSQLContentBasedRecommender:
    def __init__(self, db_config, model_path='models/'):
        self.db_config = db_config
        self.model_path = model_path
        self.model = None
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.similarity_matrix = None
        self.articles_data = None
        self.data_hash = None
        self.scaler_fitted = False
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
    def get_db_connection(self):
        """Create and return database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def load_data_from_db(self):
        """Load data from PostgreSQL database"""
        conn = self.get_db_connection()
        
        try:
            # Load articles data
            articles_query = """
            SELECT id, title, slug, province, city, active, user_id, created_at, updated_at
            FROM "Articles" as articles 
            WHERE active = true
            ORDER BY created_at DESC
            """
            articles_df = pd.read_sql(articles_query, conn)
            
            # Load likes data
            likes_query = """
            SELECT id, article_id, user_id, created_at
            FROM "Article_likes" as article_likes
            """
            likes_df = pd.read_sql(likes_query, conn)
            
            # Load comments data
            comments_query = """
            SELECT id, article_id, user_id, created_at
            from "Article_comments" aS  article_comments
            """
            comments_df = pd.read_sql(comments_query, conn)
            
            logger.info(f"Loaded {len(articles_df)} articles, {len(likes_df)} likes, {len(comments_df)} comments")
            
            return articles_df, likes_df, comments_df
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise
        finally:
            conn.close()
    
    def calculate_data_hash(self, articles_df, likes_df, comments_df):
        """Calculate hash of the data to detect changes"""
        # Include row counts and max timestamps for change detection
        data_string = (
            str(articles_df.shape[0]) + 
            str(likes_df.shape[0]) + 
            str(comments_df.shape[0]) +
            str(articles_df['updated_at'].max()) +
            str(likes_df['created_at'].max() if not likes_df.empty else 'no_likes') +
            str(comments_df['created_at'].max() if not comments_df.empty else 'no_comments')
        )
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def preprocess_data(self, articles_df, likes_df, comments_df):
        """Preprocess the loaded data"""
        # Ensure all IDs are strings
        articles_df['id'] = articles_df['id'].astype(str)
        likes_df['article_id'] = likes_df['article_id'].astype(str)
        comments_df['article_id'] = comments_df['article_id'].astype(str)
        
        # Calculate likes per article
        article_likes = likes_df.groupby('article_id').size().reset_index(name='likes_count')
        
        # Calculate comments per article
        article_comments = comments_df.groupby('article_id').size().reset_index(name='comments_count')
        
        # Combine data with articles
        articles_enriched = articles_df.copy()
        
        # Add likes count
        articles_enriched = articles_enriched.merge(
            article_likes, left_on='id', right_on='article_id', how='left'
        )
        articles_enriched['likes_count'] = articles_enriched['likes_count'].fillna(0)
        
        # Add comments count
        articles_enriched = articles_enriched.merge(
            article_comments, left_on='id', right_on='article_id', how='left'
        )
        articles_enriched['comments_count'] = articles_enriched['comments_count'].fillna(0)
        
        # Clean up merge columns
        articles_enriched = articles_enriched.drop(['article_id_x', 'article_id_y'], axis=1, errors='ignore')
        
        # Fill missing categorical values BEFORE creating text features
        articles_enriched['province'] = articles_enriched['province'].fillna('unknown')
        articles_enriched['city'] = articles_enriched['city'].fillna('unknown')
        articles_enriched['title'] = articles_enriched['title'].fillna('')
        
        # Create text features
        articles_enriched['text_features'] = (
            articles_enriched['title'] + ' ' + 
            articles_enriched['province'] + ' ' + 
            articles_enriched['city']
        )
        
        # Calculate engagement score
        articles_enriched['engagement_score'] = (
            articles_enriched['likes_count'] + (2 * articles_enriched['comments_count'])
        )
        
        # Add time-based features
        articles_enriched['created_at'] = pd.to_datetime(articles_enriched['created_at'])
        
        # Handle timezone-aware datetime comparison
        if articles_enriched['created_at'].dt.tz is not None:
            now = datetime.now(timezone.utc)
        else:
            now = datetime.now()
        
        articles_enriched['days_since_creation'] = (
            (now - articles_enriched['created_at']).dt.days
        )
        
        # Calculate recency score (newer articles get higher scores)
        max_days = articles_enriched['days_since_creation'].max()
        if max_days > 0:
            articles_enriched['recency_score'] = 1 - (articles_enriched['days_since_creation'] / max_days)
        else:
            articles_enriched['recency_score'] = 1.0
        
        logger.info(f"Preprocessed data shape: {articles_enriched.shape}")
        return articles_enriched
    
    def prepare_features(self, articles_enriched, is_training=True):
        """Prepare features for the model"""
        # Text features using TF-IDF
        if is_training or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(articles_enriched['text_features'])
        else:
            tfidf_features = self.tfidf_vectorizer.transform(articles_enriched['text_features'])
        
        # Categorical features with improved handling
        categorical_features = []
        for col in ['province', 'city']:
            if is_training or col not in self.label_encoders:
                # Create new encoder for training
                self.label_encoders[col] = SafeLabelEncoder()
                encoded = self.label_encoders[col].fit_transform(articles_enriched[col])
            else:
                # Use existing encoder for inference
                encoded = self.label_encoders[col].transform(articles_enriched[col])
            
            categorical_features.append(encoded.reshape(-1, 1))
        
        # Numerical features
        numerical_cols = ['likes_count', 'comments_count', 'engagement_score', 'recency_score']
        numerical_features = articles_enriched[numerical_cols].values
        
        if is_training or not self.scaler_fitted:
            numerical_features = self.scaler.fit_transform(numerical_features)
            self.scaler_fitted = True
        else:
            numerical_features = self.scaler.transform(numerical_features)
        
        return tfidf_features, categorical_features, numerical_features
    
    def build_model(self, tfidf_dim, categorical_dims, numerical_dim):
        """Build the TensorFlow model"""
        # Text input
        text_input = Input(shape=(tfidf_dim,), name='text_input')
        text_dense = Dense(128, activation='relu')(text_input)
        text_dense = Dense(64, activation='relu')(text_dense)
        
        # Categorical inputs
        categorical_inputs = []
        categorical_embeddings = []
        
        for i, dim in enumerate(categorical_dims):
            cat_input = Input(shape=(1,), name=f'cat_input_{i}')
            embedding_dim = min(50, max(4, dim//2))  # Ensure minimum embedding dimension
            cat_embedding = Embedding(dim, embedding_dim)(cat_input)
            cat_embedding = Flatten()(cat_embedding)
            categorical_inputs.append(cat_input)
            categorical_embeddings.append(cat_embedding)
        
        # Numerical input
        num_input = Input(shape=(numerical_dim,), name='num_input')
        num_dense = Dense(32, activation='relu')(num_input)
        
        # Combine all features
        if categorical_embeddings:
            combined = Concatenate()([text_dense] + categorical_embeddings + [num_dense])
        else:
            combined = Concatenate()([text_dense, num_dense])
        
        # Final layers
        combined = Dense(128, activation='relu')(combined)
        combined = Dense(64, activation='relu')(combined)
        output = Dense(32, activation='linear', name='content_embedding')(combined)
        
        # Create model
        all_inputs = [text_input] + categorical_inputs + [num_input]
        model = tf.keras.Model(inputs=all_inputs, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def should_retrain(self, current_hash):
        """Check if model should be retrained"""
        # Always retrain if no previous hash exists
        if self.data_hash is None:
            logger.info("No previous data hash found - will train new model")
            return True
            
        # Check if data has changed
        if self.data_hash != current_hash:
            logger.info(f"Data hash changed: {self.data_hash} -> {current_hash}")
            return True
        
        # Check if model files exist
        model_files = [
            os.path.join(self.model_path, 'tfidf_vectorizer.pkl'),
            os.path.join(self.model_path, 'scaler.pkl'),
            os.path.join(self.model_path, 'label_encoders.pkl')
        ]
        
        keras_model_exists = os.path.exists(os.path.join(self.model_path, 'content_model.keras'))
        h5_model_exists = os.path.exists(os.path.join(self.model_path, 'content_model.h5'))
        
        model_exists = keras_model_exists or h5_model_exists
        other_files_exist = all(os.path.exists(f) for f in model_files)
        
        if not (model_exists and other_files_exist):
            logger.info("Model files missing - will retrain")
            return True
            
        logger.info("No retraining needed - data unchanged and model exists")
        return False
    
    def fit(self, force_retrain=False):
        """Fit the recommender model"""
        try:
            # Load data from database
            logger.info("Loading data from database...")
            articles_df, likes_df, comments_df = self.load_data_from_db()
            
            # Calculate data hash
            current_hash = self.calculate_data_hash(articles_df, likes_df, comments_df)
            
            # Check if retraining is needed
            if not force_retrain and not self.should_retrain(current_hash):
                logger.info("Loading existing model...")
                self.load_model()
                return
            
            logger.info("Training model with new/updated data...")
            
            # Reset state for retraining
            if force_retrain:
                self.tfidf_vectorizer = None
                self.label_encoders = {}
                self.scaler = StandardScaler()
                self.scaler_fitted = False
            
            # Preprocess data
            self.articles_data = self.preprocess_data(articles_df, likes_df, comments_df)
            
            if len(self.articles_data) == 0:
                raise ValueError("No articles found for training")
            
            # Prepare features
            tfidf_features, categorical_features, numerical_features = self.prepare_features(
                self.articles_data, is_training=True
            )
            
            # Build model
            tfidf_dim = tfidf_features.shape[1]
            categorical_dims = [len(encoder.classes_) for encoder in self.label_encoders.values()]
            numerical_dim = numerical_features.shape[1]
            
            logger.info(f"Model dimensions - TF-IDF: {tfidf_dim}, Categorical: {categorical_dims}, Numerical: {numerical_dim}")
            
            self.model = self.build_model(tfidf_dim, categorical_dims, numerical_dim)
            
            # Prepare training data
            X_text = tfidf_features.toarray()
            X_categorical = [cat_feat.flatten() for cat_feat in categorical_features]
            X_numerical = numerical_features
            
            # Create target (autoencoder-like training)
            y = np.concatenate([X_text[:, :32], X_numerical], axis=1)
            if y.shape[1] > 32:
                y = y[:, :32]
            elif y.shape[1] < 32:
                y = np.pad(y, ((0, 0), (0, 32 - y.shape[1])), mode='constant')
            
            # Train model
            X_train = [X_text] + X_categorical + [X_numerical]
            
            logger.info("Starting model training...")
            history = self.model.fit(
                X_train, y,
                epochs=50,
                batch_size=min(32, len(self.articles_data)),
                validation_split=0.2 if len(self.articles_data) > 5 else 0,
                verbose=1
            )
            
            # Calculate similarity matrix
            logger.info("Calculating similarity matrix...")
            embeddings = self.model.predict(X_train)
            self.similarity_matrix = cosine_similarity(embeddings)
            
            # Update data hash
            self.data_hash = current_hash
            
            # Save model
            self.save_model()
            
            logger.info("Model training completed and saved successfully!")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def recommend(self, article_id, top_n=5):
        """Generate recommendations for a given article"""
        if self.model is None or self.articles_data is None:
            logger.info("Model not loaded, fitting first...")
            self.fit()
        
        # Convert article_id to string for consistency
        article_id = str(article_id)
        
        # Find article index
        article_idx = self.articles_data[self.articles_data['id'] == article_id].index
        
        if len(article_idx) == 0:
            logger.warning(f"Article ID {article_id} not found")
            return pd.DataFrame()
        
        article_idx = article_idx[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[article_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations (excluding the article itself)
        sim_scores = sim_scores[1:top_n+1]
        article_indices = [i[0] for i in sim_scores]
        
        # Return recommended articles
        recommendations = self.articles_data.iloc[article_indices].copy()
        recommendations['similarity_score'] = [score[1] for score in sim_scores]
        
        return recommendations[['id', 'title', 'province', 'city', 'engagement_score', 'similarity_score']]
    
    def save_model(self):
        """Save the trained model and preprocessors"""
        try:
            # Save TensorFlow model in native Keras format
            self.model.save(os.path.join(self.model_path, 'content_model.keras'))
            
            # Save preprocessors
            with open(os.path.join(self.model_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            with open(os.path.join(self.model_path, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(os.path.join(self.model_path, 'label_encoders.pkl'), 'wb') as f:
                pickle.dump(self.label_encoders, f)
            
            with open(os.path.join(self.model_path, 'similarity_matrix.pkl'), 'wb') as f:
                pickle.dump(self.similarity_matrix, f)
            
            with open(os.path.join(self.model_path, 'articles_data.pkl'), 'wb') as f:
                pickle.dump(self.articles_data, f)
            
            with open(os.path.join(self.model_path, 'data_hash.pkl'), 'wb') as f:
                pickle.dump(self.data_hash, f)
            
            logger.info("Model and preprocessors saved successfully!")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            # Try to load new format first, then fallback to old format
            keras_model_path = os.path.join(self.model_path, 'content_model.keras')
            h5_model_path = os.path.join(self.model_path, 'content_model.h5')
            
            if os.path.exists(keras_model_path):
                self.model = load_model(keras_model_path)
            elif os.path.exists(h5_model_path):
                self.model = load_model(h5_model_path)
                logger.warning("Loaded model from legacy HDF5 format")
            else:
                raise FileNotFoundError("No model file found")
            
            # Load preprocessors
            with open(os.path.join(self.model_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            with open(os.path.join(self.model_path, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
                self.scaler_fitted = True
            
            with open(os.path.join(self.model_path, 'label_encoders.pkl'), 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            if os.path.exists(os.path.join(self.model_path, 'similarity_matrix.pkl')):
                with open(os.path.join(self.model_path, 'similarity_matrix.pkl'), 'rb') as f:
                    self.similarity_matrix = pickle.load(f)
            
            if os.path.exists(os.path.join(self.model_path, 'articles_data.pkl')):
                with open(os.path.join(self.model_path, 'articles_data.pkl'), 'rb') as f:
                    self.articles_data = pickle.load(f)
            
            if os.path.exists(os.path.join(self.model_path, 'data_hash.pkl')):
                with open(os.path.join(self.model_path, 'data_hash.pkl'), 'rb') as f:
                    self.data_hash = pickle.load(f)
            
            logger.info("Model and preprocessors loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
