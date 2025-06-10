# Content-Based Recommender API

Sistem rekomendasi artikel berbasis konten dengan fitur MLOps, evaluasi model otomatis, dan hyperparameter tuning.

## 🚀 Fitur Utama

- **Content-Based Recommendation**: Rekomendasi artikel berdasarkan kesamaan konten
- **MLOps Integration**: Monitoring otomatis dan retraining model
- **Model Evaluation**: Evaluasi performa model dengan berbagai metrik
- **Hyperparameter Tuning**: Optimasi parameter model secara otomatis
- **Real-time API**: REST API untuk rekomendasi real-time
- **Batch Processing**: Rekomendasi untuk multiple artikel sekaligus

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Git

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd content-recommender-api
```

### 2. Setup Python Environment

```bash
# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup

#### Buat Database dan Tables

```sql
-- Buat database
CREATE DATABASE article_recommender;

-- Connect ke database dan buat tables
\c article_recommender;

-- Table Articles
CREATE TABLE "Articles" (
    id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT,
    province VARCHAR(100),
    city VARCHAR(100),
    engagement_score FLOAT DEFAULT 0.0,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Table Article Likes
CREATE TABLE "Article_likes" (
    id SERIAL PRIMARY KEY,
    article_id VARCHAR(50) REFERENCES "Articles"(id),
    user_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table Article Comments
CREATE TABLE "Article_comments" (
    id SERIAL PRIMARY KEY,
    article_id VARCHAR(50) REFERENCES "Articles"(id),
    user_id VARCHAR(50),
    comment_text TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### Setup MLOps Monitoring (Optional)

```sql
-- Jalankan script untuk monitoring triggers
-- (Copy dari mlops_manager.py - DATABASE_TRIGGERS_SQL)

CREATE OR REPLACE FUNCTION log_data_change() 
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO ml_model_triggers (table_name, action, triggered_at)
    VALUES (TG_TABLE_NAME, TG_OP, NOW());
    
    UPDATE ml_model_status 
    SET needs_retraining = TRUE, 
        last_data_change = NOW()
    WHERE model_name = 'content_recommender';
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create triggers dan monitoring tables
-- (Lihat lengkap di mlops_manager.py)
```

### 4. Environment Configuration

Buat file `.env` di root directory:

```env
# Database Configuration
DB_HOST=localhost
DB_NAME=article_recommender
DB_USER=your_username
DB_PASSWORD=your_password
DB_PORT=5432

# Optional: Model Configuration
MODEL_PATH=./models
LOG_LEVEL=INFO
```

### 5. Sample Data (Optional)

```sql
-- Insert sample articles
INSERT INTO "Articles" (id, title, content, province, city, engagement_score) VALUES
('art1', 'Tips Traveling Jakarta', 'Panduan lengkap wisata Jakarta...', 'DKI Jakarta', 'Jakarta', 8.5),
('art2', 'Kuliner Yogyakarta', 'Rekomendasi makanan khas Yogya...', 'DI Yogyakarta', 'Yogyakarta', 9.2),
('art3', 'Pantai Bali Terbaik', 'Daftar pantai indah di Bali...', 'Bali', 'Denpasar', 9.8),
('art4', 'Gunung di Jawa Barat', 'Panduan mendaki gunung di Jabar...', 'Jawa Barat', 'Bandung', 7.9),
('art5', 'Wisata Danau Toba', 'Keindahan Danau Toba Sumatera...', 'Sumatera Utara', 'Medan', 8.7);

-- Insert sample interactions
INSERT INTO "Article_likes" (article_id, user_id) VALUES
('art1', 'user1'), ('art1', 'user2'), ('art2', 'user1'),
('art3', 'user3'), ('art3', 'user4'), ('art4', 'user2');

INSERT INTO "Article_comments" (article_id, user_id, comment_text) VALUES
('art1', 'user1', 'Artikel sangat membantu!'),
('art2', 'user2', 'Rekomendasinya mantap'),
('art3', 'user3', 'Pengen kesana!');
```

## 🏃‍♂️ Running the Application

### 1. Start API Server

```bash
# Dari root directory
python main.py

# Atau menggunakan uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server akan berjalan di: `http://localhost:8000`

### 2. API Documentation

Akses interactive API docs di: `http://localhost:8000/docs`

## 🧪 Testing

### 1. Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "last_training": "2024-01-15T10:30:00",
  "data_freshness": "real-time",
  "uptime_seconds": 3600.5
}
```

### 2. Get Recommendations

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "article_id": "art1",
    "top_n": 5,
    "include_metadata": true
  }'
```

Expected response:
```json
{
  "article_id": "art1",
  "recommendations": [
    {
      "id": "art4",
      "title": "Gunung di Jawa Barat",
      "similarity_score": 0.85,
      "province": "Jawa Barat",
      "city": "Bandung",
      "engagement_score": 7.9
    }
  ],
  "total_found": 4,
  "generated_at": "2024-01-15T12:00:00",
  "model_version": "1.0.0"
}
```

### 3. Batch Recommendations

```bash
curl -X POST "http://localhost:8000/recommend/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "article_ids": ["art1", "art2", "art3"],
    "top_n": 3
  }'
```

### 4. Model Status

```bash
curl -X GET "http://localhost:8000/model/status"
```

### 5. Manual Retraining

```bash
curl -X POST "http://localhost:8000/model/retrain?force=true"
```

## 📊 Model Evaluation & Tuning

### 1. Run Model Evaluation

```bash
curl -X POST "http://localhost:8000/evaluation/run" \
  -H "Content-Type: application/json" \
  -d '{
    "test_size": 100,
    "k_values": [5, 10],
    "include_diversity": true,
    "save_results": true
  }'
```

### 2. Check Evaluation Status

```bash
curl -X GET "http://localhost:8000/evaluation/status"
```

### 3. Get Evaluation Results

```bash
curl -X GET "http://localhost:8000/evaluation/results"
```

### 4. Run Hyperparameter Tuning

```bash
curl -X POST "http://localhost:8000/tuning/run" \
  -H "Content-Type: application/json" \
  -d '{
    "max_combinations": 20,
    "n_jobs": 2,
    "optimization_metric": "composite"
  }'
```

### 5. Check Tuning Status

```bash
curl -X GET "http://localhost:8000/tuning/status"
```

### 6. Get Best Parameters

```bash
curl -X GET "http://localhost:8000/tuning/best-params"
```

### 7. Apply Best Parameters

```bash
curl -X POST "http://localhost:8000/tuning/apply-best-params"
```

## 🔍 Monitoring & MLOps

### 1. Get Metrics

```bash
curl -X GET "http://localhost:8000/metrics"
```

### 2. Check Model Info

```bash
curl -X GET "http://localhost:8000/evaluation/model-info"
```

### 3. Compare Evaluations

```bash
curl -X POST "http://localhost:8000/evaluation/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "baseline_evaluation_id": "eval_1705320000",
    "current_evaluation_id": "eval_1705406400",
    "metrics_to_compare": ["f1_score", "precision", "recall", "ndcg"]
  }'
```

## 📁 Project Structure

```
content-recommender-api/
├── main.py                 # Main FastAPI application
├── evaluation_api.py       # Evaluation & tuning endpoints
├── mlops_manager.py        # MLOps monitoring system
├── postgres_recommender.py # Core recommender class
├── model_evaluator.py      # Model evaluation utilities
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── models/                # Model storage directory
│   ├── similarity_matrix.pkl
│   ├── articles_data.pkl
│   └── model_metadata.json
└── logs/                  # Application logs
```

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**
   ```
   Error: could not connect to server
   ```
   - Pastikan PostgreSQL berjalan
   - Cek konfigurasi di `.env`
   - Verify database credentials

2. **Model Not Loading**
   ```
   HTTPException 503: Model is not loaded yet
   ```
   - Tunggu proses training selesai
   - Check logs untuk error details
   - Ensure database has sufficient data

3. **Out of Memory During Training**
   ```
   MemoryError during model training
   ```
   - Reduce batch size
   - Limit number of articles processed
   - Use smaller similarity matrix

4. **Evaluation/Tuning Stuck**
   ```
   Process appears to be stuck
   ```
   - Check `/evaluation/status` endpoint
   - Restart server if needed
   - Reduce test_size or max_combinations

### Debug Mode

Untuk debugging yang lebih detail:

```bash
# Set log level ke DEBUG
export LOG_LEVEL=DEBUG

# Jalankan dengan verbose logging
python main.py --log-level debug
```

## 🔧 Configuration Options

### Model Parameters (dalam postgres_recommender.py)

```python
# TF-IDF Parameters
max_features = 5000
ngram_range = (1, 2)
stop_words = 'english'
min_df = 2
max_df = 0.95

# Similarity threshold
similarity_threshold = 0.1
```

### MLOps Configuration (dalam mlops_manager.py)

```python
# Monitoring intervals
daily_check_time = "02:00"
weekly_retrain_time = "03:00"
business_hours_checks = range(9, 18)

# Change detection thresholds
article_growth_threshold = 0.05  # 5%
engagement_threshold = 50  # interactions per day
performance_drop_threshold = 0.1  # 10%
```

## 📈 Performance Optimization

### 1. Database Indexing

```sql
-- Indexes untuk performa query
CREATE INDEX idx_articles_active ON "Articles"(active);
CREATE INDEX idx_articles_updated ON "Articles"(updated_at);
CREATE INDEX idx_likes_article ON "Article_likes"(article_id);
CREATE INDEX idx_comments_article ON "Article_comments"(article_id);
```

### 2. Caching Strategy

- Model similarity matrix di-cache dalam memory
- Database connection pooling
- Response caching untuk frequent requests

### 3. Scaling Considerations

- Gunakan Redis untuk caching
- Database read replicas
- Load balancer untuk multiple API instances
- Async processing untuk batch operations

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details.
