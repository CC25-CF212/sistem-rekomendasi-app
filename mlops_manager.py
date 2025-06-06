import schedule
import time
import threading
from datetime import datetime, timedelta
import logging
import psycopg2
from typing import Dict, Any
import json
import os

class MLOpsManager:
    def __init__(self, db_config: Dict[str, Any], recommender_instance):
        self.db_config = db_config
        self.recommender = recommender_instance
        self.logger = logging.getLogger(__name__)
        self.monitoring_metrics = {
            'last_training': None,
            'data_changes_detected': 0,
            'model_performance': {},
            'training_history': []
        }
        
    def check_data_changes(self) -> bool:
        """Check if there are significant data changes that warrant retraining"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get current data stats
            stats_query = """
            SELECT 
                (SELECT COUNT(*) FROM "Articles" WHERE active = true) as article_count,
                (SELECT COUNT(*) FROM "Article_likes" WHERE created_at > NOW() - INTERVAL '1 day') as recent_likes,
                (SELECT COUNT(*) FROM "Article_comments" WHERE created_at > NOW() - INTERVAL '1 day') as recent_comments,
                (SELECT MAX(updated_at) FROM "Articles") as last_article_update
            """
            
            cursor.execute(stats_query)
            current_stats = cursor.fetchone()
            
            # Load previous stats
            stats_file = os.path.join(self.recommender.model_path, 'data_stats.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    previous_stats = json.load(f)
            else:
                previous_stats = {}
            
            # Check for significant changes
            changes_detected = False
            
            # 1. New articles added (>5% increase)
            if previous_stats.get('article_count', 0) > 0:
                article_growth = (current_stats[0] - previous_stats.get('article_count', 0)) / previous_stats.get('article_count', 1)
                if article_growth > 0.05:  # 5% increase
                    changes_detected = True
                    self.logger.info(f"Significant article growth detected: {article_growth:.2%}")
            
            # 2. High engagement activity (>50 new interactions in last day)
            total_recent_engagement = current_stats[1] + current_stats[2]
            if total_recent_engagement > 50:
                changes_detected = True
                self.logger.info(f"High recent engagement detected: {total_recent_engagement} interactions")
            
            # 3. Articles updated recently
            if current_stats[3] and previous_stats.get('last_article_update'):
                last_update = datetime.fromisoformat(str(current_stats[3]))
                prev_update = datetime.fromisoformat(previous_stats['last_article_update'])
                if last_update > prev_update:
                    changes_detected = True
                    self.logger.info("Recent article updates detected")
            
            # Save current stats
            current_stats_dict = {
                'article_count': current_stats[0],
                'recent_likes': current_stats[1],
                'recent_comments': current_stats[2],
                'last_article_update': str(current_stats[3]) if current_stats[3] else None,
                'last_check': datetime.now().isoformat()
            }
            
            with open(stats_file, 'w') as f:
                json.dump(current_stats_dict, f)
            
            if changes_detected:
                self.monitoring_metrics['data_changes_detected'] += 1
            
            conn.close()
            return changes_detected
            
        except Exception as e:
            self.logger.error(f"Error checking data changes: {e}")
            return False
    
    def evaluate_model_performance(self) -> Dict[str, float]:
        """Evaluate current model performance"""
        try:
            # Simple performance metrics
            if self.recommender.similarity_matrix is not None:
                avg_similarity = float(self.recommender.similarity_matrix.mean())
                max_similarity = float(self.recommender.similarity_matrix.max())
                
                # Check for model degradation
                performance_file = os.path.join(self.recommender.model_path, 'performance_history.json')
                if os.path.exists(performance_file):
                    with open(performance_file, 'r') as f:
                        history = json.load(f)
                    
                    # Compare with previous performance
                    if history and len(history) > 0:
                        last_performance = history[-1]['avg_similarity']
                        performance_drop = (last_performance - avg_similarity) / last_performance
                        
                        if performance_drop > 0.1:  # 10% performance drop
                            self.logger.warning(f"Model performance degradation detected: {performance_drop:.2%}")
                            return {'needs_retrain': True, 'avg_similarity': avg_similarity}
                
                # Save current performance
                current_performance = {
                    'timestamp': datetime.now().isoformat(),
                    'avg_similarity': avg_similarity,
                    'max_similarity': max_similarity
                }
                
                if os.path.exists(performance_file):
                    with open(performance_file, 'r') as f:
                        history = json.load(f)
                else:
                    history = []
                
                history.append(current_performance)
                # Keep only last 30 records
                history = history[-30:]
                
                with open(performance_file, 'w') as f:
                    json.dump(history, f)
                
                return {'needs_retrain': False, 'avg_similarity': avg_similarity}
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {e}")
            
        return {'needs_retrain': False, 'avg_similarity': 0.0}
    
    def trigger_retraining(self):
        """Trigger model retraining with proper logging"""
        try:
            self.logger.info("Starting scheduled retraining...")
            start_time = datetime.now()
            
            # Retrain model
            self.recommender.fit(force_retrain=True)
            
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            # Log training completion
            training_record = {
                'timestamp': end_time.isoformat(),
                'duration_seconds': training_duration,
                'trigger': 'scheduled',
                'success': True
            }
            
            self.monitoring_metrics['training_history'].append(training_record)
            self.monitoring_metrics['last_training'] = end_time.isoformat()
            
            self.logger.info(f"Retraining completed in {training_duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Retraining failed: {e}")
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': 0,
                'trigger': 'scheduled',
                'success': False,
                'error': str(e)
            }
            self.monitoring_metrics['training_history'].append(training_record)
    
    def scheduled_check(self):
        """Scheduled check for retraining needs"""
        self.logger.info("Running scheduled model check...")
        
        # Check data changes
        data_changed = self.check_data_changes()
        
        # Check model performance
        performance = self.evaluate_model_performance()
        
        # Decide if retraining is needed
        should_retrain = data_changed or performance.get('needs_retrain', False)
        
        if should_retrain:
            self.logger.info("Retraining triggered by automated checks")
            self.trigger_retraining()
        else:
            self.logger.info("No retraining needed")
    
    def setup_scheduler(self):
        """Setup automatic scheduling"""
        # Daily checks
        schedule.every().day.at("02:00").do(self.scheduled_check)
        
        # Weekly full retraining (optional)
        schedule.every().sunday.at("03:00").do(self.trigger_retraining)
        
        # Hourly light checks during business hours
        for hour in range(9, 18):  # 9 AM to 6 PM
            schedule.every().day.at(f"{hour:02d}:00").do(self.check_data_changes)
    
    def start_monitoring(self):
        """Start the monitoring service"""
        self.setup_scheduler()
        
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        # Run scheduler in background thread
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        self.logger.info("MLOps monitoring started")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'last_training': self.monitoring_metrics['last_training'],
            'data_changes_detected': self.monitoring_metrics['data_changes_detected'],
            'recent_training_history': self.monitoring_metrics['training_history'][-5:],
            'next_scheduled_check': str(schedule.next_run()) if schedule.jobs else None
        }

# Database Triggers (SQL) - untuk real-time detection
DATABASE_TRIGGERS_SQL = """
-- Create function to log data changes
CREATE OR REPLACE FUNCTION log_data_change() 
RETURNS TRIGGER AS $$
BEGIN
    -- Insert into monitoring table
    INSERT INTO ml_model_triggers (table_name, action, triggered_at)
    VALUES (TG_TABLE_NAME, TG_OP, NOW());
    
    -- If significant changes, flag for retraining
    UPDATE ml_model_status 
    SET needs_retraining = TRUE, 
        last_data_change = NOW()
    WHERE model_name = 'content_recommender';
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create triggers for articles table
DROP TRIGGER IF EXISTS article_change_trigger ON articles;
CREATE TRIGGER article_change_trigger
    AFTER INSERT OR UPDATE OR DELETE ON articles
    FOR EACH ROW EXECUTE FUNCTION log_data_change();

-- Create triggers for likes table
DROP TRIGGER IF EXISTS likes_change_trigger ON article_likes;
CREATE TRIGGER likes_change_trigger
    AFTER INSERT OR DELETE ON article_likes
    FOR EACH ROW EXECUTE FUNCTION log_data_change();

-- Create triggers for comments table
DROP TRIGGER IF EXISTS comments_change_trigger ON article_comments;
CREATE TRIGGER comments_change_trigger
    AFTER INSERT OR DELETE ON article_comments
    FOR EACH ROW EXECUTE FUNCTION log_data_change();

-- Create monitoring tables
CREATE TABLE IF NOT EXISTS ml_model_triggers (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100),
    action VARCHAR(10),
    triggered_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_model_status (
    model_name VARCHAR(100) PRIMARY KEY,
    needs_retraining BOOLEAN DEFAULT FALSE,
    last_training TIMESTAMP,
    last_data_change TIMESTAMP,
    performance_metrics JSONB
);

-- Initialize status
INSERT INTO ml_model_status (model_name) 
VALUES ('content_recommender') 
ON CONFLICT (model_name) DO NOTHING;
"""