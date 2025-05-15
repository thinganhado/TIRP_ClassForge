import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sqlalchemy import create_engine, text

# Try to import config
try:
    from app.config import Config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("[INFO] App config not found, using direct connection parameters")

class DatabaseLoader:
    def __init__(self, direct_connect=False):
        self.data_cache = {}
        self.db_available = False
        self.direct_connect = direct_connect

        if direct_connect:
            try:
                if HAS_CONFIG:
                    connection_string = f"mysql+pymysql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
                else:
                    db_host = "database-tirp.c1gieoo4asys.us-east-1.rds.amazonaws.com"
                    db_user = "admin"
                    db_port = 3306
                    db_name = "tirp"
                    connection_string = f"mysql+pymysql://{db_user}@{db_host}:{db_port}/{db_name}"
                
                self.engine = create_engine(connection_string, pool_pre_ping=True, pool_size=10, max_overflow=20)
                with self.engine.connect() as conn:
                    if conn.execute(text("SELECT 1")).scalar() == 1:
                        self.db_available = True
                        print("[INFO] Connected to RDS")
            except Exception as e:
                print(f"[ERROR] DB connection failed: {e}")
        else:
            try:
                from app.database import class_queries, student_queries, softcons_queries
                from app import db
                self.db = db
                self.class_queries = class_queries
                self.student_queries = student_queries
                self.softcons_queries = softcons_queries
                self.db_available = True
                print("[INFO] Connected via Flask")
            except Exception as e:
                print(f"[ERROR] Flask DB connection failed: {e}")

    def _execute_query(self, query, params=None):
        if not self.db_available:
            return None
        try:
            if self.direct_connect:
                with self.engine.connect() as conn:
                    return conn.execute(text(query), params or {})
            else:
                return self.db.session.execute(text(query), params or {})
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return None

    def load_data(self, password=None, use_cache=True):
        """
        Load data from cache if available, otherwise from database
        
        Args:
            password (str, optional): Database password for direct connection if not using config
            use_cache (bool): Whether to try loading from cache first (default: True)
        """
        # Check for cached data first
        cache_dir = 'app/cache'
        if use_cache and os.path.exists(cache_dir):
            cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet')]
            if cached_files:
                print("[INFO] Loading from cached Parquet files")
                for file in cached_files:
                    key = file.replace('.parquet', '')
                    try:
                        self.data_cache[key] = pd.read_parquet(f'{cache_dir}/{file}')
                        print(f"[INFO] Loaded {key} from cache")
                    except Exception as e:
                        print(f"[ERROR] Failed to load {key} from cache: {e}")
                
                if self.data_cache:
                    print(f"[INFO] Successfully loaded {len(self.data_cache)} datasets from cache")
                    return self.data_cache

        if not self.db_available:
            print("[ERROR] DB not available and no cache found")
            return {}

        if self.direct_connect and not HAS_CONFIG and password:
            try:
                connection_string = f"mysql+pymysql://admin:{password}@database-tirp.c1gieoo4asys.us-east-1.rds.amazonaws.com:3306/tirp"
                self.engine = create_engine(connection_string, pool_pre_ping=True, pool_size=10, max_overflow=20)
                print("[INFO] Updated DB connection with password")
            except Exception as e:
                print(f"[ERROR] Connection update failed: {e}")

        try:
            # Load participants
            students_df = pd.read_sql("""
                SELECT student_id, first_name, last_name, email, house,
                       perc_effort, attendance, perc_academic, complete_years
                FROM participants
                WHERE student_id <> 'student_id'
            """, self.engine)
            self.data_cache['students'] = students_df.to_dict(orient='records')
            print(f"[INFO] Loaded {len(students_df)} students")

            # Load wellbeing tables
            mental_df = pd.read_sql("SELECT * FROM mental_wellbeing WHERE mental_wellbeing_percent IS NOT NULL", self.engine)
            academic_df = pd.read_sql("SELECT * FROM academic_wellbeing WHERE academic_wellbeing_percent IS NOT NULL", self.engine)
            social_df = pd.read_sql("SELECT * FROM social_wellbeing WHERE social_wellbeing_percent IS NOT NULL", self.engine)

            # Load responses in bulk
            responses_df = pd.read_sql("""
                SELECT student_id, school_support_engage6, growthmindset
                FROM responses
            """, self.engine)
            responses_map = responses_df.set_index('student_id').to_dict(orient='index')

            # Build wellbeing dataset
            wellbeing_data = []
            for _, student in students_df.iterrows():
                sid = student.student_id
                mental = mental_df.loc[mental_df.student_id == sid, 'mental_wellbeing_percent'].values
                mental_val = mental[0] if len(mental) else 0
                label = 'High Wellbeing' if mental_val > 50 else 'Low Wellbeing'

                school_support = responses_map.get(sid, {}).get('school_support_engage6', 0)
                growth = responses_map.get(sid, {}).get('growthmindset', 0)

                wellbeing_data.append({
                    'Participant-ID': sid,
                    'wellbeing_label': label,
                    'School_support_engage6': school_support,
                    'GrowthMindset': growth
                })
            self.data_cache['wellbeing'] = pd.DataFrame(wellbeing_data)
            print(f"[INFO] Created wellbeing data for {len(wellbeing_data)} students")

            # Load class allocations
            allocations_df = pd.read_sql("SELECT class_id, student_id FROM allocations", self.engine)
            self.data_cache['class_ids'] = allocations_df['class_id'].unique().tolist()

            # GPA data
            gpa_data = []
            for _, row in students_df.iterrows():
                gpa = row.perc_academic or 0
                bin_val = 3 if gpa > 75 else 2 if gpa > 50 else 1 if gpa > 25 else 0
                gpa_data.append({
                    'Student_ID': row.student_id,
                    'Predicted_GPA': gpa,
                    'GPA_Bin': bin_val
                })
            self.data_cache['gpa'] = pd.DataFrame(gpa_data)

            # Disrespect data
            disrespect_df = pd.read_sql("SELECT * FROM net_disrespect", self.engine)
            bully_counts = disrespect_df['source_student_id'].value_counts()

            bullying_data = []
            for sid, count in bully_counts.items():
                is_bully = 1 if count > 2 else 0
                student_class = allocations_df.loc[allocations_df.student_id == sid, 'class_id'].values
                bullying_data.append({
                    'Student_ID': sid,
                    'Community_ID': student_class[0] if len(student_class) else 0,
                    'Is_Bully': is_bully,
                    'Primary_Bully_ID': sid if is_bully else None
                })
            self.data_cache['bullying'] = pd.DataFrame(bullying_data)

            # Social network: friendships & influence
            friends_df = pd.read_sql("SELECT * FROM net_friends", self.engine)
            influence_df = pd.read_sql("SELECT * FROM net_influential", self.engine)
            friend_counts = friends_df['source_student_id'].value_counts()
            influence_counts = influence_df['source_student_id'].value_counts()

            # Social clustering
            social_data = []
            for _, row in social_df.iterrows():
                sid = row.student_id
                score = row.social_wellbeing_percent
                cluster = 1 if score > 50 and (influence_counts.get(sid, 0) > 2 or friend_counts.get(sid, 0) > 2) else 0
                social_data.append({
                    'Participant-ID': sid,
                    'cluster': cluster,
                    'social_wellbeing': score
                })
            self.data_cache['social'] = pd.DataFrame(social_data)

            # Soft constraints
            constraints = pd.read_sql("SELECT * FROM soft_constraints ORDER BY timestamp DESC LIMIT 1", self.engine)
            if not constraints.empty:
                self.data_cache['current_constraints'] = constraints.iloc[0].to_dict()

            # Teacher comments
            try:
                comments_df = pd.read_sql("SELECT student_id, comment_text FROM teacher_comments", self.engine)
                comments_df['recommendation'] = 'Balance wellbeing and academic factors'
                self.data_cache['teacher_comments'] = comments_df
            except Exception:
                print("[INFO] No teacher_comments table found")

            # Save to cache for future use
            try:
                os.makedirs(cache_dir, exist_ok=True)
                for key, value in self.data_cache.items():
                    if isinstance(value, pd.DataFrame):
                        value.to_parquet(f'{cache_dir}/{key}.parquet', index=False)
                        print(f"[INFO] Cached {key} to parquet file")
                    elif isinstance(value, list) and key != 'class_ids':  # Simple lists can be saved directly
                        pd.DataFrame(value).to_parquet(f'{cache_dir}/{key}.parquet', index=False)
                        print(f"[INFO] Cached {key} to parquet file")
                    elif key == 'class_ids':
                        pd.DataFrame({'class_id': value}).to_parquet(f'{cache_dir}/{key}.parquet', index=False)
                        print(f"[INFO] Cached {key} to parquet file")
                    elif isinstance(value, dict) and key == 'current_constraints':
                        pd.DataFrame([value]).to_parquet(f'{cache_dir}/{key}.parquet', index=False)
                        print(f"[INFO] Cached {key} to parquet file")
                print("[INFO] Successfully cached all datasets to parquet files")
            except Exception as e:
                print(f"[ERROR] Failed to cache data: {e}")

            return self.data_cache

        except Exception as e:
            print(f"[ERROR] Data loading failed: {e}")
            return {}
