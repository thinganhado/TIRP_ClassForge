import os
import json
import pandas as pd
import requests
from datetime import datetime
import logging

# Import database query modules
try:
    from app.database import class_queries, student_queries, softcons_queries, friends_queries
    HAS_DB_MODULES = True
except ImportError:
    HAS_DB_MODULES = False
    print("[INFO] Database query modules not found, using API only")

# Try to import config
try:
    from app.config import Config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("[INFO] App config not found, using default API settings")

class ApiClient:
    """Client for making API calls to the TIRP backend services"""
    
    def __init__(self):
        # Set up API base URL from config or use default
        if HAS_CONFIG and hasattr(Config, 'API_BASE_URL'):
            self.base_url = Config.API_BASE_URL
        else:
            # Default to localhost for development
            self.base_url = "http://127.0.0.1:5000/api"
        
        # Set up API endpoints
        self.endpoints = {
            'students': '/students',
            'student_details': '/students/{student_id}',
            'classes': '/classes',
            'class_gpa': '/classes/{class_id}/avg_gpa',
            'wellbeing': '/wellbeing',
            'bullying': '/bullying',
            'gpa': '/gpa',
            'social': '/social',
            'constraints': '/constraints',
            'teacher_comments': '/teacher_comments'
        }
        
        # Set up authentication if provided in config
        self.auth_token = None
        if HAS_CONFIG and hasattr(Config, 'API_KEY'):
            self.auth_token = Config.API_KEY
        
        # Setup requests session
        self.session = requests.Session()
        if self.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
        
        # Cache for storing API responses
        self.cache = {}
        self.cache_dir = 'app/cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Setup database connection flag
        self.has_db_connection = False
        if HAS_DB_MODULES and HAS_CONFIG:
            # If we have database modules and config, we can use them directly
            # when API is not available
            try:
                from app import db
                self.db = db
                self.has_db_connection = True
                print("[INFO] Database connection available as fallback")
            except ImportError:
                print("[INFO] Flask db not available, using API only")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ApiClient')
    
    def _get_url(self, endpoint_key, **kwargs):
        """Construct the full URL for an API endpoint with path parameters"""
        if endpoint_key not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint_key}")
        
        endpoint = self.endpoints[endpoint_key]
        # Replace any path parameters in the endpoint
        for key, value in kwargs.items():
            endpoint = endpoint.replace(f"{{{key}}}", str(value))
        
        return f"{self.base_url}{endpoint}"
    
    def _make_request(self, method, endpoint_key, params=None, data=None, **kwargs):
        """Make an HTTP request to the API"""
        url = self._get_url(endpoint_key, **kwargs)
        
        try:
            if method.lower() == 'get':
                response = self.session.get(url, params=params)
            elif method.lower() == 'post':
                response = self.session.post(url, json=data)
            elif method.lower() == 'put':
                response = self.session.put(url, json=data)
            elif method.lower() == 'delete':
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            # Try direct database access if available
            if self.has_db_connection and HAS_DB_MODULES:
                self.logger.info(f"Trying direct database access for {endpoint_key}")
                return self._db_fallback(endpoint_key, **kwargs)
            return None
    
    def _db_fallback(self, endpoint_key, **kwargs):
        """Use direct database access as fallback when API is unavailable"""
        if not self.has_db_connection or not HAS_DB_MODULES:
            return None
            
        try:
            # Map endpoint to appropriate database query
            if endpoint_key == 'students':
                # Check what functions actually exist in the module
                available_funcs = [func for func in dir(student_queries) if not func.startswith('_') and callable(getattr(student_queries, func))]
                self.logger.info(f"Available student query functions: {available_funcs}")
                
                # Try all possible function names in priority order
                possible_functions = ['get_students', 'query_all_students', 'fetch_all_students', 'fetch_students']
                for func_name in possible_functions:
                    if func_name in available_funcs:
                        func = getattr(student_queries, func_name)
                        self.logger.info(f"Using {func_name} function")
                        result = func()
                        return result
                
                self.logger.error(f"No suitable function found in student_queries module")
                return None
                    
            elif endpoint_key == 'student_details':
                student_id = kwargs.get('student_id')
                if student_id:
                    available_funcs = [func for func in dir(student_queries) if not func.startswith('_') and callable(getattr(student_queries, func))]
                    possible_functions = ['get_student_details', 'query_student', 'fetch_student_details']
                    for func_name in possible_functions:
                        if func_name in available_funcs:
                            func = getattr(student_queries, func_name)
                            self.logger.info(f"Using {func_name} function")
                            result = func(student_id)
                            return result
                    
            elif endpoint_key == 'classes':
                # Check what functions actually exist in the module
                available_funcs = [func for func in dir(class_queries) if not func.startswith('_') and callable(getattr(class_queries, func))]
                self.logger.info(f"Available class query functions: {available_funcs}")
                
                possible_functions = ['get_classes', 'query_all_classes', 'fetch_all_classes', 'fetch_classes_summary', 'fetch_unique_classes']
                for func_name in possible_functions:
                    if func_name in available_funcs:
                        func = getattr(class_queries, func_name)
                        self.logger.info(f"Using {func_name} function")
                        result = func()
                        return result
                
                self.logger.error(f"No suitable function found in class_queries module")
                return None
                    
            elif endpoint_key == 'class_gpa':
                class_id = kwargs.get('class_id')
                if class_id:
                    available_funcs = [func for func in dir(class_queries) if not func.startswith('_') and callable(getattr(class_queries, func))]
                    possible_functions = ['get_class_avg_gpa', 'query_class_gpa', 'get_class_metrics']
                    for func_name in possible_functions:
                        if func_name in available_funcs:
                            func = getattr(class_queries, func_name)
                            self.logger.info(f"Using {func_name} function")
                            result = func(class_id)
                            return result
                        
            elif endpoint_key == 'constraints':
                # Check what functions actually exist in the module
                available_funcs = [func for func in dir(softcons_queries) if not func.startswith('_') and callable(getattr(softcons_queries, func))]
                self.logger.info(f"Available constraints query functions: {available_funcs}")
                
                possible_functions = ['get_constraints', 'query_constraints', 'get_current_constraints']
                for func_name in possible_functions:
                    if func_name in available_funcs:
                        func = getattr(softcons_queries, func_name)
                        self.logger.info(f"Using {func_name} function")
                        result = func()
                        return result
                
                self.logger.error(f"No suitable function found in softcons_queries module")
                return None
                    
            elif endpoint_key == 'wellbeing':
                # Since wellbeing data isn't available via database queries,
                # fallback to CSV directly
                csv_file = "app/ml_models/Clustering/output/cluster_assignments.csv"
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        self.logger.info(f"Loaded wellbeing from CSV file: {csv_file}")
                        return df.to_dict(orient='records')
                    except Exception as e:
                        self.logger.error(f"Failed to load wellbeing from CSV: {e}")
                        
            elif endpoint_key == 'bullying':
                # Similar fallback for bullying data
                csv_file = "app/ml_models/ha_outputs/community_bully_assignments.csv"
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        self.logger.info(f"Loaded bullying from CSV file: {csv_file}")
                        return df.to_dict(orient='records')
                    except Exception as e:
                        self.logger.error(f"Failed to load bullying from CSV: {e}")
                        
            elif endpoint_key == 'gpa':
                # Similar fallback for GPA data
                csv_file = "app/ml_models/ha_outputs/gpa_predictions_with_bins.csv"
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        self.logger.info(f"Loaded GPA from CSV file: {csv_file}")
                        return df.to_dict(orient='records')
                    except Exception as e:
                        self.logger.error(f"Failed to load GPA from CSV: {e}")
                        
            elif endpoint_key == 'social':
                # Similar fallback for social data
                csv_file = "app/ml_models/Clustering/output/cluster_assignments.csv"
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        self.logger.info(f"Loaded social from CSV file: {csv_file}")
                        
                        # Make sure we have all required columns
                        if 'Participant-ID' in df.columns:
                            # Create social_wellbeing column if it doesn't exist
                            if 'social_wellbeing' not in df.columns:
                                # Map wellbeing labels to numeric values
                                wellbeing_map = {
                                    'High Wellbeing': 80,
                                    'Moderate Wellbeing': 50,
                                    'Low Wellbeing': 30
                                }
                                if 'wellbeing_label' in df.columns:
                                    df['social_wellbeing'] = df['wellbeing_label'].map(wellbeing_map)
                                else:
                                    # Default value
                                    df['social_wellbeing'] = 50
                                
                            # Make sure there's a cluster column
                            if 'cluster' not in df.columns and 'cluster_label' in df.columns:
                                df['cluster'] = df['cluster_label']
                                
                        return df.to_dict(orient='records')
                    except Exception as e:
                        self.logger.error(f"Failed to load social data from CSV: {e}")
                        
            elif endpoint_key == 'teacher_comments':
                # Create synthetic teacher comments
                return self._generate_synthetic_comments()
                
            # Add more mappings as needed
            
            self.logger.warning(f"No database fallback defined for {endpoint_key}")
            return None
        except Exception as e:
            self.logger.error(f"Database fallback failed: {e}")
            return None
            
    def _generate_synthetic_comments(self):
        """Generate synthetic teacher comments"""
        try:
            comments = []
            for i in range(10):
                comments.append({
                    "student_id": f"3250{i}",
                    "comment_text": f"This is a synthetic comment for student 3250{i}",
                    "recommendation": "Balance wellbeing and academic factors"
                })
            self.logger.info(f"Generated {len(comments)} synthetic teacher comments")
            return comments
        except Exception as e:
            self.logger.error(f"Failed to generate synthetic comments: {e}")
            return []
            
    def get_students(self):
        """Get all students data"""
        cache_key = 'students'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to load from cache file first
        cache_file = f'{self.cache_dir}/{cache_key}.parquet'
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from cache file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from cache: {e}")
        
        data = self._make_request('get', 'students')
        if data:
            # Convert to dataframe and cache it
            df = pd.DataFrame(data)
            self.cache[cache_key] = df
            
            # Save to cache file
            try:
                df.to_parquet(cache_file, index=False)
                self.logger.info(f"Saved {cache_key} to cache file")
            except Exception as e:
                self.logger.error(f"Failed to save {cache_key} to cache: {e}")
                
            return df
        
        # If direct API and database fallbacks failed, try to create synthetic data
        if cache_key not in self.cache:
            # Try to extract student IDs from other datasets to create synthetic data
            student_ids = set()
            
            # Try to get wellbeing data
            wellbeing_data = self.get_wellbeing_data()
            if wellbeing_data is not None and 'Participant-ID' in wellbeing_data.columns:
                student_ids.update(wellbeing_data['Participant-ID'].unique())
            
            # Create synthetic students
            if student_ids:
                students_data = []
                for student_id in student_ids:
                    students_data.append({
                        'id': student_id,
                        'name': f"Student {student_id}",
                        'email': f"student{student_id}@example.com",
                        'class_id': "C1"  # Default class
                    })
                
                df = pd.DataFrame(students_data)
                self.cache[cache_key] = df
                self.logger.info(f"Created {len(students_data)} synthetic student records")
                return df
        
        return None
    
    def get_student_details(self, student_id):
        """Get details for a specific student"""
        cache_key = f'student_{student_id}'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        data = self._make_request('get', 'student_details', student_id=student_id)
        if data:
            self.cache[cache_key] = data
            return data
        return None
    
    def get_classes(self):
        """Get all classes data"""
        cache_key = 'classes'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to load from cache file first
        cache_file = f'{self.cache_dir}/{cache_key}.parquet'
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from cache file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from cache: {e}")
        
        data = self._make_request('get', 'classes')
        if data:
            # Store class IDs
            class_ids = [cls['id'] for cls in data]
            self.cache['class_ids'] = class_ids
            
            # Convert to dataframe and cache it
            df = pd.DataFrame(data)
            self.cache[cache_key] = df
            
            # Save to cache file
            try:
                df.to_parquet(cache_file, index=False)
                self.logger.info(f"Saved {cache_key} to cache file")
            except Exception as e:
                self.logger.error(f"Failed to save {cache_key} to cache: {e}")
                
            return df
        
        # If direct API and database fallbacks failed, try to create synthetic data
        if cache_key not in self.cache:
            # Create basic synthetic class data
            class_data = [
                {'id': 'C1', 'name': 'Class 1', 'size': 25, 'teacher': 'Teacher 1'},
                {'id': 'C2', 'name': 'Class 2', 'size': 30, 'teacher': 'Teacher 2'},
                {'id': 'C3', 'name': 'Class 3', 'size': 28, 'teacher': 'Teacher 3'}
            ]
            df = pd.DataFrame(class_data)
            self.cache[cache_key] = df
            self.logger.info("Created synthetic class data")
            return df
        
        return None
    
    def get_class_gpa(self, class_id):
        """Get average GPA for a specific class"""
        cache_key = f'class_gpa_{class_id}'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        data = self._make_request('get', 'class_gpa', class_id=class_id)
        if data:
            self.cache[cache_key] = data
            return data
        return None
    
    def get_wellbeing_data(self):
        """Get wellbeing data for all students"""
        cache_key = 'wellbeing'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to load from cache file first
        cache_file = f'{self.cache_dir}/{cache_key}.parquet'
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from cache file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from cache: {e}")
        
        # Fallback to CSV if available
        csv_file = "app/ml_models/Clustering/output/cluster_assignments.csv"
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from CSV file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from CSV: {e}")
        
        # Fetch from API if not in cache
        data = self._make_request('get', 'wellbeing')
        if data:
            # Convert to dataframe
            df = pd.DataFrame(data)
            self.cache[cache_key] = df
            
            # Save to cache file
            try:
                df.to_parquet(cache_file, index=False)
                self.logger.info(f"Saved {cache_key} to cache file")
            except Exception as e:
                self.logger.error(f"Failed to save {cache_key} to cache: {e}")
            
            return df
        return None
    
    def get_bullying_data(self):
        """Get bullying data for all students"""
        cache_key = 'bullying'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to load from cache file first
        cache_file = f'{self.cache_dir}/{cache_key}.parquet'
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from cache file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from cache: {e}")
        
        # Fallback to CSV if available
        csv_file = "app/ml_models/ha_outputs/community_bully_assignments.csv"
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from CSV file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from CSV: {e}")
        
        # Fetch from API if not in cache
        data = self._make_request('get', 'bullying')
        if data:
            # Convert to dataframe
            df = pd.DataFrame(data)
            self.cache[cache_key] = df
            
            # Save to cache file
            try:
                df.to_parquet(cache_file, index=False)
                self.logger.info(f"Saved {cache_key} to cache file")
            except Exception as e:
                self.logger.error(f"Failed to save {cache_key} to cache: {e}")
            
            return df
        return None
    
    def get_gpa_data(self):
        """Get GPA data for all students"""
        cache_key = 'gpa'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to load from cache file first
        cache_file = f'{self.cache_dir}/{cache_key}.parquet'
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from cache file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from cache: {e}")
        
        # Fallback to CSV if available
        csv_file = "app/ml_models/ha_outputs/gpa_predictions_with_bins.csv"
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from CSV file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from CSV: {e}")
        
        # Fetch from API if not in cache
        data = self._make_request('get', 'gpa')
        if data:
            # Convert to dataframe
            df = pd.DataFrame(data)
            self.cache[cache_key] = df
            
            # Save to cache file
            try:
                df.to_parquet(cache_file, index=False)
                self.logger.info(f"Saved {cache_key} to cache file")
            except Exception as e:
                self.logger.error(f"Failed to save {cache_key} to cache: {e}")
            
            return df
        return None
    
    def get_social_data(self):
        """Get social wellbeing data for all students"""
        cache_key = 'social'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to load from cache file first
        cache_file = f'{self.cache_dir}/{cache_key}.parquet'
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from cache file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from cache: {e}")
        
        # Fallback to CSV if available
        csv_file = "app/ml_models/Clustering/output/cluster_assignments.csv"
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                # Rename columns to match what's expected by the system
                if 'Participant-ID' in df.columns and 'social_wellbeing' not in df.columns:
                    # Create a social_wellbeing column based on wellbeing_label
                    # Map wellbeing labels to numeric values
                    wellbeing_map = {
                        'High Wellbeing': 80,
                        'Moderate Wellbeing': 50,
                        'Low Wellbeing': 30
                    }
                    if 'wellbeing_label' in df.columns:
                        df['social_wellbeing'] = df['wellbeing_label'].map(wellbeing_map)
                    else:
                        # Use a default value
                        df['social_wellbeing'] = 50
                        
                    # Make sure there's a cluster column
                    if 'cluster' not in df.columns and 'cluster_label' in df.columns:
                        df['cluster'] = df['cluster_label']
                        
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from CSV file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from CSV: {e}")
        
        # Fetch from API if not in cache
        data = self._make_request('get', 'social')
        if data:
            # Convert to dataframe
            df = pd.DataFrame(data)
            self.cache[cache_key] = df
            
            # Save to cache file
            try:
                df.to_parquet(cache_file, index=False)
                self.logger.info(f"Saved {cache_key} to cache file")
            except Exception as e:
                self.logger.error(f"Failed to save {cache_key} to cache: {e}")
            
            return df
        return None
    
    def get_teacher_comments(self):
        """Get teacher comments data"""
        cache_key = 'teacher_comments'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to load from cache file first
        cache_file = f'{self.cache_dir}/{cache_key}.parquet'
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                self.cache[cache_key] = df
                self.logger.info(f"Loaded {cache_key} from cache file")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load {cache_key} from cache: {e}")
        
        # Fetch from API if not in cache
        data = self._make_request('get', 'teacher_comments')
        if data:
            # Convert to dataframe
            df = pd.DataFrame(data)
            self.cache[cache_key] = df
            
            # Save to cache file
            try:
                df.to_parquet(cache_file, index=False)
                self.logger.info(f"Saved {cache_key} to cache file")
            except Exception as e:
                self.logger.error(f"Failed to save {cache_key} to cache: {e}")
            
            return df
        
        # If we still don't have teacher comments, generate synthetic ones
        if cache_key not in self.cache:
            synthetic_data = self._generate_synthetic_comments()
            if synthetic_data:
                df = pd.DataFrame(synthetic_data)
                self.cache[cache_key] = df
                return df
        
        return None
    
    def get_constraints(self):
        """Get current constraints configuration"""
        cache_key = 'constraints'
        
        # Fetch fresh data from API (don't use cache for constraints)
        data = self._make_request('get', 'constraints')
        if data:
            self.cache[cache_key] = data
            return data
        
        # If API fails and we have database access, try database directly
        if self.has_db_connection and HAS_DB_MODULES:
            try:
                available_funcs = [func for func in dir(softcons_queries) if not func.startswith('_') and callable(getattr(softcons_queries, func))]
                possible_functions = ['get_constraints', 'query_constraints', 'get_current_constraints']
                for func_name in possible_functions:
                    if func_name in available_funcs:
                        func = getattr(softcons_queries, func_name)
                        self.logger.info(f"Using {func_name} function for constraints")
                        constraints = func()
                        if constraints:
                            self.cache[cache_key] = constraints
                            return constraints
            except Exception as e:
                self.logger.error(f"Database constraint lookup failed: {e}")
        
        # If API and database fail, try to read from file
        config_file = "soft_constraints_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                self.cache[cache_key] = config
                return config
            except Exception as e:
                self.logger.error(f"Failed to load constraints from file: {e}")
        
        # Return default constraints if all else fails
        default_config = {
            "class_size": 30,
            "max_classes": 6,
            "gpa_penalty_weight": 30,
            "wellbeing_penalty_weight": 50,
            "bully_penalty_weight": 60,
            "influence_std_weight": 60, 
            "isolated_std_weight": 60,
            "min_friends_required": 1,
            "friend_inclusion_weight": 50,
            "friendship_balance_weight": 40,
            "prioritize_academic": 5,
            "prioritize_wellbeing": 4,
            "prioritize_bullying": 3,
            "prioritize_social_influence": 2,
            "prioritize_friendship": 1
        }
        self.cache[cache_key] = default_config
        return default_config
    
    def save_constraints(self, config):
        """Save constraints configuration via API"""
        result = self._make_request('post', 'constraints', data=config)
        if result and result.get('success'):
            # Also save locally
            config_file = "soft_constraints_config.json"
            try:
                with open(config_file, "w") as f:
                    json.dump(config, f, indent=2)
                return config
            except Exception as e:
                self.logger.error(f"Failed to save constraints to file: {e}")
                return result
                
        # If API fails and we have database access, try database directly
        if self.has_db_connection and HAS_DB_MODULES:
            try:
                available_funcs = [func for func in dir(softcons_queries) if not func.startswith('_') and callable(getattr(softcons_queries, func))]
                if 'save_constraints' in available_funcs:
                    saved = softcons_queries.save_constraints(config)
                    if saved:
                        self.logger.info("Saved constraints to database")
                        # Also save locally
                        config_file = "soft_constraints_config.json"
                        with open(config_file, "w") as f:
                            json.dump(config, f, indent=2)
                        return config
            except Exception as e:
                self.logger.error(f"Database constraint save failed: {e}")
        
        # If all else fails, just save locally
        config_file = "soft_constraints_config.json"
        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            self.logger.info("Saved constraints to local file only")
            return config
        except Exception as e:
            self.logger.error(f"Failed to save constraints to file: {e}")
            return None
    
    def load_data_comprehensive(self):
        """Load all data from various sources and return as a dictionary"""
        data_cache = {}
        
        # Load all data types
        students_df = self.get_students()
        if students_df is not None:
            data_cache['students'] = students_df
        
        wellbeing_df = self.get_wellbeing_data()
        if wellbeing_df is not None:
            data_cache['wellbeing'] = wellbeing_df
        
        bullying_df = self.get_bullying_data()
        if bullying_df is not None:
            data_cache['bullying'] = bullying_df
        
        gpa_df = self.get_gpa_data()
        if gpa_df is not None:
            data_cache['gpa'] = gpa_df
        
        social_df = self.get_social_data()
        if social_df is not None:
            data_cache['social'] = social_df
        
        comments_df = self.get_teacher_comments()
        if comments_df is not None:
            data_cache['teacher_comments'] = comments_df
        
        # Get classes
        classes_df = self.get_classes()
        if classes_df is not None:
            data_cache['classes'] = classes_df
            if 'class_ids' in self.cache:
                data_cache['class_ids'] = self.cache['class_ids']
        
        # Get constraints
        constraints = self.get_constraints()
        if constraints:
            data_cache['current_constraints'] = constraints
        
        # Log summary of loaded data
        self.logger.info(f"Loaded {len(data_cache)} datasets via API client")
        
        return data_cache

# Create an instance for import in other modules
api_client = ApiClient()
