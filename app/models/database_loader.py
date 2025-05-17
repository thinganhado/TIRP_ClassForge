import os
import pandas as pd
import numpy as np
from collections import defaultdict

# Import API client
from app.models.api_client import api_client

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
        # Always use API client instead of direct connections
        self.direct_connect = False
        self.db_available = True  # Assume API is available
        print("[INFO] Using API client for data access instead of direct database connection")

    def is_database_available(self):
        """Check if database access is available via API"""
        return True

    def _execute_query(self, query, params=None):
        """This method is deprecated - using API client instead"""
        print("[WARN] Using deprecated _execute_query method - API client should be used instead")
        return None

    def load_data(self, password=None, use_cache=True):
        """
        Load data using the API client
        
        Args:
            password (str, optional): No longer used, kept for compatibility
            use_cache (bool): Whether to try loading from cache first (default: True)
        """
        # Use the API client to load all data
        print("[INFO] Loading data via API client")
        
        try:
            # Use the API client's comprehensive data loading method
            api_data = api_client.load_data_comprehensive()
            
            if api_data and len(api_data) > 0:
                self.data_cache.update(api_data)
                print(f"[INFO] Successfully loaded {len(api_data)} datasets from API")
                return self.data_cache
            else:
                print("[WARN] API data loading failed, checking cache")
        except Exception as e:
            print(f"[ERROR] API data loading failed: {e}")
        
        # Check for cached data as fallback
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
        
        print("[ERROR] Could not load data via API or cache")
        return {}
        
    def save_config(self, config):
        """Save configuration using the API client"""
        try:
            # Use the API client to save the configuration
            result = api_client.save_constraints(config)
            if result:
                print("[INFO] Successfully saved config via API")
                return True
            else:
                print("[WARN] Failed to save config via API")
                return False
        except Exception as e:
            print(f"[ERROR] Error saving config via API: {e}")
            return False
