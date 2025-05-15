# Cache Directory

This directory stores cached data in Parquet format to improve application performance.

## Purpose

- Stores database query results as Parquet files for faster loading
- Avoids redundant database queries
- Significantly improves application startup time
- Reduces load on the database server

## How it works

1. When data is first loaded from the database, it is cached here as Parquet files
2. On subsequent runs, data is loaded from these files instead of the database
3. To refresh the cache, delete the files or set `use_cache=False` when calling `load_data()`

## Files

This directory will contain various `.parquet` files:
- `wellbeing.parquet` - Cached wellbeing data
- `gpa.parquet` - Cached GPA data
- `social.parquet` - Cached social network data
- And more...

**Note:** These files are automatically generated and should not be committed to git. 