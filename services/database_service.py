import mysql.connector
from mysql.connector import pooling
from mysql.connector import errorcode
from config.settings import settings
from contextlib import contextmanager
from typing import Optional, Dict, Any


class DatabaseService:
    """Database service with connection pooling"""
    
    def __init__(self):
        self.pool = None
        self._create_connection_pool()
    
    def _create_connection_pool(self):
        """Create connection pool for better performance"""
        try:
            self.pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="smart_rag_pool",
                pool_size=5,
                pool_reset_session=True,
                **settings.DB_CONFIG
            )
            print("Database connection pool created successfully")
        except mysql.connector.Error as err:
            print(f"Error creating connection pool: {err}")
            self.pool = None
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        connection = None
        try:
            if self.pool:
                connection = self.pool.get_connection()
            else:
                connection = mysql.connector.connect(**settings.DB_CONFIG)
            yield connection
        except mysql.connector.Error as err:
            if connection:
                connection.rollback()
            print(f"Database error: {err}")
            raise
        finally:
            if connection:
                connection.close()
    
    @contextmanager
    def get_cursor(self, dictionary=False):
        """Context manager for database cursors"""
        with self.get_connection() as connection:
            cursor = connection.cursor(dictionary=dictionary)
            try:
                yield cursor, connection
            finally:
                cursor.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                return True
        except Exception as e:
            print(f"Database connection test failed: {e}")
            return False


# Global database service instance
db_service = DatabaseService() 