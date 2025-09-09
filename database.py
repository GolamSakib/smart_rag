
import mysql.connector
from mysql.connector import errorcode

DB_NAME = 'smart_rag'

TABLES = {}
TABLES['images'] = (
    "CREATE TABLE `images` ("
    "  `id` int(11) NOT NULL AUTO_INCREMENT,"
    "  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,"
    "  `image_path` varchar(255) NOT NULL,"
    "  PRIMARY KEY (`id`)"
    ") ENGINE=InnoDB")

TABLES['products'] = (
    "CREATE TABLE `products` ("
    "  `id` int(11) NOT NULL AUTO_INCREMENT,"
    "  `name` varchar(255) NOT NULL,"
    "  `description` text,"
    "  `price` decimal(10, 2) NOT NULL,"
    "  `code` varchar(50) NOT NULL,"
    "  `marginal_price` decimal(10, 2) NOT NULL,"
    "  `link` text,"
    "  PRIMARY KEY (`id`)"
    ") ENGINE=InnoDB")

TABLES['product_images'] = (
    "CREATE TABLE `product_images` ("
    "  `id` int(11) NOT NULL AUTO_INCREMENT,"
    "  `product_id` int(11) NOT NULL,"
    "  `image_id` int(11) NOT NULL,"
    "  PRIMARY KEY (`id`),"
    "  FOREIGN KEY (`product_id`) REFERENCES `products` (`id`) ON DELETE CASCADE,"
    "  FOREIGN KEY (`image_id`) REFERENCES `images` (`id`) ON DELETE CASCADE"
    ") ENGINE=InnoDB")

def create_database(cursor):
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)

def create_tables(cursor):
    cursor.execute("USE {}".format(DB_NAME))
    for table_name in TABLES:
        table_description = TABLES[table_name]
        try:
            print("Creating table {}: ".format(table_name), end='')
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("OK")

if __name__ == "__main__":
    try:
        cnx = mysql.connector.connect(
            host="localhost",
            user="root",
            password=""
        )
        cursor = cnx.cursor()
        try:
            cursor.execute("USE {}".format(DB_NAME))
        except mysql.connector.Error as err:
            print("Database {} does not exists.".format(DB_NAME))
            if err.errno == errorcode.ER_BAD_DB_ERROR:
                create_database(cursor)
                print("Database {} created successfully.".format(DB_NAME))
                cnx.database = DB_NAME
            else:
                print(err)
                exit(1)
        create_tables(cursor)
        cursor.close()
        cnx.close()
    except mysql.connector.Error as err:
        print(err)
