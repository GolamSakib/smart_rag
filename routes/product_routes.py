from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
import subprocess
import decimal

from services.database_service import db_service
from config.settings import settings

router = APIRouter()


class Product(BaseModel):
    name: str
    description: str
    price: float
    code: str
    marginal_price: float
    image_ids: List[int]


@router.post("/api/products")
def create_product(product: Product):
    try:
        with db_service.get_cursor() as (cursor, conn):
            add_product = ("INSERT INTO products "
                          "(name, description, price, code, marginal_price) "
                          "VALUES (%s, %s, %s, %s, %s)")
            data_product = (product.name, product.description, product.price, product.code, product.marginal_price)
            cursor.execute(add_product, data_product)
            conn.commit()
            product_id = cursor.lastrowid

            if product.image_ids:
                add_product_image = ("INSERT INTO product_images (product_id, image_id) VALUES (%s, %s)")
                for image_id in product.image_ids:
                    cursor.execute(add_product_image, (product_id, image_id))
                conn.commit()

            return {"id": product_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/api/products")
def get_products(
    name: Optional[str] = None, 
    code: Optional[str] = None, 
    min_price: Optional[str] = None, 
    max_price: Optional[str] = None
):
    try:
        with db_service.get_cursor(dictionary=True) as (cursor, conn):
            query = ("SELECT p.*, GROUP_CONCAT(i.id) as image_ids, GROUP_CONCAT(i.image_path) as images "
                    "FROM products p "
                    "LEFT JOIN product_images pi ON p.id = pi.product_id "
                    "LEFT JOIN images i ON pi.image_id = i.id "
                    "WHERE 1=1")
            params = []

            if name:
                query += " AND p.name LIKE %s"
                params.append(f"%{name}%")
            if code:
                query += " AND p.code = %s"
                params.append(code)
            
            if min_price:
                try:
                    min_price_float = float(min_price)
                    query += " AND p.price >= %s"
                    params.append(min_price_float)
                except (ValueError, TypeError):
                    pass
                    
            if max_price:
                try:
                    max_price_float = float(max_price)
                    query += " AND p.price <= %s"
                    params.append(max_price_float)
                except (ValueError, TypeError):
                    pass

            query += " GROUP BY p.id ORDER BY p.id DESC"
            
            cursor.execute(query, tuple(params))
            products = cursor.fetchall()
            
            for product in products:
                if product['images']:
                    product['images'] = product['images'].split(',')
                    product['image_ids'] = [int(id) for id in product['image_ids'].split(',')]
                else:
                    product['images'] = []
                    product['image_ids'] = []
                    
            return products
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.put("/api/products/{product_id}")
def update_product(product_id: int, product: Product):
    try:
        with db_service.get_cursor() as (cursor, conn):
            update_prod = ("UPDATE products SET name=%s, description=%s, price=%s, code=%s, marginal_price=%s WHERE id=%s")
            data_prod = (product.name, product.description, product.price, product.code, product.marginal_price, product_id)
            cursor.execute(update_prod, data_prod)

            cursor.execute("DELETE FROM product_images WHERE product_id = %s", (product_id,))
            if product.image_ids:
                add_product_image = ("INSERT INTO product_images (product_id, image_id) VALUES (%s, %s)")
                for image_id in product.image_ids:
                    cursor.execute(add_product_image, (product_id, image_id))
            
            conn.commit()
            return {"message": "Product updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.delete("/api/products/{product_id}")
def delete_product(product_id: int):
    try:
        with db_service.get_cursor() as (cursor, conn):
            cursor.execute("DELETE FROM products WHERE id = %s", (product_id,))
            conn.commit()
            return {"message": "Product deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.post("/api/generate-json")
def generate_json():
    try:
        with db_service.get_cursor(dictionary=True) as (cursor, conn):
            cursor.execute("SELECT p.*, i.image_path FROM products p JOIN product_images pi ON p.id = pi.product_id JOIN images i ON pi.image_id = i.id")
            products = cursor.fetchall()

        for product in products:
            for key, value in product.items():
                if isinstance(value, decimal.Decimal):
                    product[key] = float(value)

        import os
        import json
        if not os.path.exists("data"):
            os.makedirs("data")

        with open(settings.PRODUCTS_JSON_PATH, "w") as f:
            json.dump(products, f, indent=4)
            
        return {"message": "products.json generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/api/train")
def train_model():
    try:
        result = subprocess.run(["python", "training.py"], capture_output=True, text=True, check=True)
        return {"message": "Training completed successfully", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}") 