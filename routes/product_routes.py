from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel
import subprocess
import decimal
from collections import defaultdict

from services.database_service import db_service
from config.settings import settings

router = APIRouter()


class ProductImageInfo(BaseModel):
    image_id: int
    is_catalogue_image: bool = False
    is_variant_image: bool = False
    is_real_image: bool = False
    is_size_related_image: bool = False


class Product(BaseModel):
    name: str
    description: str
    price: float
    code: str
    marginal_price: float
    images: List[ProductImageInfo]
    link: Optional[str] = None


@router.post("/api/products")
def create_product(product: Product):
    try:
        with db_service.get_cursor() as (cursor, conn):
            add_product = ("INSERT INTO products "
                           "(name, description, price, code, marginal_price, link) "
                           "VALUES (%s, %s, %s, %s, %s, %s)")
            data_product = (product.name, product.description, product.price, product.code, product.marginal_price, product.link)
            cursor.execute(add_product, data_product)
            product_id = cursor.lastrowid

            if product.images:
                add_product_image = ("INSERT INTO product_images "
                                     "(product_id, image_id, is_catalogue_image, is_variant_image, is_real_image, is_size_related_image) "
                                     "VALUES (%s, %s, %s, %s, %s, %s)")
                image_data = []
                for image_info in product.images:
                    image_data.append((
                        product_id,
                        image_info.image_id,
                        image_info.is_catalogue_image,
                        image_info.is_variant_image,
                        image_info.is_real_image,
                        image_info.is_size_related_image
                    ))
                cursor.executemany(add_product_image, image_data)
            
            conn.commit()
            return {"id": product_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/api/products")
def get_products(
    name: Optional[str] = None, 
    code: Optional[str] = None, 
    min_price: Optional[str] = None, 
    max_price: Optional[str] = None,
    link: Optional[str] = None,
    page: int = 1,
    page_size: int = 10
):
    try:
        with db_service.get_cursor(dictionary=True) as (cursor, conn):
            filter_conditions = ""
            params = []

            if name:
                filter_conditions += " AND p.name LIKE %s"
                params.append(f"%{name}%")
            if code:
                filter_conditions += " AND p.code = %s"
                params.append(code)
            if link:
                filter_conditions += " AND p.link = %s"
                params.append(link)
            if min_price:
                filter_conditions += " AND p.price >= %s"
                params.append(float(min_price))
            if max_price:
                filter_conditions += " AND p.price <= %s"
                params.append(float(max_price))

            count_query = f"SELECT COUNT(id) as total FROM products p WHERE 1=1 {filter_conditions}"
            cursor.execute(count_query, tuple(params))
            total_count = cursor.fetchone()['total']

            offset = (page - 1) * page_size
            product_query = f"SELECT * FROM products p WHERE 1=1 {filter_conditions} ORDER BY id DESC LIMIT %s OFFSET %s"
            cursor.execute(product_query, tuple(params + [page_size, offset]))
            products = cursor.fetchall()
            
            product_ids = [p['id'] for p in products]
            images_dict = defaultdict(list)

            if product_ids:
                placeholders = ', '.join(['%s'] * len(product_ids))
                image_query = (f"SELECT pi.product_id, pi.image_id, i.image_path, "
                               f"pi.is_catalogue_image, pi.is_variant_image, pi.is_real_image, pi.is_size_related_image "
                               f"FROM product_images pi "
                               f"JOIN images i ON pi.image_id = i.id "
                               f"WHERE pi.product_id IN ({placeholders})")
                cursor.execute(image_query, tuple(product_ids))
                images = cursor.fetchall()
                for image in images:
                    images_dict[image['product_id']].append(image)

            for product in products:
                product['images'] = images_dict.get(product['id'], [])
            
            return {"products": products, "total": total_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.put("/api/products/{product_id}")
def update_product(product_id: int, product: Product):
    try:
        with db_service.get_cursor() as (cursor, conn):
            update_prod = ("UPDATE products SET name=%s, description=%s, price=%s, code=%s, marginal_price=%s, link=%s WHERE id=%s")
            data_prod = (product.name, product.description, product.price, product.code, product.marginal_price, product.link, product_id)
            cursor.execute(update_prod, data_prod)

            cursor.execute("DELETE FROM product_images WHERE product_id = %s", (product_id,))
            
            if product.images:
                add_product_image = ("INSERT INTO product_images "
                                     "(product_id, image_id, is_catalogue_image, is_variant_image, is_real_image, is_size_related_image) "
                                     "VALUES (%s, %s, %s, %s, %s, %s)")
                image_data = []
                for image_info in product.images:
                    image_data.append((
                        product_id,
                        image_info.image_id,
                        image_info.is_catalogue_image,
                        image_info.is_variant_image,
                        image_info.is_real_image,
                        image_info.is_size_related_image
                    ))
                cursor.executemany(add_product_image, image_data)
            
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
            cursor.execute("SELECT p.*, i.image_path, pi.is_catalogue_image, pi.is_variant_image, pi.is_real_image, pi.is_size_related_image FROM products p JOIN product_images pi ON p.id = pi.product_id JOIN images i ON pi.image_id = i.id")
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