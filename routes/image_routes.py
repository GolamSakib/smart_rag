from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os

from services.database_service import db_service
from config.settings import settings

router = APIRouter()


@router.post("/api/images")
async def upload_images(files: List[UploadFile] = File(...)):
    if not os.path.exists(settings.PRODUCT_IMAGES_PATH):
        os.makedirs(settings.PRODUCT_IMAGES_PATH)

    saved_images = []
    
    try:
        with db_service.get_cursor() as (cursor, conn):
            for file in files:
                file_path = os.path.join(settings.PRODUCT_IMAGES_PATH, file.filename)
                with open(file_path, "wb") as buffer:
                    buffer.write(await file.read())

                add_image = ("INSERT INTO images (image_path) VALUES (%s)")
                data_image = (file_path,)
                cursor.execute(add_image, data_image)
                conn.commit()
                image_id = cursor.lastrowid
                saved_images.append({"id": image_id, "image_path": file_path})

        return JSONResponse(content=saved_images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/api/images")
def get_images():
    try:
        with db_service.get_cursor(dictionary=True) as (cursor, conn):
            cursor.execute("SELECT * FROM images ORDER BY created_at DESC")
            images = cursor.fetchall()
            return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.delete("/api/images/{image_id}")
def delete_image(image_id: int):
    try:
        with db_service.get_cursor(dictionary=True) as (cursor, conn):
            cursor.execute("SELECT image_path FROM images WHERE id = %s", (image_id,))
            image = cursor.fetchone()
            if image and os.path.exists(image['image_path']):
                os.remove(image['image_path'])

            cursor.execute("DELETE FROM images WHERE id = %s", (image_id,))
            conn.commit()
            return {"message": "Image deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") 