from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
from pathlib import Path

from services.database_service import db_service
from config.settings import settings

router = APIRouter()


@router.post("/api/images")
async def upload_images(files: List[UploadFile] = File(...)):
    upload_dir = settings.PRODUCT_IMAGES_PATH
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    saved_images = []
    
    try:
        with db_service.get_cursor(dictionary=True) as (cursor, conn):
            image_data_to_insert = []
            for file in files:
                filename = os.path.basename(file.filename)
                
                upload_path = Path(upload_dir) / filename
                
                counter = 1
                base, extension = upload_path.stem, upload_path.suffix
                while upload_path.exists():
                    new_filename = f"{base}_{counter}{extension}"
                    upload_path = Path(upload_dir) / new_filename
                    counter += 1
                
                with open(upload_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Construct URL path correctly
                url_path = f"/{upload_dir}/{upload_path.name}".replace('\\', '/')
                image_data_to_insert.append((url_path,))

            if image_data_to_insert:
                cursor.executemany("INSERT INTO images (image_path) VALUES (%s)", image_data_to_insert)
                conn.commit()
                
                # To get the saved images back with their IDs, we can fetch them
                # This is simpler than getting lastrowid in a loop with multiple inserts
                num_inserted = len(image_data_to_insert)
                cursor.execute("SELECT id, image_path FROM images ORDER BY id DESC LIMIT %s", (num_inserted,))
                # The order of fetched results should be the reverse of insertion, so we reverse it.
                saved_images = [dict(row) for row in reversed(cursor.fetchall())]

        return JSONResponse(content=saved_images)
    except Exception as e:
        print(f"Error during image upload: {e}")
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
            # First, get the image path from the database
            cursor.execute("SELECT image_path FROM images WHERE id = %s", (image_id,))
            image = cursor.fetchone()
            
            if image:
                url_path = image['image_path']
                # url_path is like '/product-image/image.jpg'
                # We need the filename to join it with the static directory path
                filename = Path(url_path).name
                file_system_path = Path(settings.PRODUCT_IMAGES_PATH) / filename

                if file_system_path.exists():
                    file_system_path.unlink()

            # Delete the record from the database
            cursor.execute("DELETE FROM images WHERE id = %s", (image_id,))
            conn.commit()
            
            # Check if row was actually deleted
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Image not found")

            return {"message": "Image deleted successfully"}
    except HTTPException:
        raise  # Re-raise HTTPException
    except OSError as e:
        # Handle file deletion errors without deleting DB record
        raise HTTPException(status_code=500, detail=f"File deletion error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")