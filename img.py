import pymongo
import gridfs
from bson.objectid import ObjectId
import json

# Connect to the MongoDB instance
client = pymongo.MongoClient("mongodb://localhost:27017/")

# Access the database
db = client["mydatabase"]

# Initialize GridFS
fs = gridfs.GridFS(db)

# Access the collections
image_details_collection = db["image_details"]

def retrieve_image_and_details_by_medicine_name(medicine_name, output_folder_path):
    try:
        # Retrieve the image details by medicine name
        image_details = image_details_collection.find_one({"Medicine Name": medicine_name})
        
        if image_details:
            image_id = image_details["_id"]

            # Retrieve the image from GridFS
            image = fs.get(ObjectId(image_id))
            image_save_path = f"{output_folder_path}/image_{image_id}.jpg"
            with open(image_save_path, 'wb') as file:
                file.write(image.read())
            print(f"Image saved to {image_save_path}")

            # Print image details
            print("Image Details:")
            print(json.dumps(image_details, indent=4, default=str))
        else:
            print(f"No details found for medicine: {medicine_name}")

    except gridfs.errors.NoFile:
        print(f"Image not found in GridFS for ObjectId: {image_id}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
medicine_name = input("Enter the Medicine Name: ")
output_folder_path = r"C:\Users\cuted\OneDrive\Desktop\image"  # Replace with your desired output folder path

retrieve_image_and_details_by_medicine_name(medicine_name, output_folder_path)

