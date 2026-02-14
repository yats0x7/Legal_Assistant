import zipfile
import os

# Ensure this matches your actual folder name
data_dir = "data/" 

print(f"📂 Checking {data_dir} for zip files...")

found_zip = False
for file in os.listdir(data_dir):
    if file.endswith(".zip"):
        found_zip = True
        file_path = os.path.join(data_dir, file)
        print(f"   found: {file}")
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Extract into a subfolder with the same name as the zip file
            folder_name = file.replace(".zip", "")
            extract_path = os.path.join(data_dir, folder_name)
            
            # Create the folder if it doesn't exist
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)
                
            zip_ref.extractall(extract_path)
            print(f"✅ Extracted to: {extract_path}")

if not found_zip:
    print("❌ No zip files found. Make sure they are inside the 'data' folder.")