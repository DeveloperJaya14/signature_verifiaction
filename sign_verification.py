import os
import cv2
import pandas as pd
from urllib.parse import urlparse
from skimage.metrics import structural_similarity as ssim
import re
from urllib.parse import urlparse

# File paths
excel1_path = './cropedsignatures.xlsx'
excel2_path = './extracteddetails.xlsx'
folder1_path = './signature_verifcation_images'  # Full cheque images
folder2_path = './croped_signature_img'          # Cropped signature images

# Load data
df1 = pd.read_excel(excel1_path).fillna('')
df2 = pd.read_excel(excel2_path).fillna('')

# Normalize account numbers (remove .0, spaces, etc.)
df1['account_no'] = df1['account_no'].apply(lambda x: str(x).strip().split('.')[0])
df2['account_no'] = df2['account_no'].apply(lambda x: str(x).strip().split('.')[0])
df1['cust_id'] = df1['cust_id'].apply(lambda x: str(x).strip().split('.')[0])

results = []

def extract_image_name_from_url(url):
    parsed_url = urlparse(url)
    last_segment = os.path.basename(parsed_url.path)
    match = re.search(r'[^/\\]+?\.(jpg|jpeg|png|bmp|tiff|gif)$', last_segment, re.IGNORECASE)
    return match.group(0) if match else None

def is_signature_present(full_img_path, cropped_img_path):
    try:
        full_img = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
        cropped_img = cv2.imread(cropped_img_path, cv2.IMREAD_GRAYSCALE)

        if full_img is None or cropped_img is None:
            return False, "Image not found"

        result = cv2.matchTemplate(full_img, cropped_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return (max_val > 0.1), f"Match score: {max_val:.2f}"

    except Exception as e:
        return False, str(e)

# Iterate through each row in df2 (extracteddetails)
for _, row in df1.iterrows():
    account_no = str(row['account_no']).strip().split('.')[0]
    cust_id = str(row['cust_id']).strip().split('.')[0]

    cropped_img_found = False
    account_no_found = False
    full_img_found = False
    is_match = None
    full_img_name = None
    info = ""

    # Check for cropped image using cust_id
    cropped_img_path = os.path.join(folder2_path, f"{cust_id}.png")
    if os.path.exists(cropped_img_path):
        cropped_img_found = True
    else:
        info = "Cropped image not found"

    # Check if account number is in df1 (cropedsignatures)
    matched_row = df2[df2['account_no'] == account_no]
    if not matched_row.empty:
        account_no_found = True
        img_url = matched_row.iloc[0]['image_url']
        if isinstance(img_url, str) and img_url.strip():
            full_img_name = extract_image_name_from_url(img_url)
            if full_img_name:
                full_img_path = os.path.join(folder1_path, full_img_name)
                if os.path.exists(full_img_path):
                    full_img_found = True
                    if cropped_img_found:
                        is_match, info = is_signature_present(full_img_path, cropped_img_path)
                    else:
                        info = "Cropped image not available for matching"
                else:
                    info = "Full image not found"
            else:
                info = "Invalid image name from URL"
        else:
            info = "Image URL missing"
    else:
        info = "Account number not found in extracted details"

    # Append to results
    results.append([
        cust_id,
        account_no,
        full_img_name,
        cropped_img_found,
        account_no_found,
        full_img_found,
        is_match,
        info
    ])

# Create output DataFrame
columns = [
    "cust_id", "account_no", "full_img_name",
    "cust_id_found", "account_no_found", "cheque_img_found",
    "is_match", "info"
]
result_df = pd.DataFrame(results, columns=columns)

# Save to Excel
result_df.to_excel("signature_match_results.xlsx", index=False)

# Show output
print(result_df)
