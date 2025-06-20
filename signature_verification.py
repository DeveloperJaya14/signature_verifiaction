import os
import cv2
import pandas as pd
from urllib.parse import urlparse
from skimage.metrics import structural_similarity as ssim
import re
from urllib.parse import urlparse
# Update paths
excel1_path = './cropedsignatures.xlsx'
excel2_path = './extracteddetails.xlsx'
folder1_path = './signature_verifcation_images'  # full cheques
folder2_path = './croped_signature_img'  # cropped signatures

# Read Excel files
df1 = pd.read_excel(excel1_path)
df2 = pd.read_excel(excel2_path)

results = []



def extract_image_name_from_url(url):
    # Extract last part of URL path
    parsed_url = urlparse(url)
    last_segment = os.path.basename(parsed_url.path)
    
    # Match file name with image extensions
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

        return (max_val > 0.1), f"Match score: {max_val:.2f}"  # threshold tweakable
    except Exception as e:
        return False, str(e)

# Merge and process
for _, row in df1.iterrows():
    cust_id = str(row['cust_id'])
    account_no = str(row['account_no'])

    # Get image_url from excel2
    img_url_row = df2[df2['account_no'].astype(str) == account_no]
    if img_url_row.empty:
        results.append([cust_id, account_no, None, None, "image url not found"])
        continue

    img_url = img_url_row.iloc[0]['image_url']
    # img_name = os.path.basename(urlparse(img_url).path)
    img_name = extract_image_name_from_url(img_url)
    if not img_name:    
        results.append([cust_id, account_no, None, None, "Invalid image name in URL"])
        continue

    full_img_path = os.path.join(folder1_path, img_name)
    cropped_img_path = os.path.join(folder2_path, f"{cust_id}.png")

    match, info = is_signature_present(full_img_path, cropped_img_path)
    results.append([cust_id, account_no, img_name, match, info])

# Final result
result_df = pd.DataFrame(results, columns=["cust_id", "account_no", "full_img_name", "is_match", "info"])
result_df.to_excel("signature_match_results.xlsx", index=False)
print(result_df)
