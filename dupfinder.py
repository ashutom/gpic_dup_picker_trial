import os
import csv
import logging
from PIL import Image
import imagehash
from pillow_heif import register_heif_opener
import numpy as np

# Import our custom logging module
from dupflogging import setup_logging, log_system_info

# Register HEIF opener to enable HEIC support
register_heif_opener()

def get_image_paths(root_dir, image_extensions):
    """Generator to yield all image file paths from the root directory"""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                yield os.path.join(dirpath, filename)

def calculate_phash(image_path):
    """Calculate perceptual hash for an image"""
    try:
        with Image.open(image_path) as img:
            phash = str(imagehash.phash(img))
            logging.debug(f"Calculated pHash for {image_path}: {phash}")
            return phash
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None

def find_similar_hash(phash, phash_dict, threshold=5):
    """Find all hashes with minimum hamming distance within the threshold"""
    if not phash_dict:
        return []
        
    current_hash = imagehash.hex_to_hash(phash)
    min_distance = threshold  # Only consider distances less than threshold
    min_distance_hashes = []
    
    for existing_hash in phash_dict.keys():
        existing_hash_obj = imagehash.hex_to_hash(existing_hash)
        hamming_distance = current_hash - existing_hash_obj
        
        if hamming_distance < min_distance:
            # Found a new minimum distance - clear list and update
            min_distance = hamming_distance
            min_distance_hashes = [existing_hash]
        elif hamming_distance == min_distance:
            # Found another hash with same minimum distance - add to list
            min_distance_hashes.append(existing_hash)
    
    return min_distance_hashes

def create_phash_dictionary(root_dir):
    """Create dictionary of pHash groups by scanning all images in root directory"""
    # Define supported image extensions (now including HEIC)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.heic', '.heif'}
    
    # Initialize dictionary to store phash as key and list of image paths as value
    phash_dict = {}
    
    logging.info(f"Starting image scan in directory: {root_dir}")
    processed_count = 0
    
    # Find all images and calculate pHash for each
    for img_path in get_image_paths(root_dir, image_extensions):
        phash = calculate_phash(img_path)
        if phash is None:
            continue
            
        processed_count += 1
        if processed_count % 50 == 0:  # Progress indicator
            logging.info(f"Processed {processed_count} images...")
        
        # Compare hash with existing keys and check hamming distance
        similar_hashes = find_similar_hash(phash, phash_dict)
        
        if similar_hashes:
            # Add to the first (or best representative) group with minimum distance
            # You could also implement logic to merge groups if needed
            best_hash = similar_hashes[0]  # Use first one as representative
            phash_dict[best_hash].append(img_path)
            logging.debug(f"Added {img_path} to existing group {best_hash[:8]}...")
        else:
            # Create new group with this phash
            phash_dict[phash] = [img_path]
            logging.debug(f"Created new group for {img_path} with hash {phash[:8]}...")
    
    logging.info(f"Finished processing {processed_count} images.")
    logging.info(f"Found {len(phash_dict)} unique hash groups.")
    
    return phash_dict

def get_valid_directory():
    """Get and validate root directory from user input with OS-aware path handling"""
    while True:
        root_dir = input("Enter the root directory: ").strip()
        
        # Remove quotes if user wrapped path in quotes
        if root_dir.startswith('"') and root_dir.endswith('"'):
            root_dir = root_dir[1:-1]
        elif root_dir.startswith("'") and root_dir.endswith("'"):
            root_dir = root_dir[1:-1]
        
        # Normalize path for current OS
        root_dir = os.path.normpath(os.path.expanduser(root_dir))
        
        # Check if directory exists
        if os.path.exists(root_dir) and os.path.isdir(root_dir):
            logging.info(f"Valid directory selected: {root_dir}")
            return root_dir
        else:
            logging.warning(f"Invalid directory entered: {root_dir}")
            logging.error("Please enter a valid directory path.")

def write_results_to_csv(phash_dict, output_dir=None):
    """Write pHash dictionary results to CSV file and display summary"""
    csv_filename = 'duplicates_images.csv'
    
    # Use provided output directory or current working directory
    if output_dir:
        csv_path = os.path.join(output_dir, csv_filename)
    else:
        csv_path = os.path.join(os.getcwd(), csv_filename)
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['pHash', 'Image_Paths'])
            
            # Write data - each row has phash as first column, then all image paths
            for phash, paths in phash_dict.items():
                # Create a row with phash as first element, followed by all paths
                row = [phash] + paths
                writer.writerow(row)
        
        logging.info(f"Results saved to CSV file: {csv_path}")
        
        # Show summary of groups with multiple images (potential duplicates)
        duplicate_groups = {k: v for k, v in phash_dict.items() if len(v) > 1}
        if duplicate_groups:
            logging.info(f"Found {len(duplicate_groups)} groups with potential duplicates")
            for phash, paths in duplicate_groups.items():
                logging.debug(f"Group {phash[:8]}... has {len(paths)} images: {paths}")
                logging.info(f"Group {phash[:8]}... has {len(paths)} images")
        else:
            logging.info("No duplicate images found")
            
    except Exception as e:
        logging.error(f"Error writing CSV file: {e}")

def review_duplicates(csv_path):
    """Read CSV file and display duplicate images for user review"""
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return
    
    logging.info(f"Starting duplicate review from CSV: {csv_path}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # Skip header row
            next(reader, None)
            
            # First pass: count total duplicate groups
            duplicate_rows = []
            for row in reader:
                # Ignore rows with 2 or fewer columns (no duplicates)
                if len(row) > 2:
                    duplicate_rows.append(row)
            
            total_groups = len(duplicate_rows)
            logging.info(f"Found {total_groups} duplicate groups to review")
            
            if total_groups == 0:
                logging.info("No duplicate groups found in CSV file")
                return
            
            # Second pass: display groups
            group_count = 0
            for row in duplicate_rows:
                group_count += 1
                image_paths = row[1:]  # Ignore first column (pHash)
                
                logging.debug(f"Reviewing group {group_count}/{total_groups} with {len(image_paths)} images")
                logging.info(f"--- Duplicate Group {group_count} ({len(image_paths)} images) of the total {total_groups} duplicate groups ---")
                
                for i, img_path in enumerate(image_paths, 1):
                    if not os.path.exists(img_path):
                        logging.warning(f"Image file not found: {img_path}")
                        logging.info(f"Image {i}: {img_path} (FILE NOT FOUND)")
                        continue
                    
                    logging.info(f"Image {i}: {img_path}")
                    
                    try:
                        # Open and display the image
                        with Image.open(img_path) as img:
                            img.show()
                        logging.debug(f"Successfully displayed image: {img_path}")
                    
                    except Exception as e:
                        logging.error(f"Error displaying image {img_path}: {e}")
                
                # Ask if user wants to continue to next group
                if group_count < total_groups:
                    user_input = input("Press Enter to view next group, or 'q' to quit review: ").strip().lower()
                    logging.debug(f"User input for review continuation: {user_input}")
                    if user_input == 'q':
                        logging.info("Review stopped by user")
                        return
            
            logging.info(f"Completed reviewing all {total_groups} duplicate groups")
                
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")

def select_best_image(image_paths):
    """
    Select the best image from a list based on quality metrics that correlate with visual appeal.
    
    Criteria for 'best' image:
    - Higher resolution (more pixels)
    - Better sharpness (Laplacian variance)
    - Good contrast (standard deviation of luminance)
    - Balanced brightness (not too dark or too bright)
    - Smaller file compression artifacts
    
    Args:
        image_paths (list): List of image file paths to compare
    
    Returns:
        str: Path to the best image, or None if all images fail to process
    """
    if not image_paths:
        logging.warning("Empty image list provided to select_best_image")
        return None
    
    if len(image_paths) == 1:
        logging.debug(f"Single image in list, returning: {image_paths[0]}")
        return image_paths[0]
    
    logging.info(f"Analyzing {len(image_paths)} images to select best quality")
    best_image = None
    best_score = -1
    
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                # Convert to RGB if necessary for consistent processing
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array for analysis
                img_array = np.array(img)
                
                # 1. Resolution score (normalized by max resolution in the group)
                resolution = img.width * img.height
                
                # 2. Sharpness score using Laplacian variance
                # Convert to grayscale for sharpness calculation
                gray = img.convert('L')
                gray_array = np.array(gray)
                # Apply Laplacian filter to detect edges/sharpness
                laplacian_var = np.var(np.gradient(np.gradient(gray_array)))
                
                # 3. Contrast score using standard deviation of luminance
                # Calculate luminance using standard RGB to grayscale conversion
                luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
                contrast = np.std(luminance)
                
                # 4. Brightness balance score (penalize very dark or very bright images)
                mean_brightness = np.mean(luminance)
                # Optimal brightness is around 128 (middle gray), create score that peaks there
                brightness_balance = 1.0 - abs(mean_brightness - 128) / 128
                
                # 5. Color richness (standard deviation across color channels)
                color_richness = np.mean([np.std(img_array[:,:,i]) for i in range(3)])
                
                # Combine scores with weights
                # Normalize each metric to prevent any single metric from dominating
                resolution_score = resolution / 1000000  # Normalize by 1 megapixel
                sharpness_score = min(laplacian_var / 1000, 10)  # Cap to prevent extreme values
                contrast_score = min(contrast / 50, 5)  # Normalize contrast
                brightness_score = brightness_balance * 2  # Scale brightness balance
                color_score = min(color_richness / 50, 3)  # Normalize color richness
                
                # Weighted combination (adjust weights based on importance)
                total_score = (
                    resolution_score * 0.25 +      # 25% - Resolution
                    sharpness_score * 0.30 +       # 30% - Sharpness (most important for visual appeal)
                    contrast_score * 0.20 +        # 20% - Contrast
                    brightness_score * 0.15 +      # 15% - Brightness balance
                    color_score * 0.10              # 10% - Color richness
                )
                
                logging.debug(f"Image {img_path} scores - Resolution: {resolution_score:.2f}, "
                            f"Sharpness: {sharpness_score:.2f}, Contrast: {contrast_score:.2f}, "
                            f"Brightness: {brightness_score:.2f}, Color: {color_score:.2f}, "
                            f"Total: {total_score:.2f}")
                
                if total_score > best_score:
                    best_score = total_score
                    best_image = img_path
                    
        except Exception as e:
            logging.error(f"Error analyzing image {img_path}: {e}")
            continue
    
    logging.info(f"Selected best image: {best_image} with score: {best_score:.2f}")
    return best_image


def main():
    """Main function with configurable logging levels"""
    # Configure logging levels based on requirements
    # You can change these levels as needed:
    # logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
    
    console_log_level = logging.INFO    # What appears on console
    app_log_level = logging.INFO        # What goes to application log
    # Journal log is always DEBUG (captures everything)
    
    # Setup comprehensive logging system
    setup_logging(console_level=console_log_level, app_level=app_log_level)
    logging.info("DupFinder application started")
    
    # Log system information for debugging
    log_system_info()
    
    # Get and validate root directory from user
    root_dir = get_valid_directory()
    
    # Create pHash dictionary by scanning images
    phash_dict = create_phash_dictionary(root_dir)
    
    # Write results to CSV file in the root directory
    csv_path = os.path.join(root_dir, 'duplicates_images.csv')
    write_results_to_csv(phash_dict, root_dir)
    
    # Ask user if they want to review duplicates
    if any(len(paths) > 1 for paths in phash_dict.values()):
        review_choice = input("\nWould you like to review duplicate images? (y/n): ").strip().lower()
        logging.debug(f"User choice for duplicate review: {review_choice}")
        if review_choice in ['y', 'yes']:
            review_duplicates(csv_path)
    
    # End
    logging.info("DupFinder application completed successfully")

if __name__ == "__main__":
    main()
