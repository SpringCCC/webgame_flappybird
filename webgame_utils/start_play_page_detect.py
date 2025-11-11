from skimage.metrics import structural_similarity as ssim
import cv2




def images_similar_ssim(img1, img2, ssim_thresh=0.95):
    if img1.shape != img2.shape:
        return False
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY) if img1.shape[2]==4 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY) if img2.shape[2]==4 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score = ssim(gray1, gray2)
    return score >= ssim_thresh

