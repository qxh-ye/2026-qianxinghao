import os
import cv2

def save_processed_images(save_dir, filename, gray, binary, edges):
    cv2.imwrite(
        os.path.join(save_dir, "gray_" + filename),
        gray
    )

    cv2.imwrite(
        os.path.join(save_dir, "binary_" + filename),
        binary
    )

    cv2.imwrite(
        os.path.join(save_dir, "edge_" + filename),
        edges
    )
