import cv2
import numpy as np
import open3d as o3d

global points
points = []

def click_event(event, x, y, flags, param):
    """
    Mouse callback function to select points manually.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", param)

def get_homography_matrix(image):
    """
    Compute homography matrix from user-selected points.
    """
    global points
    points = []
    temp_image = image.copy()
    cv2.imshow("Select Points", temp_image)
    cv2.setMouseCallback("Select Points", click_event, temp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required to compute homography.")
    
    world_points = np.array([[0, 0], [800, 0], [0, 600], [800, 600]])
    H, _ = cv2.findHomography(np.array(points), world_points)
    return H

def warp_to_bev(image, H, output_size=(800, 600)):
    """
    Apply homography transformation to warp the image to BEV.
    """
    bev_image = cv2.warpPerspective(image, H, output_size)
    return bev_image

def visualize_point_cloud(bev_image):
    """
    Convert BEV image to a point cloud for visualization.
    """
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    points = np.column_stack(np.where(gray > 50))  # Extract non-black pixels
    colors = bev_image[points[:, 0], points[:, 1]] / 255.0
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd])

def main():
    # Load images
    image1 = cv2.imread('image1.jpg')  # Replace with actual image paths
    image2 = cv2.imread('image2.jpg')
    
    # Compute homography matrix with manual point selection
    print("Select 4 points for Image 1")
    H1 = get_homography_matrix(image1)
    print("Select 4 points for Image 2")
    H2 = get_homography_matrix(image2)
    
    # Transform images to BEV
    bev_image1 = warp_to_bev(image1, H1)
    bev_image2 = warp_to_bev(image2, H2)
    
    # Show results
    cv2.imshow("BEV View 1", bev_image1)
    cv2.imshow("BEV View 2", bev_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Visualize as point cloud
    visualize_point_cloud(bev_image1)
    visualize_point_cloud(bev_image2)

if __name__ == "__main__":
    main()
