# 🌈 Image Thresholding and Segmentation

## 📌 Overview

This project implements a comprehensive suite of image segmentation and thresholding algorithms, all developed **from fundamental principles** without relying on high-level library functions. It tackles both **Grayscale Thresholding** using advanced statistical methods (Optimal, Otsu's) and **Color Segmentation** using unsupervised machine learning techniques (k-Means, Mean Shift, Clustering) in the **LUV color space**.

---

## Features

- ✅ **Statistical Thresholding:** Optimal and highly optimized **Otsu's Method** for global binary segmentation.
- ✅ **Spectral Thresholding:** Applying thresholding in the **Frequency Domain** using a custom FFT.
- ✅ **Color Space Transformation:** Manual implementation of **RGB to LUV** conversion.
- ✅ **Clustering Segmentation:** **k-Means**, **Agglomerative Clustering**, and **Mean Shift** for color image segmentation.
- ✅ **Region-Based Segmentation:** **Region Growing** with adaptive seeding and thresholds.

---

## Methodology (Algorithm Implementation)

### 1. Grayscale Thresholding
These methods focus on finding the best intensity value ($T$) to separate foreground and background pixels by analyzing the image histogram.

* **Optimal Thresholding:** An iterative process that calculates a new threshold $T$ by finding the mean of the two segmented classes until $T$ converges.
* **Otsu's Method:**
    * Finds the threshold that **maximizes the inter-class variance** between the two resulting groups.
    * Implementation uses **cumulative sums** for fast calculation of mean and variance at every possible threshold.
* **Spectral Thresholding:**
    * The image is transformed into the frequency domain using a custom **Fast Fourier Transform (FFT)** implementation.
    * Thresholding is then applied to frequency components to analyze patterns (global) or local details.



### 2. Color Image Segmentation
Segmentation is performed in the **LUV color space** for perceptually uniform results, leveraging the luminance (L) and chrominance (U, V) components.

* **RGB to LUV Conversion:** A manual transformation of the RGB values using linear and non-linear mappings, ensuring accurate color representation.
* **Unsupervised Clustering:**
    * **k-Means:** Groups pixels based on spatial and LUV color features using **Euclidean distance**. Includes a custom implementation of the **elbow method** for automated *k* selection.
    * **Mean Shift:** A non-parametric clustering technique that locates the modes (peaks) of the pixel density function in the LUV color space for smooth, contiguous regions.
    * **Agglomerative Clustering:** A hierarchical method that starts with each pixel as its own cluster and merges the closest pairs based on **single/complete linkage**.
* **Region Growing:** A seed-based method where regions expand by adding neighboring pixels that meet a defined **adaptive intensity threshold**.

---

🖼️ **Screenshots**:

1. Otsu's Thresholding
   
<img width="1495" height="978" alt="Otsu thresholding global" src="https://github.com/user-attachments/assets/233c95ad-0a1b-4fb6-b9fe-e6e2661715d3" />

2.Optimal Thresholding

<img width="1482" height="983" alt="Optimal thresholding local" src="https://github.com/user-attachments/assets/da585459-8adc-4837-8365-c43ef0e6bc7d" />

3.Spectral Thresholding

<img width="1481" height="979" alt="Spectral thresholding global" src="https://github.com/user-attachments/assets/04b7968f-f1a8-4122-bcb6-c643206f2f13" />

4.k-Means Clustering

<img width="1475" height="968" alt="K-means clustering" src="https://github.com/user-attachments/assets/e93ecc84-c8d0-405c-909a-b1cde6a40778" />


5.Agglomerative Clustering 

<img width="1501" height="974" alt="Agglomerative clustering" src="https://github.com/user-attachments/assets/84b11c82-aaee-4134-aa3a-2792b55d7a19" />

6.Region Growing

<img width="1487" height="976" alt="region growing" src="https://github.com/user-attachments/assets/3c67592a-e8b7-4c2b-afc1-9950793672b8" />


---

## 📊 Performance Notes

| Algorithm | Optimization Focus | Key Result |
|---|---|---|
| **Otsu's Method** | Cumulative Sums | High speed and robust global thresholding. |
| **k-Means** | Feature Space Selection | Effective segmentation by combining color and spatial information. |
| **Mean Shift** | Kernel Density Estimation | Provides smoother, boundary-free segmentation compared to k-Means. |
| **Spectral Thresholding** | Custom FFT Implementation | Demonstrates the ability to filter global/local patterns in the frequency domain. |

---

## 🛠️ Technologies

- **Python 3.x**
- **NumPy:** Essential for high-performance matrix operations and mathematical computations.
- **OpenCV (cv2):** Used only for image loading and display utility.
- **Matplotlib:** Used for visualizing the elbow curve in k-Means and result visualization.

---

## 📈 Future Work

- Implement **Active Contour Models (Snakes)** for boundary detection.
- Explore **Graph-Based Segmentation** algorithms (e.g., Graph Cuts).
- Extend clustering to include spatial coordinates for more accurate boundary preservation.

---

## Contributor

<div>
<table align="center">
  <tr>
        <td align="center">
      <a href="https://github.com/Mazenmarwan023" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127551364?v=4" width="150px;" alt="Mazen Marwan"/>
        <br />
        <sub><b>Mazen Marwan</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/mohamedddyasserr" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126451832?v=4" width="150px;" alt="Mohamed yasser"/>
        <br />
        <sub><b>Mohamed yasser</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Seiftaha" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/127027353?v=4" width="150px;" alt="Saif Mohamed"/>
        <br />
        <sub><b>Saif Mohamed</b></sub>
      </a>
    </td> 
  </tr>
</table>
</div>


---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
