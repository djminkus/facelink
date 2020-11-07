import numpy as np
import cv2
import os
import math
import matplotlib as plt
import sqlite3

faces_folder = 'user_faces'

# This might be useful: https://pypi.org/project/face-recognition/


# Create a "montage" image of small images.
def montage(all_images, shrink_factor=0.5, highlight_image_idx=None):
    count = len(all_images)
    shrunken_image = cv2.resize(all_images[0], dsize=None, fx=shrink_factor, fy=shrink_factor)
    m, n, _ = shrunken_image.shape

    mm = int(math.ceil(math.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n), dtype=np.uint8)

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            shrunken_image = cv2.resize(all_images[image_id], dsize=None, fx=shrink_factor, fy=shrink_factor)

            # Black out this image if desired.
            if not highlight_image_idx is None and image_id == highlight_image_idx:
                shrunken_image = 0

            sliceM, sliceN = j * m, k * n
            M[sliceM:sliceM + m, sliceN:sliceN + n] = shrunken_image
            image_id += 1
    return M


def read_images(faces_folder):
    all_images = []
    for subfolder in os.listdir(faces_folder):
        path_to_folder = os.path.join(faces_folder, subfolder)
        if os.path.isdir(path_to_folder):
            for filename in os.listdir(path_to_folder):
                path_to_file = os.path.join(path_to_folder, filename)
                gray_img = cv2.imread(path_to_file, cv2.COLOR_BGR2GRAY)
                all_images.append(gray_img)
    return all_images


def open_or_create_db():
    conn = sqlite3.connect('test.db')
    print("Opened database successfully")

    conn.execute('''CREATE TABLE IF NOT EXISTS COMPANY
             (ID INT PRIMARY KEY     NOT NULL,
             NAME           TEXT    NOT NULL,
             AGE            INT     NOT NULL,
             DEGREE         CHAR(35),
             SCHOOL         CHAR(35),
             GPA            REAL,
             IMG_FOLDER   TEXT);''')
    print("Table created successfully")

    conn.execute("INSERT INTO COMPANY (ID,NAME,AGE,DEGREE,SCHOOL,GPA, IMG_FOLDER) \
          VALUES (1, 'David Garrett', 29, 'B.S. Physics (Engineering)', 'University of Northern Colorado', 3.8, 'U1')")

    conn.commit()
    print("Records created successfully")

    conn.close()


def do(doMontage=True):
    # Open or create the database:
    # open_or_create_db()

    # Read all N images into a list, where each element is size [H,W].
    all_images = read_images(faces_folder)
    N = len(all_images)
    H, W, _ = all_images[0].shape
    print("Read %d images." % N)

    #TODO: replace right side with image from camera
    idx = np.random.randint(0, N)
    query_image = all_images[idx]

    if doMontage is True:
        # For visualization, display a montage of all images.
        montage_image = montage(all_images, highlight_image_idx=idx)
        cv2.imshow("All faces", montage_image)

    # Put image data into a 2D array of vectors.  Output is size [H*W, N].
    all_vectors = np.array(all_images).reshape(N, H * W).T

    # Calculate the mean (average) vector.
    mean_vector = np.mean(all_vectors, axis=1)

    # Subtract the mean vector from all vectors.
    all_vectors = all_vectors - mean_vector.reshape((H * W, 1))

    M = all_vectors.T @ all_vectors  # This is size [N,N]

    # Get eigenvalues and eigenvectors.
    # There are N eigenvalues in w, sorted from highest to lowest.
    # There are N eigenvectors in v, each of length N.
    # Each column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    w, v = np.linalg.eig(M)

    # Calculate principal components and keep the top K pieces.
    pcs = all_vectors @ v
    K = 16
    pcs = pcs[:, 0:K]

    # Here are the input vectors projected onto the principal components.
    signatures = np.zeros((N, K))
    for i in range(N):
        # Each row corresponds to an input vector, and contains the K values resulting
        # from projecting the input vector onto the K principal components.
        signatures[i, :] = all_vectors[:, i].T @ pcs  # Each row is an image signature

    # Prepare the query face, by making it a vector and subtracting the mean.
    query_vector = query_image.reshape(H * W) - mean_vector

    # Project the query vector onto the space of principal components.
    query_projected = query_vector @ pcs  # Result is size (K,)

    print("Here is the signature of the query face.")
    print("These are the coefficients of the query face projected onto the %d PCs:" % K)
    print(query_projected)

    # Ok, compare the query projection vector with all rows in "signatures".
    diff = signatures - query_projected
    euclidean_distances = np.linalg.norm(diff, axis=1)
    idx_sorted = np.argsort(euclidean_distances)
    print("Top matches:", idx_sorted[0:10])

    # Get the images for the top 4 matches.
    matched_images = [all_images[id] for id in idx_sorted[0:4] ]
    montage_image = montage(matched_images, shrink_factor=1.0)
    cv2.imshow("Top matches", montage_image)

    cv2.waitKey(0)

    # TODO: LOAD PICTURES OF OURSELVES INTO THE user_faces DIRECTORY.
    #  Then associate each image with a profile.
    #  We will probably need a database for this, so here's a tutorial on databases in Python:
    #  https://www.tutorialspoint.com/python_network_programming/python_databases_and_sql.htm


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    do()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
