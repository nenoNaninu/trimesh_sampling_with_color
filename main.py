import trimesh
import numpy as np


def sample_surface_with_color(mesh, count, face_weight=None):
    """
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    face_weight : None or len(mesh.faces) float
      Weight faces by a factor other than face area.
      If None will be the same as face_weight=mesh.area
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """

    if face_weight is None:
        # len(mesh.faces) float, array of the areas
        # of each face of the mesh
        face_weight = mesh.area_faces

    # cumulative sum of weights (len(mesh.faces))
    weight_cum = np.cumsum(face_weight)

    # last value of cumulative sum is total summed weight/area
    face_pick = np.random.random(count) * weight_cum[-1]
    # get the index of the selected faces
    face_index = np.searchsorted(weight_cum, face_pick)

    tri_origins = mesh.vertices[mesh.faces[:, 0]]  # (N, 3)
    tri_vectors = mesh.vertices[mesh.faces[:, 1:]].copy()  # (N, 2, 3)
    tri_origins_tile = np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
    tri_vectors -= tri_origins_tile

    uv_origins = mesh.visual.uv[mesh.faces[:, 0]]  # (N, 2)
    uv_vectors = mesh.visual.uv[mesh.faces[:, 1:]].copy()  # (N, 2, 2)
    uv_origins_tile = np.tile(uv_origins, (1, 2)).reshape((-1, 2, 2))
    uv_vectors -= uv_origins_tile

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]  # (N, 3)
    tri_vectors = tri_vectors[face_index]  # (N, 2, 3)

    uv_origins = uv_origins[face_index]  # (N, 2)
    uv_vectors = uv_vectors[face_index]  # (N, 2, 2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))  # (N, 2, 1)

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)  # (N, 2, 1)

    sample_vector = (tri_vectors * random_lengths).sum(axis=1)  # (N, 3)
    sample_uv_vector = (uv_vectors * random_lengths).sum(axis=1)  # (N, 3)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins
    uv_samples = sample_uv_vector + uv_origins

    texture = mesh.visual.material.image
    colors = uv_to_interpolation_color(uv_samples, texture)

    return samples, face_index, colors


def uv_to_interpolation_color(uv, image):
    """
    Get the color from texture image using bilinear sampling.

    Parameters
    -------------
    uv : (n, 2) float
      UV coordinates on texture image
    image : PIL.Image
      Texture image

    Returns
    ----------
    colors : (n, 4) float
      RGBA color at each of the UV coordinates
    """
    if image is None or uv is None:
        return None

    # UV coordinates should be (n, 2) float
    uv = np.asanyarray(uv, dtype=np.float64)

    # get texture image pixel positions of UV coordinates
    x = (uv[:, 0] * (image.width - 1))
    y = ((1 - uv[:, 1]) * (image.height - 1))

    x_floor = np.floor(x).astype(np.int64) % image.width
    y_floor = np.floor(y).astype(np.int64) % image.height

    x_ceil = np.ceil(x).astype(np.int64) % image.width
    y_ceil = np.ceil(y).astype(np.int64) % image.height

    dx = x % image.width - x_floor
    dy = y % image.height - y_floor

    img = np.asanyarray(image.convert('RGBA'))

    colors00 = img[y_floor, x_floor]
    colors01 = img[y_ceil, x_floor]
    colors10 = img[y_floor, x_ceil]
    colors11 = img[y_ceil, x_ceil]

    a00 = (1 - dx) * (1 - dy)
    a01 = dx * (1 - dy)
    a10 = (1 - dx) * dy
    a11 = dx * dy

    a00 = np.repeat(a00[:, None], 4, axis=1)
    a01 = np.repeat(a01[:, None], 4, axis=1)
    a10 = np.repeat(a10[:, None], 4, axis=1)
    a11 = np.repeat(a11[:, None], 4, axis=1)

    colors = a00 * colors00 + a01 * colors01 + a10 * colors10 + a11 * colors11

    # conversion to RGBA should have corrected shape
    assert colors.ndim == 2 and colors.shape[1] == 4

    return colors.astype(np.uint8)


if __name__ == "__main__":

    src_path = './example/fuze.obj'
    dst_path = 'result/fuze5.ply'

    mesh = trimesh.load(src_path)
    sample, _, color = sample_surface_with_color(mesh, 100000)

    pcd = trimesh.points.PointCloud(sample, color)
    pcd.export(dst_path)
