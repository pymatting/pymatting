### 1.1.9

- Add Shared Matting

### 1.1.7

- Add sanity check for image and trimap.

### 1.1.6

- Add `kernel` parameter for `knn_laplacian`.

### 1.1.5

- Add `relative_discard_threshold` for `ichol` preconditioner.

### 1.1.4

- Switch back to `njit` because ahead-of-time compilation caused too many issues with installation.

### 1.1.3

- Add optimization for `estimate_alpha_cf` which should reduce computation time if most pixels in the trimap are known.
- Allow sloppy trimaps.

### 1.1.2

- Recompile ahead-of-time-compiled modules if they are out of date.
- Add a gradient weighting term to `estimate_foreground_ml`.

### 1.1.1

- Compile on first import instead of during build to simplify PyPI upload process.

### 1.1.0

- Replace just-in-time compilation with ahead-of-time compilation for faster import times.
