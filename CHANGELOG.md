### 1.1.3
- Add optimization for `estimate_alpha_cf` which should reduce computation time if most pixels in the trimap are known.
- Allow sloppy trimaps.

### 1.1.2

- Recompile ahead-of-time-compiled modules if they are out of date.
- Add a gradient weighting term to `estimate_foreground_ml`.

### 1.1.1

- Compile on first import instead of during build to simplfy PyPI upload process.

### 1.1.0

- Replace just-in-time compilation with ahead-of-time compilation for faster import times.
