import numpy as np

def generate_cov_matrix(dim=10, strength=0.85, seed=None):
    """
    Generate a random positive-definite covariance matrix.
    Uses A = L @ L.T + diag to guarantee PD.
    """
    if seed is not None:
        np.random.seed(seed)

    L = np.tril(np.random.randn(dim, dim) * strength)
    S = L @ L.T
    S += np.diag(np.random.uniform(1, 3, dim))  # diagonal boost for strict PD

    # Verify
    np.linalg.cholesky(S)  # will raise if not PD
    return S

def to_yaml(matrices):
    lines = []
    for i, M in enumerate(matrices):
        for ri, row in enumerate(M):
            vals = ",".join(f"{v:.3g}" for v in row)
            if ri == 0:
                lines.append(f"  - [[{vals}],")
            elif ri == len(M) - 1:
                lines.append(f"     [{vals}]]")
            else:
                lines.append(f"     [{vals}],")
    return "\n".join(lines)


if __name__ == "__main__":
    matrices = [generate_cov_matrix(dim=10, strength=0.85, seed=i) for i in range(3)]

    yaml_str = to_yaml(matrices)

    with open("tmp.yaml", "w") as f:
        f.write(yaml_str + "\n")

    print("Saved to tmp.yaml")