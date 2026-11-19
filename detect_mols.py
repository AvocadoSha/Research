#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import argparse

def compact_formula(formula):
    elements = {}
    for element in formula:
        elements[element] = elements.get(element, 0) + 1
    return ''.join(f"{e}{n if n > 1 else ''}" for e, n in sorted(elements.items()))

def minimum_image_distance(vec, box):
    """Apply minimum image convention for periodic boundary conditions."""
    return vec - box * np.round(vec / box)

def build_cell_list(positions, cutoff, box):
    """
    Build cell list for neighbor search.
    
    Returns:
        cell_indices: which cell each atom belongs to
        cells: list of atoms in each cell (3D list converted to flattened)
        n_cells: number of cells in each direction
        cell_length: cell edge length
    """
    n_cells = np.floor(box / cutoff).astype(int)
    n_cells = np.maximum(n_cells, 1)  # at least 1 cell per dimension
    cell_length = box / n_cells

    # Compute cell indices for each particle
    fractional = positions / cell_length
    cell_indices = np.floor(fractional).astype(int) % n_cells

    # Build empty cells list
    cells = {}
    for ix in range(n_cells[0]):
        for iy in range(n_cells[1]):
            for iz in range(n_cells[2]):
                cells[(ix, iy, iz)] = []

    # Assign atoms to cells
    for atom_i, (ix, iy, iz) in enumerate(cell_indices):
        cells[(ix, iy, iz)].append(atom_i)

    return cell_indices, cells, n_cells, cell_length

def neighboring_cells(ix, iy, iz, n_cells):
    """Generate indices of neighboring cells including the current cell, with PBC."""
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield ((ix + dx) % n_cells[0],
                       (iy + dy) % n_cells[1],
                       (iz + dz) % n_cells[2])

def detect_molecules(positions, elements, cutoff, box):
    """
    Detect molecules using cell-linked neighbor search with PBC.
    """
    elements = np.array(elements)
    n = len(positions)

    cell_indices, cells, n_cells, cell_length = build_cell_list(positions, cutoff, box)

    # Build adjacency matrix
    adjacency_matrix = np.zeros((n, n), dtype=bool)

    for ix in range(n_cells[0]):
        for iy in range(n_cells[1]):
            for iz in range(n_cells[2]):
                # Atoms in this cell
                atoms_in_cell = cells[(ix, iy, iz)]
                if not atoms_in_cell:
                    continue

                # Check neighbors in current and neighboring cells
                for nx, ny, nz in neighboring_cells(ix, iy, iz, n_cells):
                    atoms_in_neigh = cells[(nx, ny, nz)]
                    if not atoms_in_neigh:
                        continue

                    # Compute pairwise distance checks
                    for i in atoms_in_cell:
                        for j in atoms_in_neigh:
                            if j <= i:
                                continue  # avoid double counting and self distance

                            delta = positions[j] - positions[i]
                            delta = minimum_image_distance(delta, box)
                            dist = np.linalg.norm(delta)
                            if dist < cutoff:
                                adjacency_matrix[i, j] = True
                                adjacency_matrix[j, i] = True

    visited = np.zeros(n, dtype=bool)
    molecules = []

    for i in range(n):
        if not visited[i]:
            stack = [i]
            molecule = []
            while stack:
                atom = stack.pop()
                if not visited[atom]:
                    visited[atom] = True
                    molecule.append(atom)
                    neighbors = np.where(adjacency_matrix[atom])[0]
                    stack.extend(neighbors[~visited[neighbors]])
            molecules.append(molecule)

    chemical_formulas = [compact_formula(''.join(sorted(elements[mol]))) for mol in molecules]
    return chemical_formulas

def parse_lammps_dump(filename):
    frames = []
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("ITEM: TIMESTEP"):
                _ = f.readline()            # timestep; unused here
                _ = f.readline()            # ITEM: NUMBER OF ATOMS
                natoms = int(f.readline())
                _ = f.readline()            # ITEM: BOX BOUNDS ...
                box_bounds = [list(map(float, f.readline().split())) for _ in range(3)]
                _ = f.readline()            # ITEM: ATOMS header line

                atoms = [f.readline().split() for _ in range(natoms)]
                elements = [atom[1] for atom in atoms]
                coords = np.array([[float(atom[2]), float(atom[3]), float(atom[4])] for atom in atoms])

                box = np.array([hi - lo for lo, hi in box_bounds])
                coords -= np.array([lo for lo, _ in box_bounds])

                frames.append((coords, elements, box))
    return frames

def main():
    parser = argparse.ArgumentParser(description="Molecule detection with cell-linked list neighbor search and PBC")
    parser.add_argument("dumpfile", help="Path to LAMMPS dump file")
    parser.add_argument("--cutoff", type=float, default=3.0, help="Bond cutoff [Å]")
    parser.add_argument("--timestep", type=float, default=0.01, help="Time step between frames [ps]")
    args = parser.parse_args()

    frames = parse_lammps_dump(args.dumpfile)
    if not frames:
        raise RuntimeError("No frames found in dump file!")

    elements = np.array(frames[0][1])

    species_time_course = defaultdict(list)

    for coords, _, box in tqdm(frames, desc="Processing frames", unit="frame"):
        molecules = detect_molecules(coords, elements, args.cutoff, box)
        unique, counts = np.unique(molecules, return_counts=True)
        count_dict = dict(zip(unique, counts))
        all_species = set(species_time_course.keys()) | set(count_dict.keys())
        for spec in all_species:
            species_time_course[spec].append(count_dict.get(spec, 0))

    n_frames = len(frames)
    time = np.arange(n_frames) * args.timestep

    for spec in species_time_course:
        arr = np.array(species_time_course[spec])
        if len(arr) < n_frames:
            species_time_course[spec] = np.pad(arr, (0, n_frames - len(arr)), 'constant')
        else:
            species_time_course[spec] = arr[:n_frames]

    species_variance = {spec: np.var(counts) for spec, counts in species_time_course.items()}
    important_species = sorted(species_variance, key=species_variance.get, reverse=True)[:6]

    # Plot time courses
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages('species_time_course.pdf') as pdf:
        plt.figure(figsize=(12,7))
        for spec in important_species:
            plt.plot(time, species_time_course[spec], label=spec, linewidth=1)
        plt.xlabel("Time [ps]", fontsize=14)
        plt.ylabel("Number of molecules", fontsize=14)
        plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Write data files
    for spec in important_species:
        with open(f"{spec}_timecourse.dat", "w") as f:
            f.write("# Time[ps]    Count\n")
            for t, c in zip(time, species_time_course[spec]):
                f.write(f"{t:.6f}    {c}\n")

    # Write summary statistics
    with open("molecular_species_statistics.txt", "w") as f:
        f.write("Summary Statistics:\n")
        for spec, counts in species_time_course.items():
            f.write(f"\n{spec}:\n")
            f.write(f"  Average count: {np.mean(counts):.2f}\n")
            f.write(f"  Maximum count: {np.max(counts)}\n")
            f.write(f"  Minimum count: {np.min(counts)}\n")
            f.write(f"  Final count: {counts[-1]}\n")
            f.write(f"  Variance: {np.var(counts):.2f}\n")
            f.write(f"  Data points: {len(counts)}\n")

    print("Done!")

if __name__ == '__main__':
    main()

