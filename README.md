# Serotina

A semi-analytic model (SAM) for the cosmic co-evolution of supermassive black holes and their host galaxies.

Serotina walks dark matter merger trees forward in time, painting black holes onto halos and evolving them via accretion and mergers from the epoch of seeding to the present day. It tracks black hole masses, spins, luminosities, and merger histories across a large population of halos simultaneously.

This code grew from Ricarte & Natarajan (2018a, 2018b), with spin evolution from Ricarte, Narayan & Curd (2023).

---

## Installation

### Prerequisites

Serotina requires Python ≥ 3.8 and the following packages:

```
numpy  scipy  matplotlib  cython  astropy
```

### Install from source

```bash
git clone <repo-url>
cd Serotina-Claude
pip install -e .
```

The `-e` flag installs in editable (development) mode, which is recommended. On first import, Serotina will attempt to compile its Cython extensions automatically via `pyximport`. To pre-compile them explicitly:

```bash
python setup.py build_ext --inplace
```

### Verify the installation

```python
import serotina
print(serotina.SAM)
```

---

## Merger tree input

The SAM requires dark matter merger trees in Serotina's gzipped numpy format (`.bin.gz`). These are typically produced from an N-body simulation or a Press-Schechter generator and are organized in a folder by halo mass and sample index:

```
tree_output_23_20/
    m1.00E+11_n001.bin.gz
    m1.00E+11_n002.bin.gz
    ...
    m1.00E+13_n001.bin.gz
    ...
```

Use `treeNavigator` to locate a specific tree:

```python
import serotina

lorax   = serotina.treeNavigator(path="tree_output_23_20")
treefile = lorax.findClosestTree(1e13, 1)   # closest to 1e13 Msun, sample index 1
```

---

## Quick start

### Python API

```python
import serotina
import numpy as np

lorax    = serotina.treeNavigator(path="tree_output_23_20")
treefile = lorax.findClosestTree(1e13, 1)

np.random.seed(42)   # fix seed for reproducibility

parameters = {
    'silent':                    True,
    'seedingMode':               'dcbh',
    'spinEvolution':             True,
    'MAD_super':                 True,
    'MAD_sub':                   True,
    'powerLawDecline':           2.0,
    'blackHoleMergerProbability': 1.0,
    'MsigmaNormalization':       8.0,
    'MsigmaSlope':               4.0,
}

model  = serotina.SAM(treefile, parameters=parameters)
output = model.evolveUniverse(
    outputNameBase   = None,                # set to a string to write a .pklz file
    savedRedshifts   = [0, 1, 2, 4, 6],    # None saves every timestep
    savedProperties  = ['m_bh', 'spin_bh', 'm_halo', 'redshift'],
    saveMode         = 'mainProgenitor',    # or 'progenitors' for all halos
)

# output is a dict; '_snap' keys hold lists of per-snapshot values
import numpy as np
m_bh = np.array([snap[0] for snap in output['m_bh_snap']])
z    = output['redshift']
print("Final BH mass: {:.2e} Msun".format(m_bh[-1]))
```

### TOML config file

For runs you want to version-control or share, describe the full configuration in a `.toml` file and run it from the command line:

```bash
python run_config.py myrun.toml
```

Or from Python:

```python
serotina.run_from_config("myrun.toml")
```

See `example_config.toml` in the project root for a fully-commented template.

---

## Configuration

All SAM physics is controlled by a parameter dictionary (Python) or a `[parameters]` section (TOML). Unset parameters use their defaults.

### Seeding

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seedingMode` | `'dcbh'` | Seeding prescription: `'dcbh'` (direct collapse), `'popIII'` (Population III remnants), or `'nsc'` (nuclear star clusters) |
| `seedingEfficiency` | `1.0` | Fraction of eligible halos that receive a seed |
| `fixedSeedMass` | `None` | Override seed mass in solar masses; `None` draws from the IMF |
| `minimumSeedingRedshift` | `15` | Lower redshift limit for seeding |
| `maximumSeedingRedshift` | `20` | Upper redshift limit for seeding |

### Accretion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `f_EddCrit` | `3e-2` | Eddington ratio threshold separating burst (quasar) and steady (hot) accretion modes |
| `powerLawDecline` | `None` | Slope of post-peak power-law luminosity decline; `None` disables decline |
| `MAD_super` | `False` | Use magnetically-arrested disk (MAD) model above `f_EddCrit` |
| `MAD_sub` | `False` | Use MAD model below `f_EddCrit` |
| `madness` | `1.0` | MAD efficiency parameter (1 = fully MAD, 0 = SANE) |
| `coherentAccretion` | `True` | Accretion episodes aligned with the disk axis |
| `defaultRadiativeEfficiency` | `0.1` | Radiative efficiency when spin evolution is off |

### Spin evolution

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spinEvolution` | `False` | Enable self-consistent spin evolution during accretion and mergers |
| `spin_seed` | `0.0` | Initial spin of newly seeded black holes |
| `spinMax` | `0.998` | Maximum allowed spin magnitude |
| `diskAlignmentParameters` | `[inf]` | Lense-Thirring alignment timescale parameters (Gyr); `[inf]` = instantaneous |

### Mergers

| Parameter | Default | Description |
|-----------|---------|-------------|
| `blackHoleMergerProbability` | `1.0` | Probability that two black holes merge when their halos merge |
| `majorMergerMassRatio` | `0.1` | Halo mass ratio threshold defining a major merger |
| `mergerKicks` | `False` | Enable gravitational-wave recoil kicks |
| `alignMergerSpins` | `False` | Force spin alignment before merger (gas-rich approximation) |
| `violentMergers` | `False` | Retrograde mergers flip the accretion disk orientation |

### M–σ relation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MsigmaNormalization` | `8.45` | Normalization: log(M/Msun) at σ = 200 km/s |
| `MsigmaSlope` | `5.0` | Power-law slope |
| `useMsigmaCap` | `True` | Prevent BH mass from exceeding the M–σ prediction |

---

## Changing the cosmology

Serotina defaults to Planck 2015 cosmology. To use a different cosmology, call `set_cosmology()` with any `astropy.cosmology` object **before** creating the SAM:

```python
import serotina
from astropy.cosmology import Planck18

serotina.set_cosmology(Planck18)

model = serotina.SAM(treefile, parameters=parameters)
```

This updates all internal constants (`Omega_m`, `H_0`, `h`, etc.) and rebuilds the redshift–time lookup tables used during evolution.

---

## Output format

`evolveUniverse()` returns a Python dict (and optionally writes it to a `.pklz` file). The dict structure depends on `saveMode`:

- **`'mainProgenitor'`**: each `_snap` key holds a list with one entry per saved redshift, each entry being a 1-element list for the main halo's black hole.
- **`'progenitors'`**: each `_snap` key holds a list with one entry per saved redshift, each entry being a list over all progenitor halos at that snapshot.

Standard output keys (depending on `savedProperties`):

| Key | Description |
|-----|-------------|
| `m_bh_snap` | Black hole mass (solar masses) |
| `spin_bh_snap` | Dimensionless spin parameter |
| `m_halo_snap` | Host halo virial mass (solar masses) |
| `m_star_snap` | Host stellar mass (solar masses) |
| `L_bol_snap` | Bolometric luminosity (erg/s) |
| `eddRatio_snap` | Eddington ratio |
| `sigma_snap` | Velocity dispersion (km/s) |
| `redshift` | Redshifts of saved snapshots (not a `_snap` key) |

Black hole merger events are accessible separately:

```python
mergers = model.outputMergers()
# Array with columns: [primary_mass, mass_ratio, redshift, final_spin, chi_eff, initial_spin]
```

---

## Running a batch of trees

For production runs over many trees and halo masses (with multiprocessing), see `runSAM.py` in the project root. It loops over halo masses, runs trees in parallel with `multiprocessing.Pool`, and saves one `.pklz` file per mass bin.

Post-processing and plotting are handled by `serotina.grove_analysis` (see `serotina/grove_analysis/`) and `serotina.tree_analysis.SAM_Analysis`.

---

## Citation

If you use Serotina in your research, please cite Ricarte, Natarajan, Narayan, and Palumbo (2025). Other relevant works include:
- Ricarte & Natarajan (2018a) — early SAM framework and growth models
- Ricarte & Natarajan (2018b) — exploration of seeding predictions
- Ricarte, Narayan & Curd (2023) — MAD spin evolution model from GRMHD

---

## License

GPL-3.0. See the repository root for details.
